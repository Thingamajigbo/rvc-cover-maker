"""In-memory job queue + single worker. Serializes engine use (one GPU/CPU at a
time) so jobs never collide. Adds harness-grade robustness: per-job cancel,
stall detection, hard timeout, and fine-grained progress through the slow RVC step.
"""
import queue as stdqueue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from aicovergen import run_cover
from harness import EngineCancelled, EngineTimeout

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Engine stdout substring -> (progress fraction, friendly Korean label).
_COVER_STEPS = [
    ("Downloading song", 0.05, "원곡 다운로드"),
    ("Separating Vocals from Instrumental", 0.15, "보컬/MR 분리"),
    ("Separating Main Vocals", 0.35, "메인/백업 보컬 분리"),
    ("Applying DeReverb", 0.55, "디리버브"),
    ("Converting voice using RVC", 0.80, "RVC 음성 변환"),
    ("Applying audio effects", 0.96, "이펙트 적용"),
    ("Combining", 0.98, "합치기"),
    ("Cover generated at", 1.00, "완료"),
]
_RVC_START, _RVC_END = 0.80, 0.95  # band the per-segment tqdm fills in
_SEGMENT_RE = re.compile(r"(\d+)/(\d+)\s*\[")  # tqdm "3/6 ["

STALL_SECONDS = 240
JOB_TIMEOUT = 3600

_jobs: dict[str, dict] = {}
_params: dict[str, dict] = {}
_cancels: dict[str, threading.Event] = {}
_lock = threading.Lock()
_q: "stdqueue.Queue" = stdqueue.Queue()
_current = {"id": None, "kind": None}


def get_job(jid: str) -> dict | None:
    with _lock:
        return dict(_jobs[jid]) if jid in _jobs else None


def list_jobs() -> list[dict]:
    with _lock:
        return [dict(j) for j in _jobs.values()]


def gpu_status() -> dict:
    with _lock:
        return {"busy": _current["id"] is not None, "current": _current["kind"]}


def _new_job(kind: str, params: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {
            "id": jid, "kind": kind, "status": "queued", "step": "대기 중",
            "progress": 0.0, "logs": [], "result": None, "error": None,
            "stalled": False, "last_activity": time.time(),
        }
        _params[jid] = dict(params)
        _cancels[jid] = threading.Event()
    _q.put(jid)
    return jid


def submit_cover(params: dict) -> str:
    return _new_job("cover", params)


def submit_train(params: dict) -> str:
    return _new_job("train", params)


def cancel(jid: str) -> bool:
    with _lock:
        job = _jobs.get(jid)
        ev = _cancels.get(jid)
        if not job or job["status"] in ("done", "error", "cancelled"):
            return False
        if job["status"] == "queued":
            # Not started yet — mark cancelled; the worker will skip it.
            job["status"] = "cancelled"
            job["step"] = "취소됨"
    if ev:
        ev.set()
    return True


def retry(jid: str) -> str | None:
    with _lock:
        params = _params.get(jid)
    return submit_cover(params) if params else None


def _touch(job: dict):
    job["last_activity"] = time.time()
    job["stalled"] = False


def _on_log(job: dict, line: str):
    with _lock:
        _touch(job)
        job["logs"].append(line)
        if len(job["logs"]) > 400:
            del job["logs"][:-400]
        if job["kind"] != "cover":
            return  # train progress is driven by on_progress, not log parsing
        for key, frac, label in _COVER_STEPS:
            if key in line:
                job["progress"], job["step"] = frac, label
        # Fine progress inside the long RVC step: fill 0.80..0.95 from "n/m".
        if job["step"] == "RVC 음성 변환":
            m = _SEGMENT_RE.search(line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                if total:
                    frac = _RVC_START + (_RVC_END - _RVC_START) * min(cur / total, 1.0)
                    job["progress"] = max(job["progress"], frac)


def _run_cover(job: dict, params: dict, cancel_event: threading.Event):
    out = run_cover(
        song_input=params["youtube_url"],
        model_name=params["model_name"],
        pitch=int(params.get("pitch", 0)),
        index_rate=float(params.get("index_rate", 0.5)),
        f0_method=params.get("f0_method", "rmvpe"),
        protect=float(params.get("protect", 0.33)),
        rms_mix_rate=float(params.get("rms_mix_rate", 0.25)),
        output_format=params.get("output_format", "wav"),
        on_log=lambda line: _on_log(job, line),
        on_stall=lambda: _set_stalled(job, True),
        on_resume=lambda: _set_stalled(job, False),
        cancel_event=cancel_event,
        stall_seconds=STALL_SECONDS,
        timeout=JOB_TIMEOUT,
    )
    # Serve under a URL-safe name (jobid.ext); keep the human title for display/download.
    dest = OUTPUTS / f"{job['id']}{out.suffix}"
    shutil.copy2(out, dest)
    download_name = re.sub(r"[^\w.\- ()]+", "_", out.name).strip()
    with _lock:
        job["result"] = {
            "filename": dest.name, "url": f"/outputs/{dest.name}",
            "title": out.stem, "download_name": download_name,
        }


def _run_train(job: dict, params: dict, cancel_event: threading.Event):
    import train  # lazy: pulls soundfile/faiss only when training is used

    def on_progress(frac, label):
        with _lock:
            job["progress"], job["step"] = frac, label

    result = train.run_training(
        model_name=params["model_name"],
        epochs=int(params.get("epochs", 100)),
        sample_rate=params.get("sample_rate", "40k"),
        batch_size=int(params.get("batch_size", 4)),
        f0_method=params.get("f0_method", "rmvpe"),
        on_log=lambda line: _on_log(job, line),
        on_progress=on_progress,
        on_stall=lambda: _set_stalled(job, True),
        on_resume=lambda: _set_stalled(job, False),
        cancel_event=cancel_event,
    )
    with _lock:
        job["result"] = result


def _set_stalled(job: dict, value: bool):
    with _lock:
        job["stalled"] = value


def _worker():
    while True:
        jid = _q.get()
        with _lock:
            job = _jobs[jid]
            if job["status"] == "cancelled":  # cancelled while queued
                _q.task_done()
                continue
            params = _params[jid]
            cancel_event = _cancels[jid]
            job["status"] = "running"
            _touch(job)
            _current.update(id=jid, kind=job["kind"])
        try:
            if job["kind"] == "train":
                _run_train(job, params, cancel_event)
            else:
                _run_cover(job, params, cancel_event)
            with _lock:
                job["status"], job["progress"], job["step"], job["stalled"] = "done", 1.0, "완료", False
        except EngineCancelled:
            with _lock:
                job["status"], job["step"], job["stalled"] = "cancelled", "취소됨", False
        except EngineTimeout:
            with _lock:
                job["status"], job["error"], job["stalled"] = "error", f"시간 초과({JOB_TIMEOUT}s) — 작업이 너무 오래 걸려 중단했습니다.", False
        except Exception as e:  # surface engine errors to the UI
            with _lock:
                job["status"], job["error"], job["stalled"] = "error", str(e), False
        finally:
            with _lock:
                _current.update(id=None, kind=None)
            _q.task_done()


threading.Thread(target=_worker, daemon=True, name="job-worker").start()
