"""RVC voice-model training wrapper.

Drives RVC-WebUI's training scripts (preprocess → f0 → feature → train → index)
as subprocesses in the SHARED engine venv (same one as cover; libomp already
deduped there, so no OpenMP deadlock). Honest note: real training needs a GPU —
on CPU this works but is slow. The pipeline + progress are GPU-ready as-is.
"""
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from aicovergen import ENGINE_PY, _ENGINE_ENV, engine_lock
from harness import stream_process

ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = ROOT / "external" / "RVC-WebUI"
DATASETS = ROOT / "datasets"
COVER_MODELS = ROOT / "external" / "AICoverGen" / "rvc_models"  # where trained models land

SR_NUM = {"40k": 40000, "48k": 48000, "32k": 32000}


def _device() -> str:
    """'cuda' on a GPU server, else 'cpu' (MPS deadlocks the engine, so never mps)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------- data check (no engine; fully testable) ----------------
def check_dataset(file_paths: list[Path]) -> dict:
    """Inspect training audio: total length, silence ratio, clipping, sample rate.
    Returns warnings the UI shows before the user commits to a long train."""
    total_sec = 0.0
    silence_sec = 0.0
    clipped = 0
    total_samples = 0
    srs = set()
    n_files = 0

    for p in file_paths:
        try:
            data, sr = sf.read(str(p), always_2d=True)
        except Exception:
            continue
        n_files += 1
        srs.add(sr)
        mono = data.mean(axis=1)
        dur = len(mono) / sr
        total_sec += dur
        # silence: |amp| < 0.01 (~ -40 dBFS)
        silence_sec += float(np.sum(np.abs(mono) < 0.01)) / sr
        # clipping: samples at/over full scale
        clipped += int(np.sum(np.abs(mono) >= 0.999))
        total_samples += len(mono)

    silence_ratio = (silence_sec / total_sec) if total_sec else 0.0
    clip_ratio = (clipped / total_samples) if total_samples else 0.0

    warnings = []
    if n_files == 0:
        warnings.append("읽을 수 있는 오디오가 없습니다.")
    if total_sec < 60:
        warnings.append(f"데이터가 짧습니다 ({total_sec:.0f}초). 10분 이상 권장.")
    if silence_ratio > 0.35:
        warnings.append(f"무음 비율이 높습니다 ({silence_ratio*100:.0f}%). 무음을 잘라내세요.")
    if clip_ratio > 0.005:
        warnings.append(f"클리핑이 감지됩니다 ({clip_ratio*100:.1f}%). 녹음 음량을 낮추세요.")
    if len(srs) > 1:
        warnings.append(f"샘플레이트가 섞여 있습니다 {sorted(srs)}.")

    return {
        "ok": n_files > 0,
        "files": n_files,
        "total_seconds": round(total_sec, 1),
        "silence_ratio": round(silence_ratio, 3),
        "clip_ratio": round(clip_ratio, 4),
        "sample_rates": sorted(srs),
        "warnings": warnings,
    }


# ---------------- training pipeline ----------------
_TRAIN_STEPS = {
    "preprocess": (0.05, "데이터 전처리·슬라이싱"),
    "f0": (0.15, "피치 추출 (RMVPE)"),
    "feature": (0.30, "feature 추출"),
    "train": (0.40, "학습"),
    "index": (0.92, "인덱스 생성"),
}
_TRAIN_LO, _TRAIN_HI = 0.40, 0.90
_EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")


def _step(args, cwd, env, on_log, cancel_event, label_progress):
    """Run one training subprocess; raise on nonzero exit."""
    last: list[str] = []

    def _line(l):
        last.append(l)
        if on_log:
            on_log(l)

    code = stream_process(
        [str(ENGINE_PY)] + args, cwd=str(cwd), env=env,
        on_line=_line, cancel_event=cancel_event,
        stall_seconds=600, timeout=24 * 3600, done_marker=None,
    )
    if code != 0:
        raise RuntimeError(f"학습 단계 실패 ({label_progress}, exit {code}):\n" + "\n".join(last[-15:]))


def run_training(
    model_name: str,
    epochs: int = 100,
    sample_rate: str = "40k",
    batch_size: int = 4,
    f0_method: str = "rmvpe",
    on_log=None,
    on_progress=None,   # on_progress(frac, label)
    on_stall=None,
    on_resume=None,
    cancel_event=None,
) -> dict:
    """Full training run. Dataset must already be at datasets/<model_name>/.
    Returns {name, pth, index} registered into the cover engine's models dir."""
    if not RVC_DIR.exists():
        raise RuntimeError(f"RVC-WebUI 미설치: {RVC_DIR}. setup.sh 실행 필요.")
    dataset_dir = DATASETS / model_name
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        raise RuntimeError(f"데이터셋이 없습니다: {dataset_dir}")

    env = {**__import__("os").environ, **_ENGINE_ENV}
    exp = re.sub(r"[^\w.\-]+", "_", model_name)
    logs = f"logs/{exp}"
    sr_num = SR_NUM.get(sample_rate, 40000)
    version = "v2"

    def prog(frac, label):
        if on_progress:
            on_progress(frac, label)

    (RVC_DIR / logs).mkdir(parents=True, exist_ok=True)  # scripts don't create their own log dir

    with engine_lock():
        # 1) preprocess
        prog(*_TRAIN_STEPS["preprocess"])
        _step(["infer/modules/train/preprocess.py", str(dataset_dir), str(sr_num), "4", logs, "False", "3.0"],
              RVC_DIR, env, on_log, cancel_event, "전처리")

        # 2) extract f0 (rmvpe, cpu)
        prog(*_TRAIN_STEPS["f0"])
        _step(["infer/modules/train/extract/extract_f0_rmvpe.py", "1", "0", "0", logs, "False"],
              RVC_DIR, env, on_log, cancel_event, "피치추출")

        # 3) extract features (GPU when available; CPU on Mac/no-CUDA)
        prog(*_TRAIN_STEPS["feature"])
        _step(["infer/modules/train/extract_feature_print.py", _device(), "1", "0", "0", logs, version, "False"],
              RVC_DIR, env, on_log, cancel_event, "feature추출")

        # 4) train — first build filelist.txt + config.json (what click_train does;
        #    train.py's get_hparams() needs both).
        _prepare_train_files(exp, sample_rate, version)
        prog(*_TRAIN_STEPS["train"])
        last: list[str] = []

        def _train_line(l):
            last.append(l)
            if on_log:
                on_log(l)
            m = _EPOCH_RE.search(l)
            if m and on_progress:
                ep = min(int(m.group(1)), epochs)
                on_progress(_TRAIN_LO + (_TRAIN_HI - _TRAIN_LO) * (ep / max(epochs, 1)),
                            f"학습 {ep}/{epochs} epoch")

        g = str((RVC_DIR / "assets/pretrained_v2/f0G40k.pth"))
        d = str((RVC_DIR / "assets/pretrained_v2/f0D40k.pth"))
        code = stream_process(
            [str(ENGINE_PY), "infer/modules/train/train.py", "-e", exp, "-sr", sample_rate,
             "-f0", "1", "-bs", str(batch_size), "-g", "0", "-te", str(epochs), "-se", "5",
             "-pg", g, "-pd", d, "-l", "1", "-c", "0", "-sw", "1", "-v", version],
            cwd=str(RVC_DIR), env=env, on_line=_train_line,
            on_stall=on_stall, on_resume=on_resume, cancel_event=cancel_event,
            stall_seconds=1200, timeout=72 * 3600,
        )
        if code != 0:
            raise RuntimeError("학습 실패 (exit %d):\n%s" % (code, "\n".join(last[-20:])))

        # 5) build faiss index
        prog(*_TRAIN_STEPS["index"])
        index_path = _build_index(exp, version, on_log)

    # register output into the cover engine's models dir
    pth_src = RVC_DIR / "assets" / "weights" / f"{exp}.pth"
    if not pth_src.exists():
        raise RuntimeError(f"학습은 끝났지만 .pth가 없습니다: {pth_src}")
    dest = COVER_MODELS / exp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pth_src, dest / f"{exp}.pth")
    if index_path and index_path.exists():
        shutil.copy2(index_path, dest / index_path.name)
    if on_progress:
        on_progress(1.0, "완료")
    return {"name": exp, "pth": f"{exp}.pth", "index": index_path.name if index_path else None}


def _prepare_train_files(exp: str, sr: str, version: str):
    """Replicate infer-web's click_train setup: pair gt/feature/f0 into
    filelist.txt (+ 2 mute reference samples) and drop config.json. train.py's
    get_hparams() reads both."""
    base = RVC_DIR / "logs" / exp
    gt, feat = base / "0_gt_wavs", base / ("3_feature256" if version == "v1" else "3_feature768")
    f0, f0nsf = base / "2a_f0", base / "2b-f0nsf"
    fea_dim = 256 if version == "v1" else 768
    nd = str(RVC_DIR)

    def stems(d):
        return set(p.name.split(".")[0] for p in d.iterdir() if p.is_file())

    names = stems(gt) & stems(feat) & stems(f0) & stems(f0nsf)
    opt = [f"{gt}/{n}.wav|{feat}/{n}.npy|{f0}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0" for n in names]
    mute = (f"{nd}/logs/mute/0_gt_wavs/mute{sr}.wav|{nd}/logs/mute/3_feature{fea_dim}/mute.npy|"
            f"{nd}/logs/mute/2a_f0/mute.wav.npy|{nd}/logs/mute/2b-f0nsf/mute.wav.npy|0")
    opt += [mute, mute]
    random.shuffle(opt)
    (base / "filelist.txt").write_text("\n".join(opt))

    cfg = f"v1/{sr}.json" if (version == "v1" or sr == "40k") else f"v2/{sr}.json"
    src = RVC_DIR / "configs" / cfg
    if not src.exists():
        src = next((RVC_DIR / "configs").glob(f"v*/{sr}.json"), None)
    if not src:
        raise RuntimeError(f"학습 config를 찾을 수 없습니다: {sr}")
    shutil.copy2(src, base / "config.json")


def _build_index(exp: str, version: str, on_log=None) -> Path | None:
    """Build the faiss retrieval index from extracted features (CPU)."""
    import faiss

    feat_dir = RVC_DIR / logs_feature(exp, version)
    if not feat_dir.exists():
        return None
    npys = [np.load(str(feat_dir / f)) for f in sorted(p.name for p in feat_dir.glob("*.npy"))]
    if not npys:
        return None
    big = np.concatenate(npys, 0)
    if big.shape[0] > 2e5:  # subsample very large sets (matches RVC-WebUI)
        big = big[np.random.choice(big.shape[0], int(1e5), replace=False)]
    n_ivf = min(int(16 * math.sqrt(big.shape[0])), big.shape[0] // 39) or 1
    dim = big.shape[1]
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(big)
    for i in range(0, big.shape[0], 8192):
        index.add(big[i:i + 8192])
    out = RVC_DIR / f"logs/{exp}" / f"added_IVF{n_ivf}_Flat_nprobe_1_{exp}_{version}.index"
    faiss.write_index(index, str(out))
    if on_log:
        on_log(f"[index] {out.name} ({big.shape[0]} vectors)")
    return out


def logs_feature(exp: str, version: str) -> str:
    return f"logs/{exp}/3_feature768" if version == "v2" else f"logs/{exp}/3_feature256"
