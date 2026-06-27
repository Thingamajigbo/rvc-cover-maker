"""FastAPI entrypoint for RVC Cover Maker.

Cover tab is functional (MVP). Training tab arrives in Phase 4-5.
Progress is polled via GET /api/jobs/{id} (simple + robust for a 1-user local tool).
"""
import os
import subprocess
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jobs
import models as models_mod
import train as train_mod
from device import device_report
from lyrics import search_lyrics

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="RVC Cover Maker", version="0.1.0")


class CoverRequest(BaseModel):
    youtube_url: str
    model_name: str
    pitch: int = 0
    index_rate: float = 0.5
    f0_method: str = "rmvpe"
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    output_format: str = "wav"


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/gpu")
def gpu():
    return {**jobs.gpu_status(), "device": device_report()}


@app.get("/api/models")
def get_models():
    return models_mod.list_models()


@app.post("/api/models")
async def upload_model(name: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        uploads = [(f.filename or "file", await f.read()) for f in files]
        return models_mod.save_model(name, uploads)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/jobs")
def create_job(req: CoverRequest):
    if not req.youtube_url.strip():
        raise HTTPException(status_code=400, detail="유튜브 링크를 입력하세요.")
    if not req.model_name.strip():
        raise HTTPException(status_code=400, detail="모델을 선택하세요.")
    job_id = jobs.submit_cover(req.model_dump())
    return {"job_id": job_id}


@app.post("/api/jobs/file")
async def create_job_file(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    pitch: int = Form(0),
    index_rate: float = Form(0.5),
    f0_method: str = Form("rmvpe"),
    protect: float = Form(0.33),
    rms_mix_rate: float = Form(0.25),
    output_format: str = Form("wav"),
):
    """Cover from an uploaded local audio file (no YouTube — works anywhere)."""
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="모델을 선택하세요.")
    uploads = ROOT / "uploads"
    uploads.mkdir(exist_ok=True)
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    src = uploads / f"{uuid.uuid4().hex}{ext}"
    src.write_bytes(await file.read())
    src = _ensure_wav(src)
    params = {
        "youtube_url": str(src), "model_name": model_name, "pitch": pitch,
        "index_rate": index_rate, "f0_method": f0_method, "protect": protect,
        "rms_mix_rate": rms_mix_rate, "output_format": output_format,
    }
    return {"job_id": jobs.submit_cover(params)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if jobs.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return {"cancelled": jobs.cancel(job_id)}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    new_id = jobs.retry(job_id)
    if new_id is None:
        raise HTTPException(status_code=404, detail="재시도할 작업을 찾을 수 없습니다.")
    return {"job_id": new_id}


@app.get("/api/lyrics")
def get_lyrics(title: str, artist: str = ""):
    return search_lyrics(title, artist)


# ---- training ----
DATASETS = ROOT / "datasets"
_AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def _ensure_wav(path: Path) -> Path:
    """Transcode non-wav uploads to wav via ffmpeg. soundfile (data check) can't
    decode mp3/m4a, so normalize everything to wav on arrival."""
    if path.suffix.lower() == ".wav":
        return path
    wav = path.with_suffix(".wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(path), str(wav)],
                       check=True, capture_output=True)
        path.unlink(missing_ok=True)
        return wav
    except Exception:
        return path  # leave original; downstream may still handle it


class TrainRequest(BaseModel):
    model_name: str
    epochs: int = 100
    sample_rate: str = "40k"
    batch_size: int = 4
    f0_method: str = "rmvpe"


def _safe_name(name: str) -> str:
    import re
    n = re.sub(r"[^\w.\-]+", "_", name.strip())
    if not n:
        raise HTTPException(status_code=400, detail="모델 이름이 비어 있습니다.")
    return n


@app.post("/api/train/check")
async def train_check(name: str = Form(...), files: list[UploadFile] = File(...)):
    """Save the uploaded dataset to datasets/<name>/ and analyze it."""
    name = _safe_name(name)
    dest = DATASETS / name
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        fn = (f.filename or "audio").split("/")[-1]
        if not fn.lower().endswith(_AUDIO_EXT):
            continue
        p = dest / fn
        p.write_bytes(await f.read())
        saved.append(_ensure_wav(p))
    if not saved:
        raise HTTPException(status_code=400, detail="오디오 파일(.wav/.mp3 등)이 없습니다.")
    return train_mod.check_dataset(saved)


@app.post("/api/train")
def start_train(req: TrainRequest):
    name = _safe_name(req.model_name)
    if not (DATASETS / name).exists():
        raise HTTPException(status_code=400, detail="먼저 데이터를 업로드·검사하세요.")
    params = req.model_dump()
    params["model_name"] = name
    return {"job_id": jobs.submit_train(params)}


# Generated covers (playback + download).
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


@app.get("/")
def index():
    idx = FRONTEND / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return {"message": "Backend running. Frontend not built yet."}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        app_dir=str(ROOT / "backend"),
    )
