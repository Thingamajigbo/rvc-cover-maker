"""Training data-check tests — no engine/GPU needed. Generates synthetic audio
and verifies the silence/clipping/length analysis the UI shows before training."""
import io
import shutil
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import train as train_mod
from main import app, DATASETS

client = TestClient(app)


def _wav(tmp_path, name, samples, sr=40000):
    p = tmp_path / name
    sf.write(str(p), samples.astype(np.float32), sr)
    return p


def test_check_clean_audio(tmp_path):
    t = np.linspace(0, 3, 3 * 40000, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    r = train_mod.check_dataset([_wav(tmp_path, "a.wav", tone)])
    assert r["ok"] and r["files"] == 1
    assert 2.9 < r["total_seconds"] < 3.1
    assert r["silence_ratio"] < 0.1
    assert r["clip_ratio"] == 0.0


def test_check_flags_silence(tmp_path):
    silent = np.zeros(2 * 40000)
    r = train_mod.check_dataset([_wav(tmp_path, "s.wav", silent)])
    assert r["silence_ratio"] > 0.9
    assert any("무음" in w for w in r["warnings"])


def test_check_flags_clipping(tmp_path):
    clip = np.ones(40000)  # full-scale -> clipped
    r = train_mod.check_dataset([_wav(tmp_path, "c.wav", clip)])
    assert r["clip_ratio"] > 0.5
    assert any("클리핑" in w for w in r["warnings"])


def test_check_short_dataset_warns(tmp_path):
    tone = 0.3 * np.sin(np.linspace(0, 5, 5 * 40000))
    r = train_mod.check_dataset([_wav(tmp_path, "a.wav", tone)])
    assert any("짧" in w for w in r["warnings"])  # <60s


def test_train_check_endpoint_and_dataset_saved():
    name = f"_pt_{uuid.uuid4().hex[:6]}"
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(np.linspace(0, 2, 80000))).astype(np.float32), 40000, format="WAV")
    buf.seek(0)
    r = client.post("/api/train/check", data={"name": name},
                    files=[("files", ("voice.wav", buf.read(), "audio/wav"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"] == 1 and body["ok"]
    assert (DATASETS / name / "voice.wav").exists()
    shutil.rmtree(DATASETS / name, ignore_errors=True)


def test_train_requires_dataset_first():
    r = client.post("/api/train", json={"model_name": f"_missing_{uuid.uuid4().hex[:6]}"})
    assert r.status_code == 400


def test_train_check_rejects_non_audio():
    name = f"_pt_{uuid.uuid4().hex[:6]}"
    r = client.post("/api/train/check", data={"name": name},
                    files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400
    shutil.rmtree(DATASETS / name, ignore_errors=True)
