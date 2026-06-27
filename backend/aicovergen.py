"""Wrapper around AICoverGen's CLI (src/main.py), run in its own venv via subprocess.

Kept as a subprocess (not an import) because the engine pins conflicting deps
(numpy 1.23.5, gradio 3.39) that can't share our backend's environment.
"""
import fcntl
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from harness import stream_process

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "external" / "AICoverGen"
ENGINE_PY = ENGINE / ".venv" / "bin" / "python"
ENGINE_SRC = ENGINE / "src"
SONG_OUTPUT = ENGINE / "song_output"
_LOCK_FILE = ROOT / "outputs" / ".engine.lock"

# Engine env. Critical on macOS: torch + faiss + sklearn each ship their own
# libomp.dylib; multiple OpenMP runtimes in one process deadlock (0%-CPU hang).
# KMP_DUPLICATE_LIB_OK stops the abort; bounding threads to the 4 perf cores
# shrinks the deadlock window and avoids efficiency-core thrashing.
_ENGINE_ENV = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTHONUNBUFFERED": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
}

# GPU separation: onnxruntime-gpu needs cuDNN 9, which the engine venv lacks but
# the backend venv's torch bundles. Lend it via LD_LIBRARY_PATH (no extra install,
# different soname from the engine torch's cuDNN 8 so they don't clash). No-op
# off-Linux (the dir won't exist), where separation stays on CPU.
_CUDNN = ROOT / ".venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
if _CUDNN.is_dir():
    _ENGINE_ENV["LD_LIBRARY_PATH"] = f"{_CUDNN}:{os.environ.get('LD_LIBRARY_PATH', '')}"


@contextmanager
def engine_lock(lock_path: Path | None = None):
    """Machine-wide exclusive lock: only ONE engine process at a time (cover OR
    train), even across a server restart or a stray CLI run. Prevents the
    concurrent-run contention that deadlocks the duplicate OpenMP runtimes.
    lock_path overridable for test isolation."""
    p = lock_path or _LOCK_FILE
    p.parent.mkdir(exist_ok=True)
    f = open(p, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError("다른 작업(커버/학습)이 이미 실행 중입니다. 끝난 뒤 다시 시도하세요.")
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def list_models() -> list[str]:
    """RVC model folder names available to the engine (rvc_models/<name>/*.pth)."""
    rvc_dir = ENGINE / "rvc_models"
    if not rvc_dir.exists():
        return []
    out = []
    for d in sorted(rvc_dir.iterdir()):
        if d.is_dir() and any(d.glob("*.pth")):
            out.append(d.name)
    return out


def run_cover(
    song_input: str,
    model_name: str,
    pitch: int = 0,
    index_rate: float = 0.5,
    f0_method: str = "rmvpe",
    pitch_all: int = 0,
    output_format: str = "wav",
    protect: float = 0.33,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    on_log: Callable[[str], None] | None = None,
    on_stall: Callable[[], None] | None = None,
    on_resume: Callable[[], None] | None = None,
    cancel_event: "threading.Event | None" = None,
    stall_seconds: float = 240,
    timeout: float = 3600,
) -> Path:
    """Run a full cover and return the path to the generated file.

    song_input: YouTube URL or local audio path.
    protect/filter_radius/rms_mix_rate: vocal-quality knobs (the UI's 고급 옵션).
    on_log: callback per output line (streams tqdm \\r ticks too).
    on_stall/on_resume: watchdog callbacks for output going quiet/resuming.
    cancel_event: set it to abort; raises harness.EngineCancelled.
    Raises RuntimeError on engine failure, EngineTimeout on hard timeout.
    """
    if not ENGINE_PY.exists():
        raise RuntimeError(f"Engine venv missing at {ENGINE_PY}. Run setup.sh.")

    cmd = [
        str(ENGINE_PY), "main.py",
        "-i", song_input,
        "-dir", model_name,
        "-p", str(pitch),
        "-ir", str(index_rate),
        "-palgo", f0_method,
        "-pall", str(pitch_all),
        "-pro", str(protect),
        "-fr", str(filter_radius),
        "-rms", str(rms_mix_rate),
        "-oformat", output_format,
    ]
    env = {**os.environ, **_ENGINE_ENV}

    last_lines: list[str] = []

    def _line(l: str):
        last_lines.append(l)
        if len(last_lines) > 600:
            del last_lines[:-600]
        if on_log:
            on_log(l)

    with engine_lock():
        code = stream_process(
            cmd, cwd=ENGINE_SRC, env=env,
            on_line=_line, on_stall=on_stall, on_resume=on_resume,
            cancel_event=cancel_event, done_marker="Cover generated at",
            stall_seconds=stall_seconds, timeout=timeout,
        )
    if code != 0:
        raise RuntimeError(f"AICoverGen failed (exit {code}):\n" + "\n".join(last_lines[-15:]))

    cover = _find_cover_path(last_lines, model_name, output_format)
    if cover is None or not cover.exists():
        raise RuntimeError("Cover finished but output file not found.\n" + "\n".join(last_lines[-15:]))
    return cover


def _find_cover_path(lines: list[str], model_name: str, output_format: str) -> Path | None:
    """Locate the final cover. main.py prints '[+] Cover generated at <path>'."""
    for line in reversed(lines):
        if "Cover generated at" in line:
            return Path(line.split("Cover generated at", 1)[1].strip())
    # Fallback: newest matching file under song_output.
    pattern = f"*({model_name} Ver).{output_format}"
    matches = sorted(SONG_OUTPUT.glob(f"**/{pattern}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None
