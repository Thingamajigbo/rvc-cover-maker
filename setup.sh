#!/usr/bin/env bash
# RVC Cover Maker — environment setup.
# Verified on: Apple Silicon (M5), macOS, Python 3.10 via uv.
# Strategy: TWO isolated venvs (backend + engine) because their deps conflict.
# Re-runnable: each step is skip-safe-ish; delete .venv / external to force a clean rebuild.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

command -v uv >/dev/null || { echo "ERROR: 'uv' not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARN: ffmpeg not found (brew install ffmpeg)"
command -v sox >/dev/null || echo "WARN: sox not found (brew install sox) — required by AICoverGen"

echo "========================================"
echo " 1/4  Backend venv (.venv, Python 3.10)"
echo "========================================"
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "========================================"
echo " 2/4  Clone AICoverGen engine"
echo "========================================"
if [ ! -d external/AICoverGen/.git ]; then
  git clone --depth 1 https://github.com/SociallyIneptWeeb/AICoverGen.git external/AICoverGen
fi

echo "========================================"
echo " 3/4  Engine venv (external/AICoverGen/.venv)"
echo "========================================"
# Mac-adapted: the upstream requirements.txt pins CUDA-only torch / onnxruntime_gpu
# and a stale yt-dlp. We substitute MPS/CPU builds and rebuild fairseq from source.
cd external/AICoverGen
V=.venv/bin/python
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv

# fairseq 0.12.2 build prereqs. Order matters: numpy+cython MUST exist before fairseq,
# and we need a real pip in the venv (uv's builder fails on fairseq's C++ glob).
uv pip install --python $V "pip<24.1" "setuptools<60" wheel
uv pip install --python $V "numpy==1.23.5" cython "torch==2.2.2" "torchaudio==2.2.2"
# fairseq 0.12.2's PyPI sdist is missing C++ sources (balanced_assignment.cpp),
# so a clean build fails. Build from the git tag instead (has all sources).
$V -c "import fairseq" 2>/dev/null && echo "fairseq already installed" || \
  $V -m pip install --no-build-isolation "fairseq @ git+https://github.com/facebookresearch/fairseq.git@v0.12.2"

# Remaining deps, Mac-substituted:
#   onnxruntime_gpu -> onnxruntime (CPU)   torch+cu118 -> 2.2.2 (above)
#   faiss-cpu pinned 1.8.0 (latest needs numpy>=2, which breaks np.int in the code)
#   gradio_client pinned to match gradio 3.39.0   yt-dlp bumped to latest
uv pip install --python $V \
  "gradio==3.39.0" "gradio_client==0.3.0" \
  "librosa==0.9.1" "scipy==1.11.1" "soundfile==0.12.1" \
  "praat-parselmouth>=0.4.2" "pedalboard==0.7.7" "pydub==0.25.1" "pyworld==0.3.4" \
  "Requests==2.31.0" "torchcrepe==0.0.20" "yt-dlp" "sox==1.4.1" \
  "ffmpeg-python>=0.2.0" "deemix" "faiss-cpu==1.8.0" "tqdm==4.65.0"
# MDX vocal separation engine: GPU build on CUDA boxes (cuDNN 9 is lent from the
# backend venv's torch via LD_LIBRARY_PATH at runtime — see aicovergen.py), CPU otherwise.
if command -v nvidia-smi >/dev/null 2>&1; then
  uv pip install --python $V onnxruntime-gpu
else
  uv pip install --python $V onnxruntime
fi
# Re-pin numpy last — some of the above pull it forward off 1.23.5.
uv pip install --python $V "numpy==1.23.5"

echo "========================================"
echo " 3b   Apply Mac (MPS/CPU) device patches"
echo "========================================"
# Upstream hardcodes cuda:0 in several places; patch redirects to MPS/CPU.
git apply --reverse --check "$ROOT/patches/aicovergen-mac.patch" 2>/dev/null \
  && echo "patches already applied, skipping" \
  || git apply "$ROOT/patches/aicovergen-mac.patch"

echo "========================================"
echo " 3c   Deduplicate OpenMP runtimes (macOS deadlock fix)"
echo "========================================"
# torch, faiss, sklearn each bundle their own libomp.dylib. Multiple OpenMP
# runtimes in one process deadlock torch's CPU conv at an OMP barrier (0%-CPU
# hang in the RVC step). All are LLVM OpenMP 5.0.0 / ABI-compatible, so we
# point faiss & sklearn at torch's copy -> a single runtime loads.
SP=".venv/lib/python3.10/site-packages"
TORCH_OMP="$(cd "$SP" && pwd)/torch/.dylibs/libomp.dylib"
for p in faiss sklearn; do
  d="$SP/$p/.dylibs/libomp.dylib"
  if [ -f "$d" ] && [ ! -L "$d" ]; then
    mv "$d" "$d.bak" && ln -s "$TORCH_OMP" "$d" && echo "  $p libomp -> torch (deduped)"
  fi
done

echo "========================================"
echo " 4/4  Download base models (hubert, rmvpe, MDX)"
echo "========================================"
$V src/download_models.py

cd "$ROOT"

echo "========================================"
echo " 5/5  Training engine (RVC-WebUI) — shares the AICoverGen venv"
echo "========================================"
# Training reuses the AICoverGen venv (same torch/fairseq/faiss + libomp dedup).
# Extra deps: PyAV (audio loader) + tensorboard/X (train.py's SummaryWriter).
if [ ! -d external/RVC-WebUI/.git ]; then
  git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git external/RVC-WebUI
fi
uv pip install --python external/AICoverGen/.venv/bin/python av tensorboard tensorboardX
# Mac patch: extract_f0_rmvpe.py hardcodes device="cuda".
git -C external/RVC-WebUI apply --reverse --check "$ROOT/patches/rvc-webui-mac.patch" 2>/dev/null \
  && echo "rvc-webui patch already applied" \
  || git -C external/RVC-WebUI apply "$ROOT/patches/rvc-webui-mac.patch"
# Pretrained base models for 40k v2 training (+ reuse hubert/rmvpe from cover engine).
mkdir -p external/RVC-WebUI/assets/pretrained_v2 external/RVC-WebUI/assets/hubert external/RVC-WebUI/assets/rmvpe
cp -n external/AICoverGen/rvc_models/hubert_base.pt external/RVC-WebUI/assets/hubert/ 2>/dev/null || true
cp -n external/AICoverGen/rvc_models/rmvpe.pt external/RVC-WebUI/assets/rmvpe/ 2>/dev/null || true
HF="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
for m in f0G40k.pth f0D40k.pth; do
  [ -f "external/RVC-WebUI/assets/pretrained_v2/$m" ] || curl -sL -o "external/RVC-WebUI/assets/pretrained_v2/$m" "$HF/pretrained_v2/$m"
done

echo
echo "DONE. Start the app:  .venv/bin/python -m uvicorn main:app --app-dir backend --port 8000"
