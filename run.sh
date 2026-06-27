#!/usr/bin/env bash
# Start the web server. Use this instead of calling uvicorn directly on a server.
#
# Why the symlink: on RunPod the project lives on a network volume (MooseFS) that
# intermittently fails large file writes — torch.save and libsndfile raise opaque
# errors mid-write (the cover's separated stems are the trigger). Redirecting the
# engine's scratch dir to LOCAL disk fixes it. Intermediates are disposable; the
# final covers are copied to outputs/ on the persistent volume by the job worker.
# Harmless on a normal local disk (Mac / local GPU) — it just uses /tmp.
set -e
cd "$(dirname "$0")"

SCRATCH="${RVC_SCRATCH:-/tmp/rvc_scratch}"

# Cover scratch (separated stems) -> local disk.
mkdir -p "$SCRATCH/song_output"
ln -sfn "$SCRATCH/song_output" external/AICoverGen/song_output

# Training scratch (features/checkpoints are GBs) -> local disk. Preserve the
# shipped mute reference samples (every training run references them).
RVC="external/RVC-WebUI"
if [ -d "$RVC" ]; then
  if [ ! -L "$RVC/logs" ]; then
    [ -d "$RVC/logs/mute" ] && cp -rn "$RVC/logs/mute" "$RVC/mute_ref" 2>/dev/null || true
    rm -rf "$RVC/logs"            # was on the network volume; free it
  fi
  mkdir -p "$SCRATCH/train_logs"
  [ -d "$RVC/mute_ref" ] && cp -rn "$RVC/mute_ref" "$SCRATCH/train_logs/mute" 2>/dev/null || true
  ln -sfn "$SCRATCH/train_logs" "$RVC/logs"
fi

exec .venv/bin/python -m uvicorn main:app --app-dir backend \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
