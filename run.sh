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
mkdir -p "$SCRATCH/song_output"
ln -sfn "$SCRATCH/song_output" external/AICoverGen/song_output

exec .venv/bin/python -m uvicorn main:app --app-dir backend \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
