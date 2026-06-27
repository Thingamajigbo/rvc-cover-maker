#!/usr/bin/env bash
# Start the web server. Use this instead of calling uvicorn directly on a server.
#
# Two offloads, both because RunPod's /workspace is a MooseFS network volume that
# (a) intermittently stalls large reads/writes — torch.save, libsndfile, and even
# reading the pretrained models mid-training hang in FUSE — and (b) is quota-tight.
#
#  1. SCRATCH: disposable intermediates (cover stems, training features/checkpoints)
#     -> local disk. Final outputs are copied back to outputs/ on the volume.
#  2. LOCAL:   the big read-heavy, *redownloadable* trees (both venvs + the RVC base
#     /pretrained models) -> local disk, symlinked from the repo. The volume then
#     holds only code + trained voices + outputs. Container disk is wiped on pod
#     stop, so we rebuild (setup.sh) when a venv is missing — one-time ~15 min per
#     pod start. Persistent user data (rvc_models/<voice>, outputs) stays on /workspace.
#
# Harmless on a normal local disk (Mac / local GPU): targets just live elsewhere.
set -e
cd "$(dirname "$0")"

LOCAL="${RVC_LOCAL:-/root/rvc-local}"
SCRATCH="${RVC_SCRATCH:-/tmp/rvc_scratch}"

# --- Offload 2: relocate big redownloadable trees to local disk ---------------
# Replace each repo path with a symlink into $LOCAL (deleting any real dir/old copy
# sitting on the network volume — that's the space we're reclaiming).
relocate() {  # $1 = repo path (becomes a symlink), $2 = name under $LOCAL
  local link="$1" tgt="$LOCAL/$2"
  mkdir -p "$tgt"
  [ -L "$link" ] || rm -rf "$link"
  ln -sfn "$tgt" "$link"
}
relocate ".venv" "venv-backend"
relocate "external/AICoverGen/.venv" "venv-engine"
relocate "external/RVC-WebUI/assets" "rvc-webui-assets"

# Rebuild venvs if local disk was wiped (fresh pod). setup.sh installs through the
# symlinks above, so everything lands on local disk.
if [ ! -x .venv/bin/python ] || [ ! -x external/AICoverGen/.venv/bin/python ]; then
  echo ">>> venv missing on local disk — running setup.sh (one-time, ~15 min) <<<"
  bash setup.sh
fi

# --- Offload 1: scratch dirs to local disk ------------------------------------
mkdir -p "$SCRATCH/song_output"
ln -sfn "$SCRATCH/song_output" external/AICoverGen/song_output

RVC="external/RVC-WebUI"
if [ -d "$RVC" ]; then
  if [ ! -L "$RVC/logs" ]; then
    [ -d "$RVC/logs/mute" ] && cp -rn "$RVC/logs/mute" "$RVC/mute_ref" 2>/dev/null || true
    rm -rf "$RVC/logs"
  fi
  mkdir -p "$SCRATCH/train_logs"
  [ -d "$RVC/mute_ref" ] && cp -rn "$RVC/mute_ref" "$SCRATCH/train_logs/mute" 2>/dev/null || true
  ln -sfn "$SCRATCH/train_logs" "$RVC/logs"
fi

exec .venv/bin/python -m uvicorn main:app --app-dir backend \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
