#!/usr/bin/env bash
# One-shot deploy onto a rented GPU box (RunPod / Vast / any Linux+CUDA).
# Paste this single line into the pod's web terminal or SSH session:
#
#   curl -fsSL https://raw.githubusercontent.com/Thingamajigbo/rvc-cover-maker/main/gpu-bootstrap.sh | bash
#
# It clones (or updates) the repo and starts the server via run.sh, which
# self-bootstraps the whole environment (uv + apt + venvs) on first run.
set -e
REPO=https://github.com/Thingamajigbo/rvc-cover-maker.git
BASE=$([ -d /workspace ] && echo /workspace || echo "$HOME")   # RunPod volume if present
cd "$BASE"

if [ -d rvc-cover-maker/.git ]; then
  cd rvc-cover-maker && git pull --ff-only || true
else
  command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
  git clone "$REPO" rvc-cover-maker && cd rvc-cover-maker
fi

cat <<'EOF'

============================================================
 First run builds the environment (~15-20 min, one-time).
 When it says "Uvicorn running", open your workspace at:

     https://<YOUR-POD-ID>-8000.proxy.runpod.net

 RunPod: expose HTTP port 8000 in the pod config.
 On a server use the file-upload tab (YouTube is IP-blocked).
 Stop the pod when done — it bills per hour.
============================================================

EOF
exec bash run.sh
