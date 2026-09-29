#!/bin/bash
# One-time environment setup on a Laguna LOGIN node (internet required for pip/HF).
#
#   DRUCKER_ROOT=/project/<PI>_<id>/drucker bash laguna/setup_env.sh
#
# Creates: $DRUCKER_ROOT/envs/whisper-gpu (venv), $DRUCKER_ROOT/hf-cache (large-v3 weights)
set -euo pipefail
: "${DRUCKER_ROOT:?set DRUCKER_ROOT=/project/<PI>_<id>/drucker}"

mkdir -p "$DRUCKER_ROOT"/{code,media,envs,hf-cache,logs,runs}
module load python/3.11 2>/dev/null || true

python3 -m venv "$DRUCKER_ROOT/envs/whisper-gpu"
# shellcheck disable=SC1091
source "$DRUCKER_ROOT/envs/whisper-gpu/bin/activate"
pip install --upgrade pip wheel
pip install "faster-whisper==1.2.1" nvidia-cublas-cu12 nvidia-cudnn-cu12

export HF_HOME="$DRUCKER_ROOT/hf-cache"
python3 - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Systran/faster-whisper-large-v3")
print("large-v3 cached at:", path)
PY

echo
echo "env ready: $DRUCKER_ROOT/envs/whisper-gpu"
echo "next: bash laguna/sync_push.sh   (from the container)  then sbatch"
