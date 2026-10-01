#!/usr/bin/env bash
# Drucker TTS webapp launcher — listens on the tailnet IP (falls back to localhost).
set -euo pipefail
B=/mnt/raid/shared/tts
export DRUCKER_TTS_BASE="$B"
export COSYVOICE_DIR="$B/CosyVoice"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TS_IP=""  # Tailscale removed 2026-09-30; loopback only
HOST="${TTS_HOST:-${TS_IP:-127.0.0.1}}"
PORT="${TTS_PORT:-8190}"
echo "drucker-tts: http://${HOST}:${PORT} (GPU ${CUDA_VISIBLE_DEVICES})"
exec "$B/venv/bin/python" "$B/app.py" --host "$HOST" --port "$PORT" "$@"
