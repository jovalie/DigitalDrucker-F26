#!/usr/bin/env bash
# ComfyUI on GPU0 (V100). Listens on the tailnet IP when Tailscale is up, else localhost.
set -euo pipefail
B=/mnt/raid/shared/comfyui
TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
LISTEN_IP="${COMFY_LISTEN_IP:-${TS_IP:-127.0.0.1}}"
export CUDA_VISIBLE_DEVICES=0
cd "$B/ComfyUI"
echo "starting ComfyUI on ${LISTEN_IP}:8188 (GPU0)"
exec "$B/venv/bin/python" main.py --base-directory "$B" --listen "$LISTEN_IP" --port 8188 "$@"
