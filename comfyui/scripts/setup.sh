#!/usr/bin/env bash
# One-time ComfyUI install for GPU0 (V100, sm_70 -> CUDA 12.4 wheels).
set -euo pipefail
B=/mnt/raid/shared/comfyui
cd "$B"
echo "== cloning ComfyUI =="
[ -d ComfyUI/.git ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI ComfyUI
echo "== venv =="
python3 -m venv "$B/venv"
"$B/venv/bin/pip" install -q --upgrade pip wheel
echo "== torch (cu124) =="
"$B/venv/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "== comfy requirements =="
"$B/venv/bin/pip" install -r ComfyUI/requirements.txt
echo "== GPU check on GPU0 =="
CUDA_VISIBLE_DEVICES=0 "$B/venv/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda"); y = x @ x; torch.cuda.synchronize()
print("matmul OK, sum", float(y.sum()))
PY
echo "SETUP DONE"
