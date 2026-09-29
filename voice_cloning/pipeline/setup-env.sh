#!/bin/bash
# Speaker-ID environment for the Drucker corpus (V100 / sm_70 compatible torch).
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$B"
uv venv venv --python 3.12
uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchaudio==2.7.1"
uv pip install --python venv/bin/python speechbrain soundfile numpy tqdm
venv/bin/python - <<'PY'
import torch, torchaudio, speechbrain, numpy, soundfile
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("arches", torch.cuda.get_arch_list())
print("speechbrain", speechbrain.__version__, "| numpy", numpy.__version__)
PY
echo ENV_READY
