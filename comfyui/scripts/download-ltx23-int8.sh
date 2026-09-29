#!/bin/bash
# V100-friendly LTX-2.3: int8_convrot transformer + standalone video/audio VAEs.
set -uo pipefail
B=/mnt/raid/shared/comfyui/models
L=/mnt/raid/shared/comfyui/logs/download-ltx23-int8.log
R=https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main
: > $L
dl(){ # rel size sha
  local rel=$1 size=$2 sha=$3 out=$B/$1
  mkdir -p "$(dirname "$out")"
  echo "[$(date +%H:%M:%S)] START $rel" >>$L
  curl -L -C - --retry 5 --retry-delay 5 --fail -s -o "$out" "$R/$rel" \
    || { echo "[$(date +%H:%M:%S)] FAIL $rel" >>$L; return; }
  local got; got=$(sha256sum "$out" | cut -d' ' -f1)
  [ "$got" = "$sha" ] && echo "[$(date +%H:%M:%S)] OK $(stat -c%s "$out") $rel" >>$L \
                          || echo "[$(date +%H:%M:%S)] SHA_BAD $rel" >>$L
}
dl diffusion_models/ltx-2.3-22b-dev_transformer_only_int8_convrot.safetensors 21505993064 6f4b86bd840cb83179e9021306a63119c3b4e664248292c4103613a2d4a4484c &
dl vae/LTX23_video_vae_bf16.safetensors 1452258578 01ea62d09bc139f95c5dee7b5c062ad6a3e6cd8be910a1983ac02e7eb5b8ee3b &
dl vae/LTX23_audio_vae_bf16.safetensors 364855188 5bc10fa4adecf99dda132d916e23048cbd56797702c5fa50eb5d2079048a38c3 &
wait
echo "[$(date +%H:%M:%S)] ALL_DONE" >>$L
