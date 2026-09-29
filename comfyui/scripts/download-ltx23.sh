#!/bin/bash
# LTX-2.3 model fetch for the shared ComfyUI install. Resumable + sha256-verified.
set -uo pipefail
B=/mnt/raid/shared/comfyui/models
L=/mnt/raid/shared/comfyui/logs/download-ltx23.log
mkdir -p "$B"/{checkpoints,loras,text_encoders,latent_upscale_models} "$(dirname "$L")"
: > "$L"

dl(){ # url rel expected_sha256
  local url="$1" rel="$2" sha="$3" out="$B/$2"
  echo "[$(date +%H:%M:%S)] START $rel" >>"$L"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$4" ]; then
    : # size already matches, fall through to verify
  else
    curl -L -C - --retry 5 --retry-delay 5 --fail -s -o "$out" "$url" \
      || { echo "[$(date +%H:%M:%S)] DOWNLOAD_FAIL $rel" >>"$L"; return; }
  fi
  local got; got=$(sha256sum "$out" | cut -d' ' -f1)
  if [ "$got" = "$sha" ]; then
    echo "[$(date +%H:%M:%S)] OK $(stat -c%s "$out") $rel" >>"$L"
  else
    echo "[$(date +%H:%M:%S)] SHA_BAD $rel got=$got want=$sha" >>"$L"
  fi
}

HF=https://huggingface.co

dl $HF/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors \
   checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
   28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450 29145431166 &

dl $HF/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors \
   loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors \
   31e0c0195fb841bf31af78e8b60858f489e87ddcea4a5239abc80943da65e3ac 2741024390 &

dl $HF/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors \
   loras/ltx-2.3-id-lora-talkvid-3k.safetensors \
   e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9 1157884304 &

dl $HF/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
   text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
   aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d 9447702218 &

dl $HF/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
   latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
   5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed 995743560 &

wait
echo "[$(date +%H:%M:%S)] ALL_DONE" >>"$L"
