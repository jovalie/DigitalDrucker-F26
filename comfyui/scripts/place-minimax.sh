#!/bin/bash
S=~/drucker/comfy/models
B=/mnt/raid/shared/comfyui/models
L=/mnt/raid/shared/comfyui/logs/place-minimax.log
mkdir -p $B/loras $B/vae $B/text_encoders $B/diffusion_models
: > $L
copy(){ # src rel
  if [ -f "$B/$2" ] && [ "$(stat -c%s $B/$2)" = "$(stat -c%s $S/$1)" ]; then
    echo "[$(date +%H:%M:%S)] SKIP (already present) $2" >> $L; return; fi
  echo "[$(date +%H:%M:%S)] COPY $2" >> $L
  cp -f "$S/$1" "$B/$2" && echo "[$(date +%H:%M:%S)] COPIED $(stat -c%s $B/$2) $2" >> $L || echo "[$(date +%H:%M:%S)] COPY_FAIL $2" >> $L
}
copy loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors &
copy vae/minimax_h3_audio_vae_fp32.safetensors                            vae/minimax_h3_audio_vae_fp32.safetensors &
copy vae/minimax_h3_video_vae_int8_convrot.safetensors                    vae/minimax_h3_video_vae_int8_convrot.safetensors &
copy text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors           text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors &
copy diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors    diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors &
wait
echo "[$(date +%H:%M:%S)] VERIFY" >> $L
V(){ s=$(sha256sum "$B/$1" | cut -d' ' -f1); [ "$s" = "$2" ] && echo "  OK   $1" >> $L || echo "  BAD  $1 got=$s want=$2" >> $L; }
V loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors 2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e &
V vae/minimax_h3_audio_vae_fp32.safetensors 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48 &
V vae/minimax_h3_video_vae_int8_convrot.safetensors 52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e &
V text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors 35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6 &
V diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a &
wait
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $L
