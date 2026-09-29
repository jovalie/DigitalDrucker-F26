#!/usr/bin/env bash
# ComfyUI model fetcher (GPU0 instance). Usage: bash download-models.sh
set -uo pipefail
B=/mnt/raid/shared/comfyui/models
get() { # url dest
  echo "==> $(basename "$2")"
  curl -sL --fail --retry 3 --retry-delay 3 -C - -o "$2" "$1" || echo "FAILED: $1"
  ls -l "$2" 2>/dev/null | awk '{print "    ", $5, $9}'
}
# ComfyUI workflow-template models (SAM3 / SAM3D body / SDPose / MoGe)
get https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors        $B/checkpoints/sam3.1_multiplex_fp16.safetensors
get https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors          $B/checkpoints/sdpose_wholebody_fp16.safetensors
get https://huggingface.co/Comfy-Org/SDPose/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp32.safetensors   $B/diffusion_models/rt_detr_v4-x-hgnet_fp32.safetensors
get https://huggingface.co/Comfy-Org/SDPose/resolve/main/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors   $B/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors
get https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_bf16.safetensors      $B/detection/sam_3d_body_dinov3_bf16.safetensors
get https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors  $B/geometry_estimation/moge_2_vitl_normal_fp16.safetensors
get https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_1_vitl_fp16.safetensors          $B/geometry_estimation/moge_1_vitl_fp16.safetensors
echo "done."
