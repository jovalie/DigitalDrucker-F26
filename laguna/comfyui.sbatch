#!/bin/bash
#SBATCH --job-name=drucker-comfyui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/JoaZheng_1796/drucker/runs/comfyui-%j.log
#
# ComfyUI API server on a Laguna GPU node.
#
#   sbatch laguna/comfyui.sbatch
#
# Then, from your laptop (the compute node is only reachable via the login node):
#
#   ssh -N -L 8188:<node>:8188 JoaZheng@cmc.edu@laguna1.carc.usc.edu
#
# and open http://localhost:8188  (/system_stats, /prompt, /view … all live there).
# The node name is printed at the top of the job log, and also written to
# /project/JoaZheng_1796/drucker/runs/comfyui-node.txt
set -uo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 2>/dev/null

B=/project/JoaZheng_1796/drucker
mkdir -p "$B/runs" "$B/comfyui/user/default/workflows"
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-none} $(date -u +%FT%TZ)"
echo "$(hostname)" > "$B/runs/comfyui-node.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# models + workflows live outside the ComfyUI clone; wire them in
[ -e "$B/comfyui/models" ] || ln -s "$B/models" "$B/comfyui/models"
cp -n "$B/repo/comfyui/workflows/"*.json "$B/comfyui/user/default/workflows/" 2>/dev/null || true
ls "$B/comfyui/user/default/workflows/"

cd "$B/comfyui/ComfyUI"
exec "$B/envs/comfyui/bin/python" main.py \
    --base-directory "$B/comfyui" \
    --listen 0.0.0.0 --port 8188 \
    --output-directory "$B/comfyui/output"
