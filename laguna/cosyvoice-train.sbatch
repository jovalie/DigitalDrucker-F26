#!/bin/bash
#SBATCH --job-name=drucker-cosyvoice
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/project/JoaZheng_1796/drucker/runs/cosyvoice-train-%j.log
#
# CosyVoice2-0.5B LLM-only SFT on the Drucker SFT dataset (see voice_cloning/tts/finetune/).
# Prerequisites:  bash laguna/prepare-sft-data.sh   (once, to build the parquet dataset)
#
#   sbatch laguna/cosyvoice-train.sbatch
#
# On the V100s this was run with fp16 AMP (torch 2.3 had no bf16 there); the L40S supports
# bf16 natively, which is what the stock CosyVoice2 recipe uses — drop --use_amp to train in
# bf16, or keep it for fp16 AMP exactly as validated on QCL.
set -uo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 2>/dev/null

B=/project/JoaZheng_1796/drucker
CV=$B/cosyvoice/CosyVoice
PY=$B/envs/cosyvoice/bin/python
PT=$B/cosyvoice/pretrained_models/CosyVoice2-0.5B
DATA=$B/data
CONF=$B/repo/voice_cloning/tts/finetune/conf/cosyvoice2_drucker.yaml
RUN=$B/runs/cosyvoice2/llm/torch_ddp

[ -f "$DATA/train.data.list" ] || { echo "run laguna/prepare-sft-data.sh first"; exit 1; }
mkdir -p "$RUN" "$B/runs"

export TQDM_DISABLE=1
export PYTHONPATH="$CV:$CV/third_party/Matcha-TTS"
export CUDA_VISIBLE_DEVICES=0,1
NGPU=2

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train shards: $(wc -l < "$DATA/train.data.list")  dev shards: $(wc -l < "$DATA/dev.data.list")"

cd "$CV"
exec "$B/envs/cosyvoice/bin/torchrun" --nnodes=1 --nproc_per_node=$NGPU \
    --rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
    cosyvoice/bin/train.py --train_engine torch_ddp \
    --config "$CONF" \
    --train_data "$DATA/train.data.list" \
    --cv_data "$DATA/dev.data.list" \
    --qwen_pretrain_path "$PT/CosyVoice-BlankEN" \
    --onnx_path "$PT" \
    --model llm --checkpoint "$PT/llm.pt" \
    --model_dir "$RUN" \
    --tensorboard_dir "$B/runs/tensorboard/cosyvoice2/llm/torch_ddp" \
    --ddp.dist_backend nccl --num_workers 4 --prefetch 100 --pin_memory --use_amp \
    --deepspeed_config "$B/repo/voice_cloning/tts/finetune/conf/ds_stage2.json"
