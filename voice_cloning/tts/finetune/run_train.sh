#!/bin/bash
# CosyVoice2 LLM fine-tune on the speaker-verified Drucker corpus (GPUs 1-3).
T=/mnt/raid/shared/tts
S=/mnt/raid/projects/drucker-speaker/sft
cd $T/CosyVoice || exit 1
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH=$T/CosyVoice:$T/CosyVoice/third_party/Matcha-TTS
exec $T/venv/bin/torchrun --nnodes=1 --nproc_per_node=3 --rdzv_id=1986 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
  cosyvoice/bin/train.py --train_engine torch_ddp \
  --config $S/conf/cosyvoice2_drucker.yaml \
  --train_data $S/data/drucker_train/parquet/data.list \
  --cv_data $S/data/drucker_dev/parquet/data.list \
  --qwen_pretrain_path $T/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --onnx_path $T/pretrained_models/CosyVoice2-0.5B \
  --model llm --checkpoint $T/pretrained_models/CosyVoice2-0.5B/llm.pt \
  --model_dir $S/exp/cosyvoice2/llm/torch_ddp \
  --tensorboard_dir $S/tensorboard/cosyvoice2/llm/torch_ddp \
  --ddp.dist_backend nccl --num_workers 2 --prefetch 100 --pin_memory --use_amp \
  --deepspeed_config $S/conf/ds_stage2.json
