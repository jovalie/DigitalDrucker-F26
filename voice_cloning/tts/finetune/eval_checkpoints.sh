#!/bin/bash
# Evaluate CosyVoice2 checkpoints (and optionally the base model) on fixed sentences.
#   ./eval_checkpoints.sh                 # all epochs + base
#   ./eval_checkpoints.sh 0 1 2 base      # a subset
T=/mnt/raid/shared/tts
S=/mnt/raid/projects/drucker-speaker/sft
OUT=$S/eval
GPU=${EVAL_GPU:-2}
mkdir -p "$OUT"

cat > "$OUT/texts.json" <<'JSON'
["Management is doing things right; leadership is doing the right things.",
 "The best way to predict the future is to create it.",
 "There is nothing so useless as doing efficiently that which should not be done at all.",
 "The purpose of a business is to create and keep a customer.",
 "Knowledge has to be improved, challenged, and increased constantly, or it vanishes.",
 "The most important thing in communication is to hear what is not being said.",
 "Efficiency is doing things right; effectiveness is doing the right things.",
 "Plans are only good intentions unless they immediately degenerate into hard work.",
 "The entrepreneur always searches for change, responds to it, and exploits it as an opportunity.",
 "Rank does not confer privilege or give power. It imposes responsibility."]
JSON

VOICE=${EVAL_VOICE:-$T/voices/4796_696s_10s_normalized.wav}
VTEXT=${EVAL_TEXT:-"And the question about the President's administration that people don't ask is: why does everybody think it is so much of a difference?"}
[ -z "$VTEXT" ] && VTEXT="but it's going to be a dangerous period because you will not rely on it except the fact that we are"
export TQDM_DISABLE=1

SELECT=("$@")
if [ ${#SELECT[@]} -eq 0 ]; then
  SELECT=(base)
  for ck in $(ls "$S"/exp/cosyvoice2/llm/torch_ddp/epoch_*_whole.pt | sort -V); do
    SELECT+=("$(basename "$ck" _whole.pt | sed 's/epoch_//')")
  done
fi

for tag in "${SELECT[@]}"; do
  md="$OUT/model_$tag"
  mkdir -p "$md"
  if [ "$tag" = "base" ]; then
    ck="$T/pretrained_models/CosyVoice2-0.5B/llm.pt"
  else
    ck="$S/exp/cosyvoice2/llm/torch_ddp/epoch_${tag}_whole.pt"
    [ -f "$ck" ] || { echo "skip $tag (no checkpoint)"; continue; }
  fi
  for f in "$T"/pretrained_models/CosyVoice2-0.5B/*; do
    b=$(basename "$f")
    [ "$b" = "llm.pt" ] && continue
    ln -sfn "$f" "$md/$b"
  done
  "$T/venv/bin/python" "$S/make_llm.py" "$ck" "$md/llm.pt" 2>&1 | tail -1
  echo "=== $tag"
  CUDA_VISIBLE_DEVICES=$GPU "$T/venv/bin/python" "$S/synthesize_ckpt.py" \
      "$md" "$VOICE" "$VTEXT" "$OUT/wav_$tag" "$OUT/texts.json" 2>&1 | tail -1
done
echo "SYNTH_DONE"
