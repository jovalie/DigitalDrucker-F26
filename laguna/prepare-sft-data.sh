#!/bin/bash
# Turn the shipped SFT tarball into the parquet dataset CosyVoice training expects.
#
#   bash laguna/prepare-sft-data.sh          # unpack + split + extract features (needs a GPU)
#
# Steps: unpack -> rewrite wav.scp to local paths -> split train/dev from utt_index.json ->
# extract campplus speaker embeddings -> extract speech tokens (GPU/ONNX) -> build parquet.
set -uo pipefail
B=/project/JoaZheng_1796/drucker
CV=$B/cosyvoice/CosyVoice
PY=$B/envs/cosyvoice/bin/python
PT=$B/cosyvoice/pretrained_models/CosyVoice2-0.5B
TAR=$B/repo/voice_cloning/sft-data/drucker_sft_12h.tar
DATA=$B/data

echo "=== $(date -u +%FT%TZ) unpacking"
mkdir -p "$DATA"
[ -f "$DATA/drucker/utt_index.json" ] || tar -xf "$TAR" -C "$DATA"

echo "=== rewriting wav.scp to $DATA/drucker/wav"
awk -v d="$DATA/drucker" '{print $1, d"/wav/"$1".wav"}' "$DATA/drucker/wav.scp" > "$DATA/drucker/wav.scp.new"
mv "$DATA/drucker/wav.scp.new" "$DATA/drucker/wav.scp"
echo "  $(wc -l < "$DATA/drucker/wav.scp") utts"

echo "=== splitting train/dev from utt_index.json"
"$PY" - "$DATA/drucker" <<'PY'
import json, sys, shutil
from pathlib import Path
d = Path(sys.argv[1])
idx = json.loads((d / "utt_index.json").read_text())
splits = {}
for r in idx:
    splits.setdefault(r["split"], []).append(r["utt"])
for split, utts in splits.items():
    out = d.parent / f"drucker_{split}"
    out.mkdir(exist_ok=True)
    keep = set(utts)
    for fname in ("wav.scp", "text", "utt2spk"):
        with (d / fname).open() as fh, (out / fname).open("w") as fo:
            for line in fh:
                if line.split()[0] in keep:
                    fo.write(line)
    (out / "spk2utt").write_text("drucker " + " ".join(utts) + "\n")
    print(f"  {split}: {len(utts)} utts -> {out}")
PY

echo "=== campplus speaker embeddings"
"$PY" "$CV/tools/extract_embedding.py" --dir "$DATA/drucker_train" --onnx_path "$PT/campplus.onnx" --num_thread 8 2>&1 | tail -1
"$PY" "$CV/tools/extract_embedding.py" --dir "$DATA/drucker_dev"   --onnx_path "$PT/campplus.onnx" --num_thread 8 2>&1 | tail -1

echo "=== speech tokens (GPU)"
"$PY" "$CV/tools/extract_speech_token.py" --dir "$DATA/drucker_train" --onnx_path "$PT/speech_tokenizer_v2.onnx" --num_thread 8 2>&1 | tail -1
"$PY" "$CV/tools/extract_speech_token.py" --dir "$DATA/drucker_dev"   --onnx_path "$PT/speech_tokenizer_v2.onnx" --num_thread 8 2>&1 | tail -1

echo "=== parquet shards"
for s in train dev; do
  mkdir -p "$DATA/drucker_$s/parquet"
  "$PY" "$CV/tools/make_parquet_list.py" --num_utts_per_parquet 1000 --num_processes 6 \
        --src_dir "$DATA/drucker_$s" --des_dir "$DATA/drucker_$s/parquet" 2>&1 | tail -1
done
cat "$DATA/drucker_train/parquet/data.list" > "$DATA/train.data.list"
cat "$DATA/drucker_dev/parquet/data.list"   > "$DATA/dev.data.list"
echo "=== done $(date -u +%FT%TZ)"
echo "train shards: $(wc -l < "$DATA/train.data.list")  dev shards: $(wc -l < "$DATA/dev.data.list")"
