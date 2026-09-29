#!/bin/bash
# Pull GPU transcripts from Laguna into the project.
#
#   LAGUNA_USER=... DRUCKER_ROOT=... bash laguna/sync_pull.sh
#
# GPU outputs land in runs/transcribe-<date>/. This copies them into
# transcripts-gpu/ (kept separate from the local base-model transcripts/) and
# merges the run manifests into catalog/transcription_gpu_manifest.json.
set -euo pipefail
: "${LAGUNA_USER:?set LAGUNA_USER to your EPPN}"
: "${DRUCKER_ROOT:?set DRUCKER_ROOT=/project/<PI>_<id>/drucker}"
HOST="${LAGUNA_HOST:-laguna.carc.usc.edu}"
DST="$(cd "$(dirname "$0")/.." && pwd)/transcripts-gpu"

command -v rsync >/dev/null || { echo "rsync missing: apt-get install -y rsync"; exit 1; }
mkdir -p "$DST"

rsync -avP --partial "$LAGUNA_USER@$HOST:$DRUCKER_ROOT/runs/transcribe-*/" "$DST/"

echo "== merging manifests =="
python3 - "$DST" <<'PY'
import json, sys
from pathlib import Path
dst = Path(sys.argv[1])
merged = {"source": "laguna-gpu", "items": {}}
for mf in sorted(dst.glob("manifest.json")):
    data = json.loads(mf.read_text())
    merged["items"].update(data.get("items", {}))
out = dst.parent / "catalog" / "transcription_gpu_manifest.json"
out.write_text(json.dumps(merged, indent=2))
print(f"{len(merged['items'])} GPU transcripts → {out}")
PY
echo "done: $DST"
