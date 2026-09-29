#!/bin/bash
# Push media + job scripts from this container to Laguna.
#
# Required env:
#   LAGUNA_USER   your EPPN (e.g. njzheng@cgu.edu or your USC NetID)
#   DRUCKER_ROOT  /project/<PI>_<id>/drucker
# Optional:
#   LAGUNA_HOST   default laguna.carc.usc.edu
#
#   LAGUNA_USER=... DRUCKER_ROOT=... bash laguna/sync_push.sh
set -euo pipefail
: "${LAGUNA_USER:?set LAGUNA_USER to your EPPN}"
: "${DRUCKER_ROOT:?set DRUCKER_ROOT=/project/<PI>_<id>/drucker}"
HOST="${LAGUNA_HOST:-laguna.carc.usc.edu}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

command -v rsync >/dev/null || { echo "rsync missing: apt-get install -y rsync (or use the host)"; exit 1; }

echo "== creating remote dirs =="
ssh -o BatchMode=yes "$LAGUNA_USER@$HOST" \
    "mkdir -p '$DRUCKER_ROOT/media' '$DRUCKER_ROOT/code/laguna/logs' '$DRUCKER_ROOT/logs'"

echo "== media (8.1 GB, resume-safe) =="
rsync -avP --partial "$SRC/media/" "$LAGUNA_USER@$HOST:$DRUCKER_ROOT/media/"

echo "== scripts =="
rsync -avP "$SRC/laguna/transcribe_gpu.py" "$SRC/laguna/queue.tsv" \
           "$SRC/laguna/transcribe_gpu.sbatch" "$SRC/laguna/setup_env.sh" \
           "$SRC/laguna/sync_pull.sh" "$SRC/laguna/README.md" \
           "$LAGUNA_USER@$HOST:$DRUCKER_ROOT/code/laguna/"

echo "done. One command from here:"
echo "  LAGUNA_USER=$LAGUNA_USER DRUCKER_ROOT=$DRUCKER_ROOT [LAGUNA_ACCOUNT=<PI>_<id>] bash laguna/bootstrap.sh"
