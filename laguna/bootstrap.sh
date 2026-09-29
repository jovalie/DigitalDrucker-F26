#!/bin/bash
# One-command Laguna bootstrap: verify login, push media, set up the env, submit the job.
#
# Required env:
#   LAGUNA_USER    e.g. JoaZheng@cmc.edu
#   DRUCKER_ROOT   /project/<PI>_<id>/drucker
# Optional:
#   LAGUNA_ACCOUNT project ID (e.g. <PI>_<id>); if unset, Slurm uses your default account
#   LAGUNA_HOST    default laguna1.carc.usc.edu
#
#   LAGUNA_USER=JoaZheng@cmc.edu DRUCKER_ROOT=/project/x_123/drucker bash laguna/bootstrap.sh
set -euo pipefail
: "${LAGUNA_USER:?set LAGUNA_USER, e.g. JoaZheng@cmc.edu}"
: "${DRUCKER_ROOT:?set DRUCKER_ROOT=/project/<PI>_<id>/drucker}"
HOST="${LAGUNA_HOST:-laguna1.carc.usc.edu}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1/4 verify login =="
ssh -o BatchMode=yes "$LAGUNA_USER@$HOST" "hostname; whoami; myaccount 2>&1 | head -8"

echo
echo "== 2/4 push media + scripts (8.1 GB, resume-safe) =="
LAGUNA_HOST="$HOST" LAGUNA_USER="$LAGUNA_USER" DRUCKER_ROOT="$DRUCKER_ROOT" bash "$DIR/laguna/sync_push.sh"

echo
echo "== 3/4 set up env on the login node (first run downloads large-v3) =="
ssh -o BatchMode=yes "$LAGUNA_USER@$HOST" \
    "DRUCKER_ROOT='$DRUCKER_ROOT' bash '$DRUCKER_ROOT/code/laguna/setup_env.sh'"

echo
echo "== 4/4 submit the GPU job =="
ACCOUNT_ARGS=""
[ -n "${LAGUNA_ACCOUNT:-}" ] && ACCOUNT_ARGS="--account=$LAGUNA_ACCOUNT"
ssh -o BatchMode=yes "$LAGUNA_USER@$HOST" \
    "cd '$DRUCKER_ROOT/code' && sbatch $ACCOUNT_ARGS --export=ALL,DRUCKER_ROOT='$DRUCKER_ROOT' laguna/transcribe_gpu.sbatch"
ssh -o BatchMode=yes "$LAGUNA_USER@$HOST" "squeue -u \$USER | head -5"

echo
echo "Monitor on Laguna:"
echo "  ssh $LAGUNA_USER@$HOST 'tail -f $DRUCKER_ROOT/code/laguna/logs/drk-whisper-<jobid>.out'"
echo "Pull results when done:"
echo "  LAGUNA_USER=$LAGUNA_USER DRUCKER_ROOT=$DRUCKER_ROOT bash laguna/sync_pull.sh"
