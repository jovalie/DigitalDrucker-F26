#!/bin/bash
# Poll for Laguna SSH access; on first success run the first-session checks.
# Usage: bash laguna/check_access.sh ['EPPN@domain']
set -u
EPPN="${1:-JoaZheng@cmc.edu}"
HOST="${LAGUNA_HOST:-laguna1.carc.usc.edu}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/logs/access_check.log"
mkdir -p "$DIR/logs"

for i in $(seq 1 24); do   # 24 × 10 min = 4 hours
  if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=12 -o LogLevel=ERROR "$EPPN@$HOST" true 2>/dev/null; then
    {
      echo "=== ACCESS OK at $(date -u +%FT%TZ) as $EPPN@$HOST ==="
      timeout 90 ssh -o BatchMode=yes -o LogLevel=ERROR "$EPPN@$HOST" \
        "hostname; whoami; echo '--- myaccount ---'; myaccount 2>&1 | head -25; \
         echo '--- myquota ---'; myquota 2>&1 | head -10; \
         echo '--- sinfo (gpu) ---'; sinfo -s -p gpu 2>&1 | head -6"
    } >> "$LOG" 2>&1
    echo "=== ACCESS OK $(date -u +%FT%TZ) ==="
    exit 0
  fi
  echo "$(date -u +%FT%TZ) not yet ($i/24) $EPPN@$HOST" >> "$LOG"
  sleep 600
done
echo "$(date -u +%FT%TZ) gave up after 4h" >> "$LOG"
