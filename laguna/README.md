# Laguna GPU transcription (faster-whisper large-v3)

High-accuracy transcription of the 80-recording Drucker corpus on CARC Laguna (L40S GPU).
The local base-model outputs in `transcripts/` are a coverage stopgap; these large-v3
transcripts are the canonical ones for the avatar/voice work.

```
container ──rsync──▶ Laguna /project/<PI>_<id>/drucker
                        ├── media/            8.1 GB recordings
                        ├── code/laguna/      scripts + queue.tsv
                        ├── envs/whisper-gpu/ venv (faster-whisper + CUDA wheels)
                        ├── hf-cache/         large-v3 weights
                        └── runs/transcribe-<date>/   txt/srt/json + manifest
container ◀──rsync── transcripts-gpu/ + catalog/transcription_gpu_manifest.json
```

## Prerequisites

- CARC account + active allocation (project ID form: `<PI>_<id>`).
- The **pi-container SSH key** must be the registered one. Its public key is:
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINYkJ9lhi1PlIKb3cHJt6NXJneCvK5P+CHjnR0aRd/vW carc-pi
  ```
  fingerprint `SHA256:J7USF1HgOCsmJ7Dwj6KNDLVBY+tq/vB1g0SzsQq+bQM`.
  Register/compare at <https://hpcaccount.usc.edu> → **User Profile → Public SSH Key**
  (allow 1–2 h for propagation).
- Export the connection values in this shell:
  ```bash
  export LAGUNA_USER='<EPPN>'                       # from hpcaccount.usc.edu (incommon page)
  export DRUCKER_ROOT='/project/<PI>_<id>/drucker'
  ```

## 1. Verify SSH access (once)

```bash
ssh "$LAGUNA_USER@laguna.carc.usc.edu" "hostname; whoami; myaccount; myquota"
```

## One command instead of steps 2–4

Once SSH works, everything below (push → env setup → submit) is one command:

```bash
LAGUNA_USER=JoaZheng@cmc.edu \
DRUCKER_ROOT=/project/<PI>_<id>/drucker \
LAGUNA_ACCOUNT=<PI>_<id> \
    bash laguna/bootstrap.sh
```

`LAGUNA_ACCOUNT` is optional — without it, Slurm submits under your default project account.

## 2. Push media + scripts (from this container)

```bash
bash laguna/sync_push.sh          # ~8.1 GB, rsync --partial, resume-safe
```

`rsync` must exist in the container (`apt-get install -y rsync` if missing).

## 3. One-time environment setup (Laguna login node)

```bash
DRUCKER_ROOT=/project/<PI>_<id>/drucker bash $DRUCKER_ROOT/code/laguna/setup_env.sh
```

Creates the venv (`faster-whisper` + pip CUDA wheels) and pre-downloads **large-v3** into
`$DRUCKER_ROOT/hf-cache`.

## 4. Submit the GPU job

```bash
cd "$DRUCKER_ROOT/code"
sbatch --account=<PI>_<id> --export=ALL,DRUCKER_ROOT="$DRUCKER_ROOT" laguna/transcribe_gpu.sbatch
squeue -u "$USER"
tail -f laguna/logs/drk-whisper-<jobid>.out
```

One L40S, `large-v3`, `float16`, VAD on. Estimate: **1–3 h** for all 80 recordings
(55.4 h of audio). The job reserves 4 h and is resumable — finished `.json` files are
skipped, so re-submitting after a timeout continues where it stopped.

## 5. Pull results

```bash
bash laguna/sync_pull.sh
# → transcripts-gpu/*.{txt,srt,json}, catalog/transcription_gpu_manifest.json
```

## Costs and rules

- SUs are charged on reserved resources: **1 L40S = 4 SU/min** → a 4 h reservation ≈ 960 SU.
  The corpus fits the Medium allocation comfortably.
- Never run this on the login node.
- Archival media stays on CARC storage; no third-party clouds or paid APIs.
- The queue file is `laguna/queue.tsv` (`pointer<TAB>media-path`, 80 rows). Regenerate it
  (`python3 -` script in the project history, or rebuild from `catalog/voice_clarity.csv`)
  if the media set changes.
