# Drucker voice clone — pipeline

Reproduce the whole chain: separate Drucker from the other speakers in the archival corpus, keep a durable
record of when he speaks, curate reference clips, fine-tune CosyVoice 2 on his voice, and serve it.

## What's here

| Path | Contents |
|---|---|
| `pipeline/` | The GPU pipeline as run on qclgpu: `speaker_id.py` (embed / label / mask), `calibrate.py`, `regions.py`, `inspect_speaker.py`, `clip_select.py`, `interval_sims.py`, `transcribe_drucker.py`, `setup-env.sh` (builds the venv: torch 2.7.1+cu126 + speechbrain) |
| `scripts/` | Corpus-side tooling: `build_drucker_clips.py`, `export_drucker_clips.py`, `curate_voice_references.py`, `export_speech_intervals.py`, `map_edited_timestamps.py` |
| `catalog/` | `drucker_speech_intervals.{json,csv}` (the ledger of Drucker speech, checksum-tied to the source media), `drucker_interval_sims.json` (per-interval speaker confidence), `drucker_clip_pool.json` (1,469 verified clips), `speaker_labels/` (per-recording clustering), `drucker_labels/` (Audacity label tracks for editors) |
| `transcripts-drucker/` | 80 Drucker-only transcripts (`txt`/`srt`/`json`), 21,091 segments, verbatim — fillers and stop words kept |
| `voice_reference/` | The registered CosyVoice reference clips (44.1 kHz mono, −18 LUFS) + `VOICE_REFERENCES.md` (what each clip is) + rights notes |
| `tts/` | Serving app (`app.py`, `voices.json`, `serve-tts.sh`, systemd unit) |
| `tts/finetune/` | The fine-tuning kit: `build_sft_data.py`, `run_train.sh` + `conf/` (CosyVoice2 config), `make_llm.py`, `eval_checkpoints.sh`, `synthesize_ckpt.py`, `score_eval.py`, `wer_eval.py` |
| `sft-data/` | The 12 h fine-tuning dataset (Git LFS) + manifest |
| `docs/` | `SPEAKER_ID.md` (how Drucker is separated, with validation) and `SPEECH_TIMESTAMPS.md` (ledger schema + re-mapping onto noise-edited audio) |

## The chain

1. **Speaker ID** (`pipeline/speaker_id.py embed|label|mask`) — 2 s/0.5 s window ECAPA embeddings anchored
   to a clean 1981 reel centroid, per-recording cosine 2-means, hysteresis + median smoothing, speech gate.
   Result: **69 of the 80 recordings contain a second speaker**, and several files catalogued as "solo
   lecture" are panels (e.g. `3553` is ~80 % another speaker). Validation table in `docs/SPEAKER_ID.md`.
2. **Drucker-only audio + transcripts** — non-Drucker windows are zeroed (timeline preserved) and
   re-transcribed with faster-whisper large-v3 using a filler-friendly VAD. This also cleans up the noisy
   1974–78 cassettes (median word containment vs. the original pass 0.90, with *better* text on the bad tapes).
3. **Ledger** — `catalog/drucker_speech_intervals.json` records when Drucker speaks, in source-media time,
   tied to each file by sha256. `scripts/map_edited_timestamps.py` re-maps it onto noise-edited audio
   (concatenated cut: exact arithmetic; arbitrary edit: word-alignment, with per-interval confidence).
4. **Clips + references** — `scripts/build_drucker_clips.py` cuts segment-aligned clips inside Drucker
   regions; the curated few become the CosyVoice references in `voice_reference/`.
5. **Fine-tune** — `tts/finetune/build_sft_data.py` turns the masked audio into the SFT dataset
   (12 h / 5,712 clips), `run_train.sh` runs CosyVoice2-0.5B LLM-only SFT. Best checkpoint (epoch 5):
   speaker similarity **0.734** vs 0.697 for the stock model, WER 0.027 vs 0.009. Training overfits after
   ~epoch 5 (CV loss 3.99 → 7.99 by epoch 18), so the run was stopped and epoch 5 deployed; a lower-LR run
   is the obvious next improvement.
6. **Serve** — `tts/app.py` (FastAPI) loads the fine-tuned model and exposes `/api/tts`, `/api/voices`,
   `/health`. Model selection: `DRUCKER_TTS_MODEL` env → `pretrained_models/ACTIVE` file → stock model.

## Running it on Laguna

```bash
# 1. pipeline venv (needs a GPU; torch cu126 keeps V100/sm_70 support)
bash voice_cloning/pipeline/setup-env.sh
B=$PWD/voice_cloning
# 2. (only if re-deriving) audio prep + speaker ID + masking, sharded per GPU
python $B/pipeline/speaker_id.py embed --audio-root <flac dir> --out $B/work --shard 0 --shards 3
python $B/pipeline/speaker_id.py label --out $B/work
python $B/pipeline/speaker_id.py mask  --audio-root <flac dir> --out $B/work --shard 0 --shards 3
# 3. transcripts (faster-whisper large-v3)
python $B/pipeline/transcribe_drucker.py --queue $B/work/queue.tsv \
       --media-root $B/work/audio-drucker --out $B/work/transcripts-drucker \
       --model large-v3 --compute-type float16 --shard 0 --shards 3
# 4. CosyVoice fine-tune (clone the repo + fetch CosyVoice2-0.5B weights first)
bash $B/tts/finetune/run_train.sh
# 5. serve
python $B/tts/app.py --host 0.0.0.0 --port 8190
```

The dataset in `sft-data/` means step 2–3 can be skipped for training; you only need them to re-derive
transcripts or to extend the corpus.

### CosyVoice prerequisites (not in this repo)

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice
# CosyVoice2-0.5B weights (llm.pt, flow.pt, hift.pt, campplus.onnx,
# speech_tokenizer_v2.onnx, CosyVoice-BlankEN/) from FunAudioLLM/CosyVoice2-0.5B
```
`run_train.sh` expects `$TTS/venv`, `$TTS/CosyVoice` and `$TTS/pretrained_models/CosyVoice2-0.5B`; edit the
first lines to match your layout. Training checkpoints carry extra `epoch`/`step` keys that the inference
loader rejects — strip them with `tts/finetune/make_llm.py` before deploying.

## Caveats

- **1987 symposium tapes are narrow-band** (spectral centroid 225–607 Hz vs 1405 Hz for the 1981
  reel-to-reel): high SNR but muffled timbre. An earlier SNR-scored reference set was removed for this
  reason (see `voice_reference/VOICE_REFERENCES.md`); select references on centroid/bandwidth, not SNR.
- **`4899`/`4900`** (1966 Wolf interview) are not separable — ECAPA and ResNet both place Drucker and the
  interviewer at ~0.30 — so they are excluded. `5030`/`5031`/`5034` are music/noise tapes with no speech.
- Source recordings are **not** in this repo (Drucker Institute rights, see
  `voice_reference/CLIP_CITATION.md`); the ledger and transcripts are derived works for internal use.
