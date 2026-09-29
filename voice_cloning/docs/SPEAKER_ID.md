# Speaker identification pass — Drucker vs. other speakers

**Goal:** the 80 recordings are Drucker lectures/symposia, but several contain a second
speaker (panelists, moderators, questioners, co-lecturers). Transcribe and clone **only
Drucker**, verbatim (stop words and fillers kept).

Run on `qclgpu` (4× V100), working dir `/mnt/raid/projects/drucker-speaker/`.
Code: `speaker_id.py` (embed / label / mask), `calibrate.py`, `regions.py`,
`inspect_speaker.py`, `clip_select.py`, `transcribe_drucker.py`.

## Method

1. **Window embeddings** — 2 s windows, 0.5 s hop over every FLAC (16 kHz), energy gate
   at −45 dBFS → ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-d, L2-normalised).
2. **Anchored per-recording clustering** — a *Drucker centroid* is built from the four
   clean 1981 reel-to-reel reference clips (4796).  Each recording runs a cosine 2-means
   seeded with that centroid, so channel/era differences are absorbed per recording while
   "which cluster is Drucker" stays pinned to the reference voice.
3. **Decisions**
   - best cluster similarity `< 0.38` → **no Drucker** in that recording;
   - other cluster `≥ 0.38` → both clusters are Drucker (cassette side / EQ change, not a
     speaker change) → keep everything;
   - otherwise per-window hysteresis (`sim_drucker > sim_other`) + 5-window median filter,
     regions ≥ 1 s, gaps ≤ 0.6 s merged.
4. **Speech gate** — recordings with `< 15 %` transcribed speech (music/noise tapes) are
   marked `no_speech` and excluded.
5. **Drucker-only audio** — non-Drucker windows are zeroed (timeline preserved), then
   re-transcribed with faster-whisper large-v3 (fp16, beam 5) using a filler-friendly VAD
   (`min_speech_duration_ms=100`, `threshold=0.35`, `min_silence_duration_ms=700`).

## Validation

| Control | Expectation | Result |
|---|---|---|
| `4794` (Bill May only) | no Drucker | ✓ 0 segments; cluster sim 0.15 |
| `3553` ("solo lecture" in catalog) | catalog wrong | ✓ 80 % of tape is another speaker ("Thank you very much, Peter…") |
| `3552`, `3558`, `3559`, `4792` ("solo") | catalog wrong | ✓ panel/Q&A speech found and excluded |
| `4795` (Drucker + Bill May) | mixed | ✓ other cluster sim 0.117 ≈ Bill May's 0.15 |
| `5013`, `5021`, `5023`, `5034` | channel split, not speaker | ✓ unified by the ≥0.38 rule |
| `3556`, `4796`, `5129` | solo Drucker | ✓ 93–97 % kept, no speaker splits |

Quality check on the masked re-transcription: median word containment vs. the original
pass is 0.90.  On the noisy 1974–78 cassettes the new text is **better**, not a subset —
e.g. `5028` old "because of the treasure that and keep it" → new "Now may I please
repeat… I have no deadline for paper… I just turn it in to Lloyd Ingram".

## Outputs

| Path | Contents |
|---|---|
| `transcripts-drucker/<ptr>.{txt,srt,json}` | Drucker-only transcripts, 80 files, 21,091 segments, 1.47 M chars |
| `catalog/speaker_labels/<ptr>.json` | per-recording clusters, regions, `speech_frac`, `no_speech` |
| `catalog/speaker_labels/summary.csv` | one row per recording (similarities, regions, Drucker fraction) |
| `catalog/drucker_clip_pool.json` | 1,469 Drucker-only clips (segment-aligned, 4–12 s) |
| `voice_reference/` | 8 curated reference clips (44.1 kHz raw + normalized) + `manifest.json` |
| `voice_reference/../…` TTS | `/mnt/raid/shared/tts/voices/` + `voices.json` (13 voices) |

## Known gaps

- **`4899`/`4900`** (1966 W. B. Wolf interview): Drucker speaks, but ECAPA **and** ResNet
  embeddings place both voices ~0.30 (era drift + poor tape) and neither model separates
  them → excluded from Drucker-only output.  Needs manual review if wanted.
- **`5030`, `5031`, `5034`**: music/noise tapes, `< 1 %` speech → `no_speech`.
- Catalog `content_type` labels are unreliable: several files marked "lecture (single
  speaker)" are panels (3552, 3553, 3558, 3559, 4792), and 3557's Drucker stretches are
  interleaved with a second speaker.

## Reproduce

```bash
B=/mnt/raid/projects/drucker-speaker
$B/venv/bin/python $B/speaker_id.py embed --out $B --shard i --shards 3   # i=0..2, GPUs 1..3
$B/venv/bin/python $B/speaker_id.py label --out $B --abs-thr 0.38
$B/venv/bin/python $B/speaker_id.py mask  --out $B --shard i --shards 3
~/drucker/env/bin/python $B/transcribe_drucker.py --queue $B/queue-drucker.tsv \
    --media-root $B/audio-drucker --out $B/transcripts-drucker --model large-v3 \
    --compute-type float16 --shard i --shards 3
bash qcl/sync_drucker.sh
python3 scripts/build_drucker_clips.py
```
