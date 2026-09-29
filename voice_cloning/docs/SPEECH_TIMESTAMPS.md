# Drucker speech timestamps — ledger and re-mapping

The voice/likeness pipeline needs to know **when Drucker is speaking** in every recording,
and that knowledge has to survive re-editing (the noise-removed audio that is being
prepared for this project has a **different timeline**).  This document defines the
durable record and the tools that move it between timelines.

## What is stored

| File | Contents |
|---|---|
| `catalog/drucker_speech_intervals.json` | canonical ledger, one entry per recording |
| `catalog/drucker_speech_intervals.csv` | flat table, one row per interval |
| `catalog/drucker_labels/<ptr>.txt` | Audacity label track (`start<TAB>end<TAB>DRUCKER`), speech intervals only |
| `catalog/drucker_interval_sims.json` | per-interval Drucker similarity from the ECAPA centroid |

### Ledger schema (version 1)

```jsonc
{
  "version": 1,
  "timeline": "source media audio track, unedited; seconds from start of file …",
  "speaker_pass": "speaker_id.py: ECAPA window embeddings … abs_thr=0.38",
  "items": {
    "4796": {
      "media": "media/1981/4796__….mp4",
      "media_sha256": "…",            // ties the timeline to an exact file
      "duration_s": 2689.73,
      "speech_frac": 0.31,            // fraction of the recording with transcribed speech
      "no_speech": false,             // true = music/noise tape, excluded
      "drucker_seconds": 2512.0,
      "drucker_frac": 0.9339,
      "cluster_sim": [0.8100, 0.1723],// best / other per-recording cluster similarity
      "other_is_speaker": true,       // a genuine second speaker is present
      "intervals": [
        {
          "id": "4796-0004",
          "kind": "speech",           // "speech" = has transcribed Drucker text, "gap" = masked window with no speech
          "start": 164.0, "end": 492.5, "dur": 328.5,
          "sim": 0.63,                // mean ECAPA similarity to the Drucker centroid
          "segments": 41,
          "text": "…"                 // Drucker-only transcript text inside the interval
        }
      ]
    }
  }
}
```

Rules of the record:

- Times are **always** in the source media's own timeline, in seconds, and are only valid
  for the file whose `media_sha256` matches.
- `kind: "speech"` intervals are the ones that carry Drucker speech (use these for cutting
  audio, clipping, alignment). `kind: "gap"` intervals are mask windows that came back with
  no transcribed speech — they exist so that "what was kept" is auditable, but they are not
  speech and are omitted from the Audacity labels.
- `sim` is a cheap confidence: corpus median ≈ 0.60; 65 of 1,321 speech intervals sit below
  0.38 (mostly the noisy 1978 tapes and one 1987 symposium) — treat those as review items.

## When the noise-edited files arrive

Two modes, both in `scripts/map_edited_timestamps.py`, writing
`catalog/drucker_speech_intervals.edited.<asset>.json`:

### 1. The edited file is the Drucker intervals spliced together (noise removed)

This is the expected shape of the upcoming files.  The mapping is exact arithmetic, and the
duration is used as a check:

```bash
python3 scripts/map_edited_timestamps.py --pointer 4796 \
    --edited /path/4796_drucker_only.wav --assume-concatenated
```

Verified behaviour: for a synthesised cut of `4796`, all 7 speech intervals map exactly
(`duration_error = 0.0`), gaps correctly report `edited_start: null`, and the record carries
`duration_check: "ok"`.  If the duration check fails (> 2 % off) the record is flagged
`duration_check: "failed"` — use alignment instead.

### 2. Any other edit (also re-encodes, partial noise removal, different master)

Transcribe the edited file (the QCL GPU pipeline does 55 h in ~25 min) and align word
sequences with the original Drucker-only transcript; the tool fits a monotone
piecewise-linear time map through the matches and reports a confidence per interval:

```bash
python3 scripts/map_edited_timestamps.py --pointer 4796 \
    --edited /path/4796_clean.wav --edited-transcript /path/4796_clean.json
```

A `.srt` works as well as a `.json`.  Identity self-test: 215 anchors, 0.000 s error,
median confidence 0.93.  Intervals whose confidence is < 0.5 or that fall outside the
anchored range are listed under `flags.low_confidence_intervals` / `flags.unmapped_intervals`
— they should not be used for clips without inspection.

## Regenerating the ledger

```bash
python3 scripts/export_speech_intervals.py --checksums   # from catalog/speaker_labels + transcripts-drucker
```

The `--checksums` pass is the only slow step (15 GB of media) and is cached in the ledger, so
later runs reuse the recorded hashes.

## Limits

- The ledger reflects the speaker-ID pass (`SPEAKER_ID.md`); its known gaps (1966 interview
  `4899`/`4900`, music tapes `5030`/`5031`/`5034`) have no intervals.
- Intervals are contiguous runs of 2 s windows, so boundaries carry ±0.5 s of slop; the
  transcript text attached to each interval is the authoritative content.
- If a future edit is made from a *different* source master than `media/*.mp4` (e.g. a 44.1 kHz
  transfer with a different lead-in), always use alignment mode — the checksum will not match
  and arithmetic mapping would be off by the lead-in offset.
