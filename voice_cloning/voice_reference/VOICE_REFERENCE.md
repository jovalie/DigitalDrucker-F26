# Voice reference recommendation — Digital Drucker

**Decision (2026-09-20): use pointer `4796` — *Peter Drucker Symposium, Reel II Part 2 of 2,
1981-04-22* — clip at `11:28.5`.**

This file contains both the tail of the Q&A and Drucker's solo talk *"Towards the Next
Economics"*. The selected window is a clean, single-speaker passage from the Q&A where Drucker
answers at length.

| | |
|---|---|
| Source file | `media/1981/4796__peter-drucker-symposium---reel-ii-part-2-of-2-1981-04-22.mp4` |
| **Video, 10 s (recommended)** | `voice_reference/4796_696s_10s.mp4` — 720×480 H.264/AAC, starts 11:36.5 |
| Video, 10 s (alternate) | `voice_reference/4796_688s_10s.mp4` — starts 11:28.5 |
| **Clip 0 (10 s, recommended)** | `voice_reference/4796_696s_10s_raw.wav` — 10 s from 11:36.5, complete sentence |
| Clip 0b (10 s trim) | `voice_reference/4796_688s_10s_raw.wav` — 10 s from 11:28.5 |
| Clip 1 (15 s) | `voice_reference/4796_688s_15s_raw.wav` — 15 s from 11:28.5 |
| Clip 2 (30 s) | `voice_reference/4796_688s_30s_raw.wav` — 30 s from 11:28.5 |
| Normalized | `..._normalized.wav` — 44.1 kHz mono, −18 LUFS, −1.5 dBTP, 70 Hz high-pass |
| Format | PCM 16-bit, 44.1 kHz, mono |
| Archive rights | "All rights are retained by The Drucker Institute" — internal project use only |

## Transcript of the 10 s clip (recommended, complete sentence)

> "And the question about the President's administration that people don't ask is: why does
> everybody think it is so much of a difference?"

## Transcript of the 15 s clip

> "You know, I'm being asked — I always wonder about the question people don't ask. And the
> question about the President's administration that people don't ask is: why does everybody
> think it is so much of a difference?"

## Why this recording and window

| Candidate | Centroid | HF content | Flatness (hiss) | SNR | Notes |
|---|---:|---:|---:|---:|---|
| **4796 (chosen)** | **1405 Hz** | **0.14 %** | **0.0036** | **35.4 dB** | CD transfer of reel-to-reel; clean wideband |
| 4796 window @11:28 | 1322 Hz | 0.20 % | 0.0034 | 32.4 dB | best single window, continuous speech 84 % |
| 4795 (same event, Q&A) | 1350 Hz | 0.12 % | 0.0029 | 35.4 dB | two speakers (Drucker + Bill May) |
| 4793 (Drucker talk, reel I) | 1642 Hz | 2.74 % | 0.0372 | 33.4 dB | "bandwidth" is mostly tape hiss |
| 5129 (Drucker talk, day 1) | 1765 Hz | 0.94 % | 0.0304 | 30.3 dB | cassette, hissy |
| 3558 (1980 lecture) | 568 Hz | 0.01 % | 0.0001 | 32.7 dB | guaranteed single speaker, but cassette band-limited (~2–3 kHz) |

Reasoning:
1. **Single speaker** — the window sits inside a long Drucker answer (no Bill May, no audience).
2. **Wideband and clean** — 4796 is a reel-to-reel transfer; the 1980 cassettes have 3–5× less
   high-frequency content, which would make a cloned "HD" voice sound muffled.
3. **Intelligible** — `faster-whisper base` transcribes this window cleanly; the same model
   garbles the hammier talk section, a practical clarity check.
4. **Natural delivery** — a complete thought with question intonation, not a fragment.

## Alternates

- **Pure lecture register, same file:** `39:12–39:27` (mid *"Towards the Next Economics"*).
  Export: `python3 scripts/find_voice_reference.py <file> --export 15 --export-start 2352`.
  Wider metrics are similar, but intelligibility is lower (more reverberant / faster speech).
- **Guaranteed single-speaker lecture:** pointer `3558` (1980-04-23) — cleanest cassette, but
  narrow band. Export with the same script after a scan.
- **Do not use:** `4794` (Bill May only), `4793` (hiss), `4795` (Q&A with May).

## Reproduce

```bash
# score windows across a recording
python3 scripts/find_voice_reference.py media/1981/4796__....mp4 --win 12 --top 15

# export the primary clip + normalized copy
python3 scripts/find_voice_reference.py media/1981/4796__....mp4 \
    --export 15 --export-start 688.5 --out-dir voice_reference
```

## Real video clip — Drucker speaking on camera (1986)

Most files in this collection are audio recordings digitized with a static frame (motion = 0.0).
The only downloaded recordings with **actual moving footage** are pointers **4849/4850 —
*Peter Drucker Lecture, 1986*** (color videocassette, Drucker Archives Box 80). The archive
description and an `small`-model transcript confirm the speaker: an introduction of
“Professor Drucker” runs to ~4:47, and Drucker himself speaks from **~4:47** to the end of part 2.

| Clip | What |
|---|---|
| **`4849_340s_10s.mp4` (recommended)** | 10 s from 5:40 — Drucker’s opening remarks (“I wasn’t even around when the academy was founded…”), largest face detected (face area 2228 px at scan resolution) |
| `4849_824s_10s.mp4` | 10 s from 13:44 — mid-lecture, strongest on-camera motion (face-motion 16.6) |
| `4850_1155s_10s.mp4` | 10 s from 19:15 of part 2 |
| `4849_340s_10s_audio_{raw,normalized}.wav` | audio of the recommended clip |

**Audio caveat:** the 1986 VHS audio is noisy (clarity SNR ≈ 12 dB vs 35 dB for 4796). For a voice
model use the 4796 audio above; use these clips as the **visual/motion reference**. The selection
method is `scripts/find_speaker_clip.py` (YuNet face detection + face-region motion + cut
rejection).

## Governance

Voice/likeness model training and storage still require the written terms tracked in
`governance/RIGHTS.md` (G9). Until then this reference is **internal prototype use only** —
not for publication, product, or third-party upload.
