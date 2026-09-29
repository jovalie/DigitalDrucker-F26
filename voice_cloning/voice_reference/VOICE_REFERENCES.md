# Drucker voice references — what each one is

Every clip registered in the CosyVoice voice catalog (`voices.json`, 5 entries) lives in
this folder (and in `/mnt/raid/shared/tts/voices/` on qclgpu) as **44.1 kHz mono PCM16**,
filtered `highpass=f=70` and normalized to **−18 LUFS / −1.5 dBTP**. Each has a `_raw.wav`
(unfiltered) beside it; the original selection analysis is in `VOICE_REFERENCE.md`.

> **2026-09-29 — the 8 auto-selected references were removed.** A batch of clips chosen by
> an SNR-driven score (`drucker_1987_*`, `drucker_1979_4997_226s`, `drucker_1981_4795_1534s`,
> `drucker_1981_4796_2559s`, `drucker_1980_3561_456s`) was rejected as sounding bad and
> removed from this folder and from `voices.json`. Files were quarantined, not deleted:
> `/mnt/raid/shared/tts/voices/.removed-20260929/`.
> **Why they failed:** SNR rewards a quiet noise floor, not a usable *timbre*. The 1987
> symposium tapes score 36–43 dB SNR but are narrow-band — spectral centroid 225–607 Hz vs
> 1405 Hz for the 1981 reel-to-reel — so the clone comes out muffled. Selection should be
> driven by **spectral centroid / bandwidth and high-frequency ratio**, with SNR only as a
> floor. The generator (`../scripts/build_drucker_clips.py`) can rebuild a set with that
> criterion.

## Catalog at a glance

| Voice id | Clip | Year | Source | Dur | SNR | Centroid | Transcript |
|---|---|---|---|---|---|---|---|
| `drucker_1981_reel_10s` | `4796_696s_10s` | 1981 | symposium reel II/2, 11:36.5 | 10 s | 35 dB | 1405 Hz | ✅ |
| `drucker_1981_reel_15s` | `4796_688s_15s` | 1981 | symposium reel II/2, 11:28.5 | 15 s | 35 dB | 1405 Hz | ✅ |
| `drucker_1981_reel_30s` | `4796_688s_30s` | 1981 | symposium reel II/2, 11:28.5 | 30 s | 35 dB | 1405 Hz | ❌ |
| `drucker_1981_lecture_15s` | `4796_2352s_15s` | 1981 | symposium reel II/2, 39:12 | 15 s | 35 dB | 1405 Hz | ❌ |
| `drucker_1986_vhs_10s` | `4849_340s_10s` | 1986 | lecture (VHS), 5:40 | 10 s | 12 dB | 645 Hz | ❌ |

Only the first two carry a transcript, which matters: the app's `mode: "clone"` needs audio
**and** transcript, otherwise it falls back to `instruct2` / cross-lingual.

## The five voices

### `drucker_1981_reel_10s` — default reference ⭐
Source `4796` *Peter Drucker Symposium, Reel II Part 2 of 2, 1981-04-22* at 11:36.5.
> "And the question about the President's administration that people don't ask is: why does
> everybody think it is so much of a difference?"

Reel-to-reel transfer, SNR 35 dB, wideband (centroid 1405 Hz), one complete sentence with
natural question intonation. **This is the reference used for the fine-tune evaluation and
the best default for cloning.**

### `drucker_1981_reel_15s`
Same recording at 11:28.5 — the same answer with its lead-in
> "You know, I'm being asked — I always wonder about the question people don't ask…"

More prosody and context, but starts mid-turn.

### `drucker_1981_reel_30s`
Same window, 30 s. **No transcript** — use for `instruct2` style prompting, not zero-shot.

### `drucker_1981_lecture_15s`
Same recording at 39:12, mid *"Towards the Next Economics"* — lecture register rather than
Q&A. **No transcript**; the original notes call it more reverberant and faster.

### `drucker_1986_vhs_10s` — visual reference, not for cloning
Source `4849` *Peter Drucker Lecture, 1986* at 5:40 — the only recording with **moving
footage**, kept for lip-sync/face work. VHS audio (SNR ≈ 12 dB); its embedding sits 0.27
from the Drucker centroid (vs 0.86–0.97 for the clean 1981 clips), so it is excluded from
clone prompts.

## Which one should you use?

| Goal | Use |
|---|---|
| Zero-shot clone (default) | `drucker_1981_reel_10s` |
| Clone with more prosody | `drucker_1981_reel_15s` |
| `instruct2` / style prompting | `drucker_1981_reel_30s`, `drucker_1981_lecture_15s` |
| Lip-sync / visual reference only | `drucker_1986_vhs_10s` |

Because the portal now runs the **fine-tuned model** (below), the reference mainly supplies
timbre; all of these are from the same 1981 reel, so they sound consistent.

## Fine-tuned model (portal default)

The web portal no longer relies on zero-shot alone: it loads
`/mnt/raid/shared/tts/pretrained_models/Drucker-CosyVoice2-0.5B/` — a copy of
CosyVoice2-0.5B with the LLM fine-tuned on 12 h of speaker-verified Drucker speech
(5,712 clips, masked audio, no other speakers).

| Checkpoint | Speaker sim ↑ | WER ↓ |
|---|---|---|
| base (no fine-tune) | 0.697 | 0.009 |
| epoch 1 | 0.704 | 0.035 |
| epoch 3 | 0.730 | 0.077 |
| **epoch 5 (deployed)** | **0.734** | **0.027** |
| epoch 10 | 0.744* | 0.122 |

*epoch 10 measured against the removed reference; not comparable. Scores are from 10 fixed
sentences with the `drucker_1981_reel_10s` reference, ECAPA similarity to the Drucker
centroid + WER from large-v3. Training overfits after epoch ~5 (CV loss rises from 3.99 at
epoch 1 to 7.99 at epoch 18), which is why the run was stopped and epoch 5 chosen.

Switching models (no sudo needed):

```bash
echo Drucker-CosyVoice2-0.5B > /mnt/raid/shared/tts/pretrained_models/ACTIVE   # fine-tuned
echo CosyVoice2-0.5B         > /mnt/raid/shared/tts/pretrained_models/ACTIVE   # stock model
# restart the unit (kill triggers systemd Restart=on-failure) or: sudo systemctl restart drucker-tts
```
`DRUCKER_TTS_MODEL=/path/to/model` overrides both. Checkpoint → deployable `llm.pt`
requires stripping the training `epoch`/`step` keys: `sft/make_llm.py`.

## Regenerating or curating a new reference set

```bash
python3 scripts/build_drucker_clips.py          # pool + curated refs (SNR-scored; see caveat)
python3 scripts/export_drucker_clips.py --all-pool   # export any clip from the pool
```
`../catalog/drucker_clip_pool.json` holds 1,469 verified Drucker clips (3.4 h);
`../catalog/drucker_speech_intervals.json` records when Drucker speaks in every source
recording (useful for picking alternates); `../catalog/voice_clarity.csv` has per-recording
`centroid_hz`, `bandwidth_hz` and `hf_ratio` for a timbre-first selection.
