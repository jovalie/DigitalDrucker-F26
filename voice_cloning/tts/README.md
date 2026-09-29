# Drucker TTS webapp (qclgpu)

Text-to-speech in Peter Drucker's archived voice, with intonation prompting. Runs on **qclgpu,
GPU1** (V100) and is reachable over the tailnet.

- **UI / API:** <http://qclgpu.taild6baec.ts.net:8190> (or `http://100.69.136.49:8190`)
- **Engine:** [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice) `CosyVoice2-0.5B` (Apache-2.0),
  zero-shot clone from a 10–30 s archive clip, style controlled by a natural-language prompt.
- **Models:** `/mnt/raid/shared/tts/pretrained_models/CosyVoice2-0.5B` (4.6 GB)
- **Outputs:** `/mnt/raid/shared/tts/out/` (served at `/out/<file>.wav`)
- **Service:** `systemctl status drucker-tts` (enabled at boot; `serve-tts.sh` binds the tailnet IP)

## GPU allocation on qclgpu

| GPU | Used by | Notes |
|---|---|---|
| 0 | **ComfyUI** (`comfyui.service`, :8188) | kept available |
| 1 | **Drucker TTS** (`drucker-tts.service`, :8190) | ~2.6 GB VRAM when the model is resident |
| 2–3 | free | Ollama (DeepSeek) was stopped to free these: `docker start ollama` to bring it back |

## API

```bash
# list voices + intonation presets
curl -s http://100.69.136.49:8190/api/voices

# generate — intonation via natural-language prompt
curl -s -X POST http://100.69.136.49:8190/api/tts -H 'Content-Type: application/json' -d '{
  "text": "The best way to predict the future is to create it.",
  "voice": "drucker_1981_reel_10s",
  "mode": "instruct",
  "instruct": "measured and deliberate, with pauses for emphasis"
}'
# -> {"url": "/out/20260927-002812-....wav", "seconds": 8.1, "duration": 7.3}

# mode "clone" reproduces the reference clip's register (uses its transcript)
# mode "instruct" (default) follows the intonation prompt
# no transcript + no prompt -> cross-lingual copy of the voice
```

`GET /health` reports model state, GPU and VRAM. `GET /api/docs` is the OpenAPI UI.

First request after a restart loads the model (~25 s); after that synthesis runs at roughly
real-time on one V100 (~6–9 s for ~5–9 s of audio).

## Voices

`voices/voices.json` maps voice ids → reference wav + its transcript + a note. Current entries come
from `voice_reference/` in the Digital Drucker project:

| id | source | note |
|---|---|---|
| `drucker_1981_reel_10s` | 4796, 11:36 | reel-to-reel, SNR 35 dB — **recommended** |
| `drucker_1981_reel_15s` | 4796, 11:28 | same event, longer window |
| `drucker_1981_reel_30s` | 4796, 11:28 | more prosody, no transcript |
| `drucker_1981_lecture_15s` | 4796, 39:12 | lecture register |
| `drucker_1986_vhs_10s` | 4849, 5:40 | VHS audio, noisy — timbre only |

To add a voice: drop the wav in `voices/`, add an entry (with transcript for best cloning quality),
and refresh the page. No restart needed.

## Rebuild / move

```bash
bash /mnt/raid/shared/tts/install-cosyvoice.sh     # clone repo, venv, torch cu121, weights
sudo systemctl restart drucker-tts
tail -f /mnt/raid/shared/tts/logs/install.log
```

Notes from the install: `setuptools<81` is required (whisper's build) and `openai-whisper` must be
installed with `--no-build-isolation`; TensorRT is skipped (not needed on V100).

## Governance

Prototype use only. The archive audio is *"all rights retained by The Drucker Institute"* and this
is a synthetic voice of a deceased person — see `governance/RIGHTS.md` (G9) and
`governance/POSTHUMOUS_DIGITAL_PORTRAYAL_CASES.md` before any publication or external access.
