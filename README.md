# Digital Drucker — Fall 2026

Working repository for the Digital Drucker project: running **ComfyUI (LTX 2.3 / MiniMax H3 video generation)
on USC CARC Laguna**, and **training / serving the Drucker voice clone** (CosyVoice 2) from the archival
lecture corpus.

Everything here was developed and validated on `qclgpu` (4× Tesla V100-32GB, ComfyUI 0.37.0,
torch 2.7.1+cu126). GPUs with more VRAM (A100/H100) work too and relax the memory tricks noted below.

## Repo map

| Path | What it is |
|---|---|
| `comfyui/` | ComfyUI setup for Laguna: the saved workflows (LTX 2.3 ID-LoRA int8, MiniMax H3 i2v), the model manifest + resumable fetcher, and an API client example |
| `voice_cloning/` | The whole Drucker voice pipeline: speaker-ID (Drucker vs other speakers), Drucker-only transcripts, the speech-interval ledger, reference clips, the CosyVoice 2 fine-tuning kit, and the serving app |
| `voice_cloning/sft-data/` | The fine-tuning dataset (12 h of Drucker-only audio, Git LFS) + its manifest |
| `laguna/` | CARC Laguna access, bootstrap and sync scripts (from the transcription phase) |

## Quick start — ComfyUI on Laguna

```bash
git lfs install && git clone git@github.com:jovalie/DigitalDrucker-F26.git && cd DigitalDrucker-F26
# 1. ComfyUI + deps (see comfyui/README.md for the exact versions)
# 2. models (~90 GB for both workflows; fetcher is resumable + sha256-verified)
python3 comfyui/scripts/fetch-models.py --models-dir /path/to/ComfyUI/models --only ltx
# 3. workflows: copy comfyui/workflows/*.json into <base>/user/default/workflows/
# 4. serve (the HTTP API is always on):
python main.py --base-directory /path/to/base --listen 0.0.0.0 --port 8188
# 5. drive it: python3 comfyui/scripts/api-example.py --help
```

The LTX workflows ship in three variants because V100 has only 32 GB: `…(int8, V100)` keeps the Gemma
text encoder on the GPU, `…(TE on CPU)` moves it to CPU for ~11 GB more headroom, and
`…(TE CPU, 640x384x49)` also shrinks the base latent so the second (×2 upscale) pass fits.
On a bigger card use the first one.

## Quick start — voice clone

```bash
cd voice_cloning
python3 pipeline/speaker_id.py --help              # speaker ID over the corpus
python3 scripts/build_drucker_clips.py             # reference clips from the ledger
python3 tts/finetune/build_sft_data.py --help      # rebuild the SFT dataset from masked audio
bash   tts/finetune/run_train.sh                   # CosyVoice2 LLM fine-tune (see the script)
python3 tts/app.py --host 0.0.0.0 --port 8190      # the web portal
```

`voice_cloning/README.md` explains the whole chain; `voice_cloning/docs/` has the two design docs
(`SPEAKER_ID.md`, `SPEECH_TIMESTAMPS.md`).

## Rights

The archival recordings are The Drucker Institute material — **all rights retained, internal project use
only** (see `voice_cloning/voice_reference/CLIP_CITATION.md`). The source media are *not* in this repo;
only short reference clips, transcripts and derived metadata are.
