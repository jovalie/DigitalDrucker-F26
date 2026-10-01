---
title: Drucker TTS
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 30m
short_description: Speak in Peter Drucker's voice with CosyVoice 2
---

# Drucker TTS

Type a line, pick a reference clip from Peter Drucker's 1981 reel-to-reel seminars, and hear it
spoken in his voice.

* **Model:** [CosyVoice 2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) (0.5B) with the LLM
  fine-tuned on **12 hours of speaker-verified Drucker speech** — weights in
  [jovalie/Drucker-CosyVoice2-0.5B](https://huggingface.co/jovalie/Drucker-CosyVoice2-0.5B).
* **Cloning:** zero-shot from a reference clip; the clip's own transcript is the prompt, so the
  voices that ship with an accurate transcript give the best results.
* **Speaker verification:** the fine-tuning audio came from a speaker-identification pass over the
  archival corpus that separates Drucker from everyone else in the recordings (69 of the 80
  recordings contain another speaker; several are panels). Only Drucker-only segments were used.
* **Hardware:** ZeroGPU — the `@spaces.GPU` decorated synthesis call is the only part that needs a
  GPU, and the requested slice is sized to the text (60–150 s) because free accounts get ~5 min of
  ZeroGPU time per day.

Rights: the reference clips are excerpts from **The Drucker Institute** archival recordings, all
rights retained; they are used here for internal research. The model output is synthetic speech, not
a recording of Peter Drucker.

## Notes for whoever maintains this Space

The Space pins the pieces of CosyVoice's stack that ZeroGPU does not already provide (see
`requirements.txt` — never pin `torch`, `gradio`, `spaces`, `huggingface_hub`, `numpy` or
`pydantic`, the image ships those). `stubs/` is put on `sys.path` before CosyVoice is imported.

Synthesis was verified **token-for-token identical** to the reference deployment (a FastAPI portal
on the fine-tuning host) for the same input: 166 speech tokens, same mel shape `(1, 80, 332)`, same
6.64 s of audio. Re-check that agreement whenever the runtime image changes — that is the fastest way
to catch a silently degraded stack. What it took to get there, because ZeroGPU ships a much newer
stack than CosyVoice targets (torch 2.13 / transformers 5.x / numpy 2.x vs torch 2.3.1 /
transformers 4.51.3 / numpy 1.26):

| issue | fix in this repo |
| --- | --- |
| `hyperpyyaml` imports `cosyvoice.dataset.processor` and `cosyvoice.tokenizer.tokenizer` just to resolve class names, and those import openai-whisper (~2 GB build chain) and pyworld (source build) | `stubs/` provides `whisper.log_mel_spectrogram` reimplemented with librosa + torch (verified bit-identical to openai-whisper) and empty `pyworld` |
| newer torchaudio delegates file I/O to torchcodec, which the image does not ship | `_patch_torchaudio_io()` maps `torchaudio.load` onto soundfile, and the output wav is written with soundfile |
| transformers 5 instantiates `from_pretrained()` in the checkpoint's saved dtype, and the Qwen2.5 config says bfloat16 — CosyVoice builds fp32 activations, so generation died with "mat1 and mat2 must have the same dtype" | `_setup()` patches `Qwen2Encoder` to load the LLM with `torch.float32` |
| transformers 5 no longer expands a short attention mask to cover cached keys; CosyVoice's step-by-step decode passes a mask for the current chunk only, so everything after the first token attended to the wrong keys (fluent-sounding babble) | `_setup()` rebuilds the mask as all-visible in `Qwen2Encoder.forward_one_step` |

The fine-tuned `llm.pt` (1.9 GB) lives in the model repo rather than in the Space, which only allows
1 GB per repo.
