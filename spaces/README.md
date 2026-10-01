# Hugging Face Space — Drucker TTS

**Live:** https://huggingface.co/spaces/jovalie/druckertts (public, Gradio SDK, ZeroGPU `zero-a10g`).
Weights: https://huggingface.co/jovalie/Drucker-CosyVoice2-0.5B (`llm.pt`, 1.9 GB — it does not fit
in the Space repo, which caps at 1 GB).

`druckertts/` is the whole Space, version-controlled here so it can be rebuilt or moved:

| file | role |
| --- | --- |
| `app.py` | clone CosyVoice on boot, pull base + fine-tuned weights, Gradio UI, `@spaces.GPU` synthesis |
| `requirements.txt` | CosyVoice's runtime deps, minus what the ZeroGPU image ships (`torch`, `gradio`, `spaces`, `huggingface_hub`, `numpy`, `pydantic` must never be pinned there) |
| `stubs/` | `whisper` (real `log_mel_spectrogram` via librosa, bit-identical to openai-whisper) and `pyworld` |
| `voices.json` | the 5 reference voices: clip file, display name, and the clip's transcript |

The `voices/*.wav` files are the same clips as `voice_cloning/voice_reference/` — the Space keeps its
own copies because it uploads from a working directory, and the Space repo is standalone.

## Why the app patches CosyVoice

ZeroGPU ships torch 2.13 / transformers 5.18 / numpy 2.5, while CosyVoice targets torch 2.3.1 /
transformers 4.51.3 / numpy 1.26 (and a 4.x transformers cannot be installed there: every 4.x
requires `huggingface_hub<1.0`, while Gradio 6.29 requires `>=1.16`). Three patches live in `app.py`;
each one was needed to get past a concrete failure:

1. **`whisper` / `pyworld` stubs.** `cosyvoice2.yaml` is resolved with hyperpyyaml, which *imports*
   `cosyvoice.dataset.processor` and `cosyvoice.tokenizer.tokenizer` just to look up class names;
   those import openai-whisper (a ~2 GB build chain) and pyworld (source build) at module level.
   Whisper is not just an import-time problem though — `_extract_speech_token` really does call
   `whisper.log_mel_spectrogram` on the prompt wav and feeds the 128-mel features to the ONNX speech
   tokenizer, so the stub reimplements that function and was verified **bit-identical** (max|Δ| = 0)
   against the installed openai-whisper 20231117.
2. **LLM dtype.** `Qwen2Encoder` calls `Qwen2ForCausalLM.from_pretrained(pretrain_path)` with no
   dtype. transformers 5 instantiates in the checkpoint's *saved* dtype, and the Qwen2.5 config says
   bfloat16, while CosyVoice builds every activation in float32 → `RuntimeError: mat1 and mat2 must
   have the same dtype, but got Float and BFloat16`. The patch forces `torch.float32` (which is also
   what the fine-tuned `llm.pt` stores).
3. **Cached-decode attention mask.** `inference_wrapper` passes
   `tril(ones(1, T, T))` where `T` is the *current* chunk length, and `forward_one_step` reduces that
   to the last row. On the first step that mask spans the prompt (correct), but on every later step
   it is length 1 while the KV cache holds the whole prompt. transformers 4.x expanded such a mask;
   5.x takes it literally, so the model attended to the wrong keys and produced fluent-sounding
   babble (240 tokens, whisper transcribed "Ollie"). The patch rebuilds the mask as all-visible for
   `past + current` tokens.

Also patched for the same reason at a smaller scale: newer torchaudio routes `load`/`save` through
`torchcodec`, which the image does not ship, so `torchaudio.load` is redirected to soundfile and the
output wav is written with soundfile.

## Verifying a build (do this after any runtime-image change)

The fine-tuned model is **deterministic** for a fixed input (CosyVoice seeds the sampler), so the
generated speech-token sequence is a strong fingerprint. The reference values, produced by the
FastAPI portal on the fine-tuning host, for
`"The best way to predict the future is to create it."` with voice `1981 reel — 10 s`:

```
prompt_speech_tokens = 250
llm_tokens           = 166   sha1 = 618c936e4c5d
flow mel             = (1, 80, 332)
audio                = 159360 samples @ 24 kHz = 6.64 s
```

To re-measure on the Space, re-add a wrapper around `model.model.llm.inference` that records the
yielded token ids (see git history of `app.py`, commit "instrumented"), read the numbers out of
`hf spaces logs`, and compare. If the tokens match, everything downstream (flow, HiFi-GAN, wav
writing) is byte-equivalent to the reference deployment.

## Operational notes

* Free ZeroGPU accounts get ~5 minutes of GPU time per day, and a request is charged the slice it
  reserves — hence `_gpu_seconds()` sizes the slice from the text length (60–150 s) instead of
  asking for a fixed 180 s.
* The first call of a session also pays for ~4.5 GB of weight loading into VRAM.
* 600 characters is the input cap in `app.py`; the LLM's own `max_token_text_ratio` bounds the take.
