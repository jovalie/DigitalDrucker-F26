# Optional-dependency stubs

`app.py` puts this directory on `sys.path` before importing CosyVoice.

CosyVoice's config (`cosyvoice2.yaml`) is resolved with hyperpyyaml, which *imports* modules such
as `cosyvoice.dataset.processor` and `cosyvoice.tokenizer.tokenizer` merely to look up class names.
Those modules import two dependencies at module level that are not worth installing on ZeroGPU:

| module | what it really needs | why a stub is enough |
| --- | --- | --- |
| `whisper` | openai-whisper (~2 GB of build chain) | `log_mel_spectrogram` is reimplemented here (librosa mel filters + torch STFT) and verified bit-identical to openai-whisper — it is the only whisper function the CosyVoice 2 inference path calls. `whisper.tokenizer` is only for CosyVoice 1. |
| `pyworld` | a C++ source build | used only by `cosyvoice/dataset/processor.py`, i.e. training-data preparation. |

Everything else follows CosyVoice's own `requirements.txt` (see `requirements.txt` here).
