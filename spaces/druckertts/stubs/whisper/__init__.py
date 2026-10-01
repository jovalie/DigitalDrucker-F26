"""Minimal stand-in for openai-whisper (20231117), implemented with librosa + torch.

CosyVoice's front end feeds whisper-style 128-mel features of the *prompt* wav straight into
its ONNX speech tokenizer, so `log_mel_spectrogram` has to be real: it is reproduced here
bit-for-bit (verified max|Δ| = 0 against the installed openai-whisper) without pulling in the
~2 GB openai-whisper dependency chain.

Note the shape: `torch.stft` keeps the leading dims of the input, so a (1, samples) prompt
comes back as (1, n_mels, frames) — CosyVoice reads `feat.shape[2]` for the tokenizer length.
"""

import librosa
import numpy as np
import torch

N_FFT = 400
HOP_LENGTH = 160

_FILTERS = {}


def mel_filters(device, n_mels):
    if n_mels not in _FILTERS:
        filters = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=n_mels).astype(np.float32)
        _FILTERS[n_mels] = torch.from_numpy(filters)
    return _FILTERS[n_mels].to(device)


def log_mel_spectrogram(audio, n_mels=128, padding=0, device=None):
    """Log-mel spectrogram of 16 kHz audio; identical to openai-whisper's implementation."""
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio))
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = torch.nn.functional.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters(audio.device, n_mels) @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def load_model(*args, **kwargs):
    raise RuntimeError("openai-whisper is not installed in this Space")


def load_audio(*args, **kwargs):
    raise RuntimeError("openai-whisper is not installed in this Space")
