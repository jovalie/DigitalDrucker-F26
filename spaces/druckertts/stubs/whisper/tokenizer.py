"""Stub for whisper.tokenizer: only used by CosyVoice 1's text tokenizer, not CosyVoice 2."""


class Tokenizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("openai-whisper is not installed in this Space")
