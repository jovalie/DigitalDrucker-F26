#!/usr/bin/env python3
"""Synthesize fixed test sentences with a CosyVoice2 model dir (checkpoint eval).

  python synthesize_ckpt.py <model_dir> <voice_wav> <voice_text> <out_dir> <texts.json>
"""
import json
import os
import sys
from pathlib import Path

T = Path("/mnt/raid/shared/tts")
sys.path.insert(0, str(T / "CosyVoice"))
sys.path.insert(0, str(T / "CosyVoice/third_party/Matcha-TTS"))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402

model_dir, voice_wav, voice_text, out_dir, texts_json = sys.argv[1:6]
texts = json.load(open(texts_json))
os.makedirs(out_dir, exist_ok=True)
m = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
for i, tx in enumerate(texts):
    chunks = [o["tts_speech"] for o in m.inference_zero_shot(tx, voice_text, voice_wav,
                                                             stream=False)]
    wav = torch.cat(chunks, dim=1)
    torchaudio.save(f"{out_dir}/{i:02d}.wav", wav, m.sample_rate)
    print("wrote", f"{out_dir}/{i:02d}.wav", flush=True)
