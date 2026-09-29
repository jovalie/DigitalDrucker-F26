#!/usr/bin/env python3
"""Strip CosyVoice training bookkeeping keys so a checkpoint can be used as llm.pt.

CosyVoice's train.py saves {'epoch': .., 'step': .., ...state_dict...}; the inference
loader (CosyVoice2.__init__) uses strict=True and rejects those extra keys.
"""
import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
sd = torch.load(str(src), map_location="cpu")
for k in ("epoch", "step"):
    sd.pop(k, None)
if isinstance(sd.get("state_dict"), dict):
    sd = sd["state_dict"]
torch.save(sd, str(dst))
print(f"{src.name} -> {dst} ({len(sd)} tensors)")
