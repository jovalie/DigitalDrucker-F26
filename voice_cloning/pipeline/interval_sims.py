#!/usr/bin/env python3
"""Per-interval mean Drucker similarity (ECAPA), 20 s chunks to bound memory."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/mnt/raid/projects/drucker-speaker")
from speaker_id import get_model, embed_windows, load_audio, l2norm

B = Path("/mnt/raid/projects/drucker-speaker")
model = get_model(B / "models" / "ecapa")
refs = sorted(glob.glob("/mnt/raid/shared/tts/voices/4796_*_normalized.wav"))
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in refs]).mean(0, keepdims=True))
out = {}
for lp in sorted((B / "labels").glob("*.json")):
    rec = json.loads(lp.read_text())
    regs = rec.get("drucker_regions") or []
    if not regs:
        continue
    x = load_audio(B / "audio-drucker" / f"{lp.stem}.flac")
    item = {}
    for i, (t0, t1) in enumerate(regs, 1):
        y = x[int(t0 * 16000):int(t1 * 16000)]
        if len(y) < 16000:
            item[f"{lp.stem}-{i:04d}"] = None
            continue
        chunks = [y[j:j + 10 * 16000] for j in range(0, len(y), 10 * 16000)
                  if len(y[j:j + 10 * 16000]) >= 16000]
        if not chunks:
            item[f"{lp.stem}-{i:04d}"] = None
            continue
        e = l2norm(embed_windows(model, chunks, batch=4).mean(axis=0, keepdims=True))
        item[f"{lp.stem}-{i:04d}"] = round(float((e @ cent.T).ravel()[0]), 4)
    out[lp.stem] = item
(B / "interval_sims.json").write_text(json.dumps(out, indent=1))
print("wrote interval_sims.json:", sum(len(v) for v in out.values()), "intervals")
