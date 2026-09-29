#!/usr/bin/env python3
"""Score synthesized checkpoint samples: ECAPA speaker similarity to the Drucker centroid."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/raid/projects/drucker-speaker")
from speaker_id import embed_windows, get_model, l2norm, load_audio  # noqa: E402

B = Path("/mnt/raid/projects/drucker-speaker")
model = get_model(B / "models" / "ecapa")
refs = sorted(glob.glob("/mnt/raid/shared/tts/voices/4796_*_normalized.wav"))
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in refs]).mean(0, keepdims=True))

out = {}
for d in sorted(glob.glob(str(B / "sft/eval/wav_*"))):
    ep = Path(d).name.replace("wav_", "")
    sims = []
    for w in sorted(glob.glob(d + "/*.wav")):
        e = embed_windows(model, [load_audio(Path(w))])
        sims.append(round(float((e @ cent.T).ravel()[0]), 4))
    out[ep] = {"n": len(sims), "mean_sim": round(float(np.mean(sims)), 4) if sims else None,
               "min_sim": min(sims) if sims else None, "sims": sims}
    print(ep, out[ep]["mean_sim"])
print(json.dumps(out, indent=1))
(B / "sft/eval/similarity.json").write_text(json.dumps(out, indent=1))
