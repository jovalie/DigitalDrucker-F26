#!/usr/bin/env python3
"""Calibrate the Drucker/other decision from the cached window embeddings.

Prints, per recording, the window similarity distribution to the Drucker reference
centroid and the two per-recording 2-means cluster means, so the absolute threshold
can be set from evidence rather than guessed.  Also reports the reference clips'
pairwise similarity (within-speaker floor) and how the known controls behave:
4794 (Bill May only), 4795 (Drucker + May), 3552/3558 (solo Drucker), 4796 (mostly Drucker).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_id import get_model, embed_windows, load_audio, l2norm, two_means  # noqa: E402

B = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/raid/projects/drucker-speaker")
REFS = sorted(glob.glob("/mnt/raid/shared/tts/voices/*_normalized.wav"))
CONTROLS = {"4794": "Bill May only", "4795": "Drucker+May Q&A", "4796": "mostly Drucker",
            "3552": "solo Drucker", "3558": "solo Drucker", "4849": "1986 VHS Drucker",
            "4899": "interview", "4939": "symposium"}

model = get_model(B / "models" / "ecapa")

# --- reference clips: within-speaker floor and each clip vs the clean-4796 centroid
clean = [p for p in REFS if "4796" in p]
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in clean]).mean(0, keepdims=True))
print(f"clean-4796 centroid from {len(clean)} clips\n")
print("reference clip similarities (to centroid | max vs another clip):")
E = embed_windows(model, [load_audio(Path(p)) for p in REFS])
for p, e in zip(REFS, E):
    print(f"  {Path(p).name:45s} {float((e @ cent.T).item()):+.3f} | {float((E @ e.T).max().item()):+.3f}")

# --- corpus
rows = []
for f in sorted((B / "embeddings").glob("*.npz")):
    d = np.load(f)
    emb = d["emb"]
    if len(emb) < 8:
        continue
    s = (emb @ cent.T).ravel()
    lab, cents = two_means(emb, cent)
    cs = [(cents[i] @ cent.T).item() for i in range(2)]
    hi, lo = max(cs), min(cs)
    rows.append({
        "p": f.stem, "n": len(emb), "p50": np.percentile(s, 50), "p90": np.percentile(s, 90),
        "hi": hi, "lo": lo, "gap": hi - lo,
        "f30": float((s >= 0.30).mean()), "f35": float((s >= 0.35).mean()),
        "f40": float((s >= 0.40).mean()),
    })

rows.sort(key=lambda r: -r["hi"])
print(f"\n{'pointer':8s} {'nwin':>5s} {'p50':>6s} {'p90':>6s} {'hi':>6s} {'lo':>6s} "
      f"{'gap':>5s} {'>=.30':>6s} {'>=.35':>6s} {'>=.40':>6s}  note")
print("-" * 92)
for r in rows:
    print(f"{r['p']:8s} {r['n']:5d} {r['p50']:+.3f} {r['p90']:+.3f} {r['hi']:+.3f} "
          f"{r['lo']:+.3f} {r['gap']:.3f} {r['f30']:6.2f} {r['f35']:6.2f} {r['f40']:6.2f}  "
          f"{CONTROLS.get(r['p'], '')}")

hi = np.array([r["hi"] for r in rows])
print(f"\ncluster-high sims across {len(rows)} recordings: "
      f"min={hi.min():+.3f} p10={np.percentile(hi,10):+.3f} p25={np.percentile(hi,25):+.3f} "
      f"median={np.median(hi):+.3f} p75={np.percentile(hi,75):+.3f} max={hi.max():+.3f}")
