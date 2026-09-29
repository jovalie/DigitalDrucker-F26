#!/usr/bin/env python3
"""Show contiguous speaker regions for a recording, with transcript excerpts.

Helps decide whether a low-similarity stretch is a real other speaker (Q&A/dialogue)
or an acoustic/channel shift of the same speaker (continuous lecture content).

  python regions.py <pointer> [--min 60]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

B = Path("/mnt/raid/projects/drucker-speaker")
sys.path.insert(0, str(B))
from speaker_id import get_model, embed_windows, load_audio, l2norm, two_means  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pointer")
ap.add_argument("--min", type=float, default=60.0, help="min region length (s)")
ap.add_argument("--abs-thr", type=float, default=0.38)
args = ap.parse_args()

model = get_model(B / "models" / "ecapa")
refs = sorted(glob.glob("/mnt/raid/shared/tts/voices/4796_*_normalized.wav"))
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in refs]).mean(0, keepdims=True))

d = np.load(B / "embeddings" / f"{args.pointer}.npz")
emb = d["emb"]
starts = d["starts"][d["keep"]]
sims = (emb @ cent.T).ravel()
lab, cents = two_means(emb, cent)
cs = [(cents[i] @ cent.T).item() for i in range(2)]
dr = 0 if cs[0] >= cs[1] else 1
ok = (lab == dr) & (sims >= args.abs_thr)

segs = []
p = Path.home() / "drucker" / "out" / f"{args.pointer}.json"
if p.exists():
    segs = json.loads(p.read_text())["segments"]

def excerpt(t0, t1, n=240):
    txt = " ".join(s["text"] for s in segs if s["start"] >= t0 and s["end"] <= t1)
    return " ".join(txt.split())[:n]

# contiguous runs of the same label
regions = []
i = 0
while i < len(ok):
    j = i
    while j + 1 < len(ok) and ok[j + 1] == ok[i]:
        j += 1
    t0 = starts[i]
    t1 = starts[j] + 2.0
    if t1 - t0 >= args.min:
        regions.append((t0, t1, bool(ok[i]), sims[i:j + 1].mean()))
    i = j + 1

print(f"{args.pointer}: sims {cs[0]:+.3f}/{cs[1]:+.3f} drucker_cluster={dr}  "
      f"regions >= {args.min:.0f}s: {len(regions)}")
total_dr = sum(t1 - t0 for t0, t1, k, _ in regions if k)
print(f"  Drucker time in regions: {total_dr/60:.1f} min of {d['duration_s']/60:.1f} min\n")
for t0, t1, k, m in regions:
    tag = "DRUCKER" if k else "other  "
    print(f"[{t0/60:6.2f}-{t1/60:6.2f} min] {tag} sim={m:+.2f} dur={(t1-t0)/60:5.2f}min")
    print(f"    {excerpt(t0, t1)}")
