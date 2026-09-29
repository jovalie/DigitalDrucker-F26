#!/usr/bin/env python3
"""Inspect the per-window Drucker similarity timeline for one recording, with the
Whisper transcript around each candidate speaker transition.

  python inspect.py <pointer> [<pointer> ...] [--centroid refs]

Prints a 2-minute-bucket similarity trace and, for every cluster flip, a few seconds
of transcript on each side so the transition can be judged by content.
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
ap.add_argument("pointers", nargs="+")
ap.add_argument("--bucket", type=float, default=120.0, help="seconds per timeline row")
ap.add_argument("--ctx", type=float, default=25.0, help="seconds of transcript around a flip")
args = ap.parse_args()

model = get_model(B / "models" / "ecapa")
refs = sorted(glob.glob("/mnt/raid/shared/tts/voices/4796_*_normalized.wav"))
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in refs]).mean(0, keepdims=True))

for ptr in args.pointers:
    d = np.load(B / "embeddings" / f"{ptr}.npz")
    emb = d["emb"]
    starts = d["starts"][d["keep"]]   # emb rows correspond to kept windows only
    sims = (emb @ cent.T).ravel()
    lab, cents = two_means(emb, cent)
    cs = [(cents[i] @ cent.T).item() for i in range(2)]
    dr = 0 if cs[0] >= cs[1] else 1
    is_spk = lab == dr
    tj = B.parent.parent / "digital-drucker"  # placeholder, transcripts read below
    tr_path = Path.home() / "drucker" / "out" / f"{ptr}.json"
    segs = json.loads(tr_path.read_text())["segments"] if tr_path.exists() else []

    print(f"\n{'='*100}\n{ptr}: cluster sims {cs[0]:+.3f}/{cs[1]:+.3f}  drucker_cluster={dr}  "
          f"windows={len(emb)}  duration={d['duration_s']/60:.1f} min")
    # timeline
    nb = int(np.ceil(d["duration_s"] / args.bucket))
    print(f"{'t (min)':>9s} {'mean':>7s} {'min':>7s} {'max':>7s} {'%spk':>5s}  bar")
    for b in range(nb):
        lo, hi = b * args.bucket, (b + 1) * args.bucket
        m = (starts >= lo) & (starts < hi)
        if not m.any():
            continue
        s = sims[m]
        pct = is_spk[m].mean()
        bar = "#" * int(pct * 40)
        print(f"{lo/60:9.1f} {s.mean():+.3f} {s.min():+.3f} {s.max():+.3f} {pct*100:5.0f}  {bar}")

    # flips
    flips = np.where(np.diff(is_spk.astype(int)) != 0)[0]
    if len(flips):
        print(f"\n  transitions (showing transcript +/-{args.ctx:.0f}s):")
    for i in flips[:40]:
        t = starts[i]
        before = " ".join(s["text"] for s in segs if t - args.ctx <= s["end"] <= t)[-220:]
        after = " ".join(s["text"] for s in segs if t <= s["start"] <= t + args.ctx)[:220]
        print(f"   {t/60:7.2f} min  {'->DRUCKER' if is_spk[i+1] else '->other  '} "
              f"sim {sims[i]:+.2f}->{sims[i+1]:+.2f}")
        print(f"        before: ...{before}")
        print(f"        after : {after}...")
