#!/usr/bin/env python3
"""Cut N long Drucker-only audio clips (default 5 x 120 s) from the corpus.

Windows are taken from inside a *single* Drucker region, so they contain no other speaker at all
(the speaker-ID pass bounded the regions). Sources are picked for fidelity (SNR + spectral
centroid from catalog/voice_clarity.csv) among recordings that have a long enough region, and the
audio is cut from the original media at 44.1 kHz rather than the 16 kHz analysis copies.

  python3 scripts/make_drucker_audio_clips.py --count 5 --seconds 120 --out drucker_clips
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LABELS = PROJECT / "catalog" / "speaker_labels"
CLARITY = PROJECT / "catalog" / "voice_clarity.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="drucker_clips")
    ap.add_argument("--pointers", default="",
                    help="comma-separated pointers to force (default: pick the best)")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    clarity = {r["pointer"]: r for r in csv.DictReader(CLARITY.open())}
    forced = [p for p in args.pointers.split(",") if p]

    cands = []
    for p, row in clarity.items():
        lp = LABELS / f"{p}.json"
        if not lp.exists():
            continue
        regs = (json.loads(lp.read_text()).get("drucker_regions") or [])
        if not regs:
            continue
        s, e = max(regs, key=lambda r: r[1] - r[0])
        if e - s < args.seconds + 20:          # keep a margin inside the region
            continue
        media = PROJECT / row["file"]
        if not media.exists():
            continue
        # fidelity score: wide band and clean wins; era is a tie-break only
        score = (float(row["snr_db"]) / 40.0 + min(float(row["centroid_hz"]), 1800) / 1800.0
                 + min(float(row["hf_ratio"]) * 10, 1.0))
        cands.append((score, p, s, e, row))

    if forced:
        chosen = [c for c in cands if c[1] in forced][:args.count]
    else:
        cands.sort(key=lambda c: -c[0])
        chosen, used_year = [], set()
        for c in cands:                        # prefer one clip per recording and per year
            if c[4]["year"] in used_year and len(chosen) < args.count:
                continue
            chosen.append(c)
            used_year.add(c[4]["year"])
            if len(chosen) == args.count:
                break

    out = PROJECT / args.out
    out.mkdir(exist_ok=True)
    print(f"{'ptr':6s} {'year':>5s} {'snr':>5s} {'centroid':>9s} {'window':>18s}  file")
    made = []
    for score, p, s, e, row in chosen:
        dur = e - s
        start = s + min(60.0, max(0.0, (dur - args.seconds) / 2))
        media = PROJECT / row["file"]
        raw = out / f"drucker_{p}_{row['year']}_{int(start)}s_{int(args.seconds)}s_raw.wav"
        dst = out / f"drucker_{p}_{row['year']}_{int(start)}s_{int(args.seconds)}s.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{args.seconds:.2f}",
                        "-i", str(media), "-vn", "-ac", "1", "-ar", "44100",
                        "-c:a", "pcm_s16le", "-y", str(raw)], check=True)
        if args.no_normalize:
            raw.replace(dst)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw), "-af",
                            "highpass=f=70,loudnorm=I=-18:TP=-1.5:LRA=11",
                            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(dst)],
                           check=True)
            raw.unlink()
        made.append({"pointer": p, "year": row["year"], "start": round(start, 2),
                     "seconds": args.seconds, "snr_db": float(row["snr_db"]),
                     "centroid_hz": float(row["centroid_hz"]), "file": dst.name,
                     "source": row["file"], "title": row["title"]})
        print(f"{p:6s} {row['year']:>5s} {row['snr_db']:>5s} {row['centroid_hz']:>9s} "
              f"{start:9.0f}-{start+args.seconds:<8.0f}  {dst.name}")
    (out / "manifest.json").write_text(json.dumps(made, indent=1))
    total = sum(f.stat().st_size for f in out.glob("*.wav")) / 1e6
    print(f"\n{len(made)} clips, {args.seconds:.0f} s each, {total:.1f} MB → {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
