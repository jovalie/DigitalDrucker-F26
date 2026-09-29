#!/usr/bin/env python3
"""Export the curated Drucker voice-reference clips at archival quality (local).

Reads the clip selection made on QCL (catalog/drucker_clips_manifest.json), pulls each
window from the original media (44.1 kHz source) and writes, matching the existing
voice_reference conventions:

  voice_reference/<ptr>_<start>s_<dur>s_raw.wav         PCM16 44.1 kHz mono
  voice_reference/<ptr>_<start>s_<dur>s_normalized.wav  highpass 70 Hz, -18 LUFS

  python3 scripts/export_drucker_clips.py                  # curated references only
  python3 scripts/export_drucker_clips.py --all-pool       # every clip in the manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "catalog" / "drucker_clips_manifest.json"
CLARITY = PROJECT / "catalog" / "voice_clarity.csv"
OUT = PROJECT / "voice_reference"


def media_for(pointer: str) -> Path | None:
    for row in csv.DictReader(CLARITY.open()):
        if row["pointer"] == pointer:
            p = PROJECT / row["file"]
            if p.exists():
                return p
            alt = p.with_suffix(".mp4")
            if alt.exists():
                return alt
    return None


def extract(media: Path, start: float, dur: float, pointer: str) -> tuple[Path, Path]:
    raw = OUT / f"{pointer}_{int(start)}s_{int(dur)}s_raw.wav"
    norm = OUT / f"{pointer}_{int(start)}s_{int(dur)}s_normalized.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                    "-i", str(media), "-vn", "-ac", "1", "-ar", "44100",
                    "-c:a", "pcm_s16le", "-y", str(raw)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw),
                    "-af", "highpass=f=70,loudnorm=I=-18:TP=-1.5:LRA=11",
                    "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(norm)],
                   check=True)
    return raw, norm


def curate(pool: list[dict], top: int = 8) -> list[dict]:
    """Pick the reference set: clean, wide-band recordings, one clip per recording."""
    ok = [c for c in pool if c.get("snr_db", 0) >= 25 and c.get("density", 0) >= 0.85]
    ok.sort(key=lambda c: -c["score"])
    out, seen = [], set()
    for c in ok:
        if c["pointer"] in seen:
            continue
        seen.add(c["pointer"])
        out.append(c)
        if len(out) >= top:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-pool", action="store_true",
                    help="export the whole clip pool, not just the curated references")
    ap.add_argument("--top", type=int, default=8, help="curated reference clips")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    pool = json.loads(Path(args.manifest).read_text())
    if args.all_pool:
        data = pool
    else:
        data = curate(pool, args.top)
        (OUT / "manifest.json").write_text(json.dumps(data, indent=1))
    print(f"exporting {len(data)} clips")
    missing = 0
    for c in data:
        media = media_for(c["pointer"])
        if media is None:
            print(f"  MISSING media for {c['pointer']}")
            missing += 1
            continue
        raw, norm = extract(media, c["start"], c["dur"], c["pointer"])
        c["raw_file"] = raw.name
        c["reference_file"] = norm.name
        print(f"  {c['pointer']} {c['start']:.1f}s {c['dur']:.0f}s sim={c['sim']} "
              f"snr={c['snr_db']} → {norm.name}")
    if not args.all_pool:
        (OUT / "manifest.json").write_text(json.dumps(data, indent=1))
    print(f"done ({missing} missing media)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
