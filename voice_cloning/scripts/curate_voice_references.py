#!/usr/bin/env python3
"""Curate the CosyVoice reference clips from the QCL clip pool.

Takes the ranked clip pool (catalog/drucker_clips_manifest.json), snaps each candidate
window to whole transcript segments so the prompt text matches the prompt audio exactly,
drops clips with Whisper repetition loops, exports archival 44.1 kHz raw + normalized wavs
into voice_reference/, and writes voice_reference/manifest.json.

  python3 scripts/curate_voice_references.py [--top 8]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
POOL = PROJECT / "catalog" / "drucker_clips_manifest.json"
TRANSCRIPTS = PROJECT / "transcripts-drucker"
REFDIR = PROJECT / "voice_reference"
CLARITY = PROJECT / "catalog" / "voice_clarity.csv"


def media_for(pointer: str) -> Path | None:
    import csv
    for row in csv.DictReader(CLARITY.open()):
        if row["pointer"] == pointer:
            p = PROJECT / row["file"]
            if p.exists():
                return p
            for ext in (".mp4", ".f4v"):
                alt = p.with_suffix(ext)
                if alt.exists():
                    return alt
    return None


def snap(pointer: str, start: float, dur: float, max_gap: float = 0.6):
    """Return (t0, t1, text) aligned to whole transcript segments, or None."""
    tp = TRANSCRIPTS / f"{pointer}.json"
    if not tp.exists():
        return None
    segs = json.loads(tp.read_text())["segments"]
    w0, w1 = start, start + dur
    inside = [s for s in segs if s["start"] >= w0 - 0.05 and s["end"] <= w1 + 0.05]
    if not inside:
        return None
    # longest run of segments with only small gaps between them
    best, cur = [], []
    def longer(a, b):
        a = a or []
        b = b or []
        if not a:
            return b
        if not b:
            return a
        return a if (a[-1]["end"] - a[0]["start"]) >= (b[-1]["end"] - b[0]["start"]) else b
    for s in inside:
        if cur and s["start"] - cur[-1]["end"] > max_gap:
            best = longer(best, cur)
            cur = []
        cur.append(s)
    best = longer(best, cur)
    t0, t1 = best[0]["start"], best[-1]["end"]
    if not (4.0 <= t1 - t0 <= 12.0):
        return None
    text = " ".join(s["text"].strip() for s in best)
    words = re.findall(r"[a-z']+", text.lower())
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if grams and max(grams.count(g) for g in set(grams)) >= 3:
        return None                                    # repetition loop
    if len(words) < 12:
        return None
    return round(t0, 2), round(t1 - t0, 2), text


def extract(media: Path, start: float, dur: float, pointer: str):
    raw = REFDIR / f"{pointer}_{int(start)}s_{int(dur)}s_raw.wav"
    norm = REFDIR / f"{pointer}_{int(start)}s_{int(dur)}s_normalized.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                    "-i", str(media), "-vn", "-ac", "1", "-ar", "44100",
                    "-c:a", "pcm_s16le", "-y", str(raw)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw),
                    "-af", "highpass=f=70,loudnorm=I=-18:TP=-1.5:LRA=11",
                    "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(norm)],
                   check=True)
    return raw, norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--pool", default=str(POOL))
    args = ap.parse_args()

    pool = json.loads(Path(args.pool).read_text())
    pool.sort(key=lambda c: -c["score"])
    REFDIR.mkdir(exist_ok=True)
    chosen, seen = [], set()
    for c in pool:
        if c["pointer"] in seen or c.get("snr_db", 0) < 25:
            continue
        s = snap(c["pointer"], c["start"], c["dur"])
        if not s:
            continue
        t0, dur, text = s
        media = media_for(c["pointer"])
        if media is None:
            continue
        raw, norm = extract(media, t0, dur, c["pointer"])
        seen.add(c["pointer"])
        rec = dict(c, start=t0, dur=dur, text=text,
                   raw_file=raw.name, reference_file=norm.name)
        chosen.append(rec)
        print(f"[ref] {c['pointer']} {t0:.1f}s {dur:.1f}s sim={c['sim']} snr={c['snr_db']}")
        print(f"      {text[:120]}")
        if len(chosen) >= args.top:
            break
    (REFDIR / "manifest.json").write_text(json.dumps(chosen, indent=1))
    print(f"\n{len(chosen)} curated reference clips → {REFDIR}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
