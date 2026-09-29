#!/usr/bin/env python3
"""Build Drucker-only voice clips from the speaker-ID labels + Drucker-only transcripts.

Clips are runs of consecutive transcript segments (gaps <= 0.5 s) that lie inside a
Drucker region, so the audio and the prompt text always line up.  Outputs:

  catalog/drucker_clip_pool.json     every candidate clip (times, text, score, snr)
  voice_reference/                   top clips as archival 44.1 kHz raw + normalized wavs
  voice_reference/manifest.json      the curated reference set

  python3 scripts/build_drucker_clips.py [--ref-top 8] [--pool-per-rec 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = PROJECT / "transcripts-drucker"
LABELS = PROJECT / "catalog" / "speaker_labels"
CLARITY = PROJECT / "catalog" / "voice_clarity.csv"
REFDIR = PROJECT / "voice_reference"
POOL_OUT = PROJECT / "catalog" / "drucker_clip_pool.json"


def load_clarity() -> dict:
    out = {}
    for row in csv.DictReader(CLARITY.open()):
        try:
            out[row["pointer"]] = {
                "snr": float(row["snr_db"]), "year": row.get("year", ""),
                "media": row["file"], "ctype": row.get("content_type", ""),
            }
        except Exception:
            pass
    return out


def repeats(text: str, n: int = 3, limit: int = 3) -> bool:
    w = re.findall(r"[a-z']+", text.lower())
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(grams) and max(grams.count(g) for g in set(grams)) >= limit


def in_region(t0: float, t1: float, regions: list, min_cov: float = 0.9) -> bool:
    """True if [t0,t1] is covered by Drucker regions (allowing small tolerance)."""
    covered = 0.0
    for a, b in regions:
        covered += max(0.0, min(b, t1) - max(a, t0))
    return covered >= (t1 - t0) * min_cov


def candidates(pointer: str, snr: float, year: str, min_d: float, max_d: float):
    tp = TRANSCRIPTS / f"{pointer}.json"
    lp = LABELS / f"{pointer}.json"
    if not tp.exists() or not lp.exists():
        return []
    segs = json.loads(tp.read_text())["segments"]
    regions = json.loads(lp.read_text()).get("drucker_regions") or []
    if not segs or not regions:
        return []
    out = []
    for i in range(len(segs)):
        t0 = segs[i]["start"]
        for j in range(i, len(segs)):
            if j > i and segs[j]["start"] - segs[j - 1]["end"] > 0.5:
                break
            t1 = segs[j]["end"]
            d = t1 - t0
            if d < min_d:
                continue
            if d > max_d:
                break
            text = " ".join(s["text"].strip() for s in segs[i:j + 1])
            words = len(re.findall(r"[a-z']+", text.lower()))
            if words < 12 or repeats(text):
                continue
            if not in_region(t0, t1, regions):
                continue
            score = (min(snr, 45) / 45.0
                     + (1.0 - abs(d - 8.0) / 8.0) * 0.6
                     + min(words / 40.0, 1.0) * 0.4)
            out.append({"pointer": pointer, "start": round(t0, 2), "dur": round(d, 2),
                        "text": text, "words": words, "snr_db": snr, "year": year,
                        "score": round(score, 4)})
    return out


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
    ap.add_argument("--ref-top", type=int, default=8)
    ap.add_argument("--pool-per-rec", type=int, default=20)
    ap.add_argument("--dur-min", type=float, default=4.0)
    ap.add_argument("--dur-max", type=float, default=12.0)
    args = ap.parse_args()

    clarity = load_clarity()
    pool = []
    per_rec = {}
    for p in sorted(TRANSCRIPTS.glob("*.json")):
        ptr = p.stem
        if ptr not in clarity:
            continue
        cl = clarity[ptr]
        cands = candidates(ptr, cl["snr"], cl["year"], args.dur_min, args.dur_max)
        cands.sort(key=lambda c: -c["score"])
        # keep the best non-overlapping clips
        kept = []
        for c in cands:
            if any(c["start"] < k["start"] + k["dur"] and c["start"] + c["dur"] > k["start"]
                   for k in kept):
                continue
            kept.append(c)
            if len(kept) >= args.pool_per_rec:
                break
        per_rec[ptr] = kept
        pool.extend(kept)
        print(f"[clip] {ptr} snr={cl['snr']:.0f} {cl['year']}: {len(kept)} clips "
              f"(from {len(cands)} candidates)")
    POOL_OUT.write_text(json.dumps(pool, indent=1))
    print(f"\npool: {len(pool)} clips from {len(per_rec)} recordings → {POOL_OUT.relative_to(PROJECT)}")

    # curated references: one per recording, best score, wide-band recordings only
    REFDIR.mkdir(exist_ok=True)
    ranked = sorted((c for c in pool if c["snr_db"] >= 28), key=lambda c: -c["score"])
    chosen, seen = [], set()
    for c in ranked:
        if c["pointer"] in seen:
            continue
        media = PROJECT / clarity[c["pointer"]]["media"]
        if not media.exists():
            media = media.with_suffix(".mp4")
        if not media.exists():
            continue
        raw, norm = extract(media, c["start"], c["dur"], c["pointer"])
        seen.add(c["pointer"])
        chosen.append(dict(c, raw_file=raw.name, reference_file=norm.name))
        print(f"[ref] {c['pointer']} {c['year']} {c['start']:.1f}s {c['dur']:.1f}s "
              f"snr={c['snr_db']:.0f} :: {c['text'][:100]}")
        if len(chosen) >= args.ref_top:
            break
    (REFDIR / "manifest.json").write_text(json.dumps(chosen, indent=1))
    print(f"\n{len(chosen)} reference clips → voice_reference/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
