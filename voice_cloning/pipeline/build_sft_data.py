#!/usr/bin/env python3
"""Build the CosyVoice2 SFT dataset from the speaker-verified Drucker audio.

Clips are cut from the *masked* Drucker-only audio (so no other speaker can leak in),
aligned to transcript-segment runs, 3-30 s.  Writes wav.scp / text / utt2spk / spk2utt
in the layout CosyVoice's tools expect, plus utt_index.json for traceability.

  python build_sft_data.py --out /mnt/raid/projects/drucker-speaker/sft --snr-min 20 \
      --cap-min 20 --holdout 4941,4997
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

B = Path("/mnt/raid/projects/drucker-speaker")


def repeats(text: str, n: int = 3, limit: int = 3) -> bool:
    w = re.findall(r"[a-z']+", text.lower())
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(g) and max(g.count(x) for x in set(g)) >= limit


def coverage(t0: float, t1: float, regions: list) -> float:
    cov = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in regions)
    return cov / (t1 - t0) if t1 > t0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B / "sft"))
    ap.add_argument("--audio-root", default=str(B / "audio-drucker"))
    ap.add_argument("--snr-min", type=float, default=20.0)
    ap.add_argument("--cap-min", type=float, default=20.0,
                    help="max minutes of speech taken per recording")
    ap.add_argument("--dur-min", type=float, default=3.0)
    ap.add_argument("--dur-max", type=float, default=30.0)
    ap.add_argument("--holdout", default="4941,4997")
    args = ap.parse_args()

    out = Path(args.out)
    data = out / "data" / "drucker"
    wav_dir = data / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    holdout = {h for h in args.holdout.split(",") if h}

    clarity = {}
    for row in csv.DictReader((B / "voice_clarity.csv").open()):
        try:
            clarity[row["pointer"]] = float(row["snr_db"])
        except Exception:
            pass

    index, stats = [], {}
    for lab_path in sorted((B / "labels").glob("*.json")):
        ptr = lab_path.stem
        snr = clarity.get(ptr, 0.0)
        if snr < args.snr_min:
            continue
        lab = json.loads(lab_path.read_text())
        regions = lab.get("drucker_regions") or []
        tp = B / "transcripts-drucker" / f"{ptr}.json"
        if not regions or not tp.exists():
            continue
        segs = json.loads(tp.read_text())["segments"]
        src = Path(args.audio_root) / f"{ptr}.flac"
        if not segs or not src.exists():
            continue
        x, sr = sf.read(str(src), dtype="float32")
        budget = args.cap_min * 60.0
        used = 0.0
        n_before = len(index)
        for i in range(len(segs)):
            if used >= budget:
                break
            t0 = segs[i]["start"]
            for j in range(i, len(segs)):
                if j > i and segs[j]["start"] - segs[j - 1]["end"] > 0.4:
                    break
                t1 = segs[j]["end"]
                d = t1 - t0
                if d < args.dur_min:
                    continue
                if d > args.dur_max:
                    break
                text = " ".join(s["text"].strip() for s in segs[i:j + 1])
                words = len(re.findall(r"[a-z']+", text.lower()))
                if words < 6 or repeats(text) or coverage(t0, t1, regions) < 0.95:
                    continue
                utt = f"{ptr}_{int(round(t0 * 1000)):07d}"
                y = x[int(t0 * sr):int(t1 * sr)]
                if len(y) < int(1.5 * sr) or float(np.abs(y).max()) < 1e-3:
                    continue
                sf.write(str(wav_dir / f"{utt}.wav"), y, sr, subtype="PCM_16")
                index.append({"utt": utt, "pointer": ptr, "start": round(t0, 2),
                              "end": round(t1, 2), "dur": round(d, 2),
                              "snr_db": snr, "split": "dev" if ptr in holdout else "train",
                              "text": text})
                used += d
                break    # next start index
        stats[ptr] = (snr, used / 60.0, len(index) - n_before)

    for split in ("train", "dev"):
        rows = [r for r in index if r["split"] == split]
        with (data / f"wav.scp.{split}").open("w") as fw, (data / f"text.{split}").open("w") as ft:
            for r in rows:
                fw.write(f"{r['utt']} {wav_dir / (r['utt'] + '.wav')}\n")
                ft.write(f"{r['utt']} {r['text']}\n")
        (data / f"utt2spk.{split}").write_text("".join(f"{r['utt']} drucker\n" for r in rows))
        (data / f"spk2utt.{split}").write_text("drucker " + " ".join(r["utt"] for r in rows) + "\n")
    (data / "utt_index.json").write_text(json.dumps(index, indent=1))

    tot = sum(r["dur"] for r in index) / 3600
    dev = sum(r["dur"] for r in index if r["split"] == "dev") / 3600
    print(f"clips: {len(index)}  audio: {tot:.2f} h (dev {dev:.2f} h, train {tot - dev:.2f} h)")
    print(f"{'ptr':6s} {'snr':>5s} {'used_min':>8s} {'clips':>5s}")
    for p, (snr, mins, n) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"{p:6s} {snr:5.1f} {mins:8.1f} {n:5d}")
    print(f"recordings used: {len(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
