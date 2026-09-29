#!/usr/bin/env python3
"""Export the canonical ledger of Drucker-speech intervals for the corpus.

The ledger is the durable record of *when Drucker is speaking* in each source recording.
It is written against the unedited source timeline and tied to the source media by
checksum, so that when noise-edited / re-cut audio arrives the intervals can be mapped
onto the new files (see scripts/map_edited_timestamps.py).

Outputs
  catalog/drucker_speech_intervals.json   canonical ledger (per recording)
  catalog/drucker_speech_intervals.csv    flat table, one row per interval
  catalog/drucker_labels/<ptr>.txt        Audacity label track (start<TAB>end<TAB>DRUCKER)

  python3 scripts/export_speech_intervals.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LABELS = PROJECT / "catalog" / "speaker_labels"
TRANSCRIPTS = PROJECT / "transcripts-drucker"
CLARITY = PROJECT / "catalog" / "voice_clarity.csv"
SIMS = PROJECT / "catalog" / "drucker_interval_sims.json"
LEDGER = PROJECT / "catalog" / "drucker_speech_intervals.json"
FLAT = PROJECT / "catalog" / "drucker_speech_intervals.csv"
LABELDIR = PROJECT / "catalog" / "drucker_labels"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_for(t0: float, t1: float, segs: list, min_cov: float = 0.8) -> tuple[str, int]:
    """Transcript text of the segments lying inside [t0,t1] (>= min_cov of their span)."""
    keep = []
    for s in segs:
        ov = max(0.0, min(s["end"], t1) - max(s["start"], t0))
        span = max(1e-6, s["end"] - s["start"])
        if ov / span >= min_cov:
            keep.append(s)
    return " ".join(s["text"].strip() for s in keep), len(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checksums", action="store_true",
                    help="compute media sha256 (slow, first run only)")
    args = ap.parse_args()

    media = {}
    for row in csv.DictReader(CLARITY.open()):
        media[row["pointer"]] = row["file"]

    sims = json.loads(SIMS.read_text()) if SIMS.exists() else {}

    old = {}
    if LEDGER.exists():
        old = json.loads(LEDGER.read_text()).get("items", {})

    ledger = {
        "version": 1,
        "generated_by": "scripts/export_speech_intervals.py",
        "timeline": ("source media audio track, unedited; seconds from start of file. "
                     "times are valid for the 16 kHz FLAC derived from the same track"),
        "speaker_pass": ("speaker_id.py: ECAPA window embeddings (2 s / 0.5 s hop) anchored to "
                         "the 4796 reel centroid; per-recording cosine 2-means; abs_thr=0.38"),
        "interval_kinds": {"speech": "interval containing transcribed Drucker speech",
                           "gap": "masked-in window with no transcribed speech (breath/noise)"},
        "items": {},
    }
    rows = []
    for p in sorted(LABELS.glob("*.json")):
        ptr = p.stem
        rec = json.loads(p.read_text())
        regions = rec.get("drucker_regions") or []
        segs = []
        tp = TRANSCRIPTS / f"{ptr}.json"
        if tp.exists():
            segs = json.loads(tp.read_text()).get("segments", [])
        item = {
            "media": media.get(ptr, ""),
            "duration_s": rec["duration_s"],
            "speech_frac": rec.get("speech_frac"),
            "no_speech": rec.get("no_speech", False),
            "drucker_seconds": rec["drucker_s"],
            "drucker_frac": rec["drucker_frac"],
            "cluster_sim": rec["cluster_sim"],
            "other_is_speaker": rec.get("other_is_speaker"),
            "intervals": [],
        }
        for i, (t0, t1) in enumerate(regions, 1):
            txt, nseg = text_for(t0, t1, segs)
            iid = f"{ptr}-{i:04d}"
            kind = "speech" if txt.strip() else "gap"
            item["intervals"].append({
                "id": iid,
                "kind": kind,
                "start": round(t0, 3),
                "end": round(t1, 3),
                "dur": round(t1 - t0, 3),
                "sim": sims.get(ptr, {}).get(iid),
                "segments": nseg,
                "text": txt,
            })
            rows.append([ptr, iid, f"{t0:.3f}", f"{t1:.3f}", f"{t1 - t0:.3f}",
                         sims.get(ptr, {}).get(iid, ""), kind, nseg, media.get(ptr, ""), txt])
        # checksum of the source media (cached across runs)
        if args.checksums and item["media"]:
            mp = PROJECT / item["media"]
            if mp.exists():
                prev = old.get(ptr, {})
                if prev.get("media_sha256") and prev.get("media") == item["media"]:
                    item["media_sha256"] = prev["media_sha256"]
                else:
                    item["media_sha256"] = sha256(mp)
        ledger["items"][ptr] = item

    LEDGER.write_text(json.dumps(ledger, indent=1))
    with FLAT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pointer", "interval_id", "start_s", "end_s", "dur_s", "sim",
                    "kind", "segments", "media", "text"])
        w.writerows(rows)

    LABELDIR.mkdir(exist_ok=True)
    for ptr, item in ledger["items"].items():
        with (LABELDIR / f"{ptr}.txt").open("w") as fh:
            for iv in item["intervals"]:
                if iv["kind"] == "speech":
                    fh.write(f"{iv['start']:.3f}\t{iv['end']:.3f}\tDRUCKER\n")

    tot = sum(i["drucker_seconds"] for i in ledger["items"].values())
    nsp = sum(1 for r in rows if r[6] == "speech")
    print(f"ledger: {len(ledger['items'])} recordings, {len(rows)} intervals "
          f"({nsp} speech / {len(rows) - nsp} gap), {tot/3600:.2f} h of Drucker speech")
    print(f"  {LEDGER.relative_to(PROJECT)}")
    print(f"  {FLAT.relative_to(PROJECT)}")
    print(f"  {LABELDIR.relative_to(PROJECT)}/  (Audacity label tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
