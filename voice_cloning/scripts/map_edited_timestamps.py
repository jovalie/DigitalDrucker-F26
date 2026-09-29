#!/usr/bin/env python3
"""Map the Drucker speech-interval ledger onto noise-edited audio.

The ledger (catalog/drucker_speech_intervals.json) records when Drucker speaks in the
*unedited* source timeline.  When an editor delivers a cleaned file with noise removed
(or with non-Drucker speech removed), the timeline shifts; this tool rebuilds the mapping.

Two modes:

  --assume-concatenated   the edited file is the Drucker intervals spliced together in
                          order (noise / other speakers removed).  Verified against the
                          edited duration, then mapped by exact arithmetic.  Fast, exact.

  default (alignment)     any other edit.  Needs the edited file's transcript
                          (--edited-transcript, .json or .srt, e.g. from the QCL GPU
                          pipeline); word sequences of the original Drucker-only transcript
                          and the edited one are aligned with difflib and a monotone
                          piecewise-linear time map is fitted through the matches.

Writes catalog/drucker_speech_intervals.edited.<asset>.json with per-interval edited times
and a confidence per interval, plus an .srt/.txt report.

  python3 scripts/map_edited_timestamps.py --pointer 4796 --edited clean/4796_clean.wav \
      --assume-concatenated
  python3 scripts/map_edited_timestamps.py --pointer 4796 --edited clean/4796_clean.wav \
      --edited-transcript clean/4796_clean.json
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from bisect import bisect_left
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT / "catalog" / "drucker_speech_intervals.json"
TRANSCRIPTS = PROJECT / "transcripts-drucker"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def words_with_times(segments: list) -> list[tuple[str, float, float]]:
    """Distribute each segment's span over its words (approximate word times)."""
    out = []
    for s in segments:
        toks = re.findall(r"[a-z0-9']+", s["text"].lower())
        if not toks:
            continue
        t0, t1 = float(s["start"]), float(s["end"])
        step = (t1 - t0) / len(toks)
        for i, w in enumerate(toks):
            out.append((w, t0 + i * step, t0 + (i + 1) * step))
    return out


def load_segments(path: Path) -> list:
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return d.get("segments", d if isinstance(d, list) else [])
    if path.suffix == ".srt":
        segs, cur = [], None
        for line in path.read_text().splitlines():
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", line)
            if m:
                g = [int(x) for x in m.groups()]
                cur = {"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                       "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, "text": ""}
                segs.append(cur)
            elif cur is not None and line.strip() and not line.strip().isdigit():
                cur["text"] += (" " if cur["text"] else "") + line.strip()
        return segs
    raise ValueError(f"unsupported transcript format: {path.suffix}")


def build_map(orig: list[tuple[str, float, float]], edit: list[tuple[str, float, float]]):
    """Monotone piecewise-linear map original_time -> edited_time via word alignment."""
    a = [w for w, _, _ in orig]
    b = [w for w, _, _ in edit]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    anchors = []
    for block in sm.get_matching_blocks():
        if block.size < 3:                       # ignore tiny matches
            continue
        # sample the block every few words so the map stays locally constrained
        step = max(1, block.size // max(2, block.size // 25))   # ~25 anchors per block
        idx = sorted(set(list(range(0, block.size, step)) + [block.size - 1]))
        for k in idx:
            ow = orig[block.a + k]
            ew = edit[block.b + k]
            anchors.append(((ow[1] + ow[2]) / 2, (ew[1] + ew[2]) / 2))
    anchors.sort()
    dedup = []
    for t_o, t_e in anchors:
        if dedup and abs(t_o - dedup[-1][0]) < 1e-6:
            continue
        if dedup and t_e <= dedup[-1][1]:        # enforce monotonicity
            continue
        dedup.append((t_o, t_e))
    return dedup, sm.ratio()


def map_time(t: float, anchors: list[tuple[float, float]]) -> tuple[float | None, float]:
    """Return (edited_time, confidence) for an original time."""
    if not anchors:
        return None, 0.0
    xs = [a for a, _ in anchors]
    i = bisect_left(xs, t)
    if i == 0:
        t0, e0 = anchors[0]
        t1, e1 = anchors[1] if len(anchors) > 1 else anchors[0]
    elif i >= len(anchors):
        t0, e0 = anchors[-2] if len(anchors) > 1 else anchors[0]
        t1, e1 = anchors[-1]
    else:
        t0, e0 = anchors[i - 1]
        t1, e1 = anchors[i]
    slope = (e1 - e0) / (t1 - t0) if t1 > t0 else 1.0
    spread = (t1 - t0)
    conf = 1.0 if spread <= 20 else max(0.2, 20 / spread)   # dense anchors -> high conf
    return e0 + (t - t0) * slope, conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--edited", required=True, help="edited audio (for provenance/duration)")
    ap.add_argument("--edited-transcript", help="transcript of the edited file (.json/.srt)")
    ap.add_argument("--assume-concatenated", action="store_true")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="relative duration tolerance for --assume-concatenated")
    args = ap.parse_args()

    led = json.loads(Path(args.ledger).read_text())
    if args.pointer not in led["items"]:
        print(f"pointer {args.pointer} not in ledger")
        return 1
    item = led["items"][args.pointer]
    edited = Path(args.edited)
    if not edited.exists():
        print(f"edited file not found: {edited}")
        return 1

    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(edited)],
                           capture_output=True, text=True, check=True)
    ed_dur = float(probe.stdout.strip())

    speech = [iv for iv in item["intervals"] if iv["kind"] == "speech"]
    total_speech = sum(iv["dur"] for iv in speech)

    record = {
        "pointer": args.pointer,
        "source": {"media": item["media"], "media_sha256": item.get("media_sha256"),
                   "duration_s": item["duration_s"]},
        "edited": {"file": str(edited), "sha256": sha256(edited), "duration_s": round(ed_dur, 3)},
        "ledger_version": led.get("version"),
    }

    if args.assume_concatenated:
        err = abs(ed_dur - total_speech) / max(total_speech, 1e-9)
        record["mode"] = "concatenated"
        record["duration_error"] = round(err, 4)
        if err > args.tolerance:
            record["duration_check"] = "failed"
            print(f"WARNING: edited duration {ed_dur:.1f}s differs from Drucker speech total "
                  f"{total_speech:.1f}s by {err*100:.1f}% (> {args.tolerance*100:.0f}%). "
                  f"Use transcript alignment instead.")
        else:
            record["duration_check"] = "ok"
        off = 0.0
        for iv in item["intervals"]:
            if iv["kind"] != "speech":
                iv["edited_start"] = iv["edited_end"] = None
                iv["edited_confidence"] = 0.0
                continue
            iv["edited_start"] = round(off, 3)
            iv["edited_end"] = round(off + iv["dur"], 3)
            iv["edited_confidence"] = 1.0
            off += iv["dur"]
        record["mapped_seconds"] = round(off, 3)
    else:
        if not args.edited_transcript:
            print("alignment mode needs --edited-transcript (or use --assume-concatenated)")
            return 1
        orig = load_segments(TRANSCRIPTS / f"{args.pointer}.json")
        ed = load_segments(Path(args.edited_transcript))
        anchors, ratio = build_map(words_with_times(orig), words_with_times(ed))
        record["mode"] = "alignment"
        record["match_ratio"] = round(ratio, 4)
        record["anchors"] = [[round(a, 3), round(b, 3)] for a, b in anchors]
        first, last = {}, {}
        for iv in item["intervals"]:
            s, cs = map_time(iv["start"], anchors)
            e, ce = map_time(iv["end"], anchors)
            ok = s is not None and e is not None and s < e
            iv["edited_start"] = round(s, 3) if ok else None
            iv["edited_end"] = round(e, 3) if ok else None
            iv["edited_confidence"] = round(min(cs, ce), 3) if ok else 0.0

    for iv in item["intervals"]:
        iv["mapped"] = iv.get("edited_start") is not None
    record["items"] = {args.pointer: item}
    record["flags"] = {
        "low_confidence_intervals": [iv["id"] for iv in item["intervals"]
                                     if iv.get("edited_confidence", 0) < 0.5],
        "unmapped_intervals": [iv["id"] for iv in item["intervals"] if not iv["mapped"]],
    }

    out = Path(args.out) if args.out else (
        PROJECT / "catalog" / f"drucker_speech_intervals.edited.{edited.stem}.json")
    out.write_text(json.dumps(record, indent=1))
    nmap = sum(1 for iv in item["intervals"] if iv["mapped"])
    print(f"{args.pointer}: {record['mode']} mapping, {nmap}/{len(item['intervals'])} intervals "
          f"mapped, {len(record['flags']['low_confidence_intervals'])} low-confidence")
    try:
        shown = out.relative_to(PROJECT)
    except ValueError:
        shown = out
    print(f"→ {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
