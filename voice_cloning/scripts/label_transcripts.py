#!/usr/bin/env python3
"""Annotate the full transcripts with a speaker decision per segment.

The speaker-ID pass produced (a) Drucker-only transcripts and (b) per-recording regions.
This adds the missing piece: every segment of the *original* transcripts gets a
`speaker` field, so the multi-speaker files can be read or filtered in place.

    speaker = "drucker"   >= 60 % of the segment lies inside a Drucker region
              "other"     <= 30 %
              "uncertain" in between (kept, but flagged)

Outputs
  transcripts-labeled/<ptr>.json   original segments + speaker + drucker_coverage
  transcripts-labeled/<ptr>.txt    one line per segment, prefixed [DRUCKER]/[OTHER]/[?]
  catalog/transcript_speakers.csv  flat table: pointer, start, end, speaker, text
  catalog/transcript_speakers.md   summary (per recording + corpus totals)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "transcripts-qcl"
LABELS = PROJECT / "catalog" / "speaker_labels"
OUT = PROJECT / "transcripts-labeled"
CSV_OUT = PROJECT / "catalog" / "transcript_speakers.csv"
MD_OUT = PROJECT / "catalog" / "transcript_speakers.md"

TAG = {"drucker": "DRUCKER", "other": "OTHER", "uncertain": "?"}


def coverage(t0: float, t1: float, regions: list) -> float:
    if t1 <= t0:
        return 0.0
    cov = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in regions)
    return min(1.0, cov / (t1 - t0))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = []
    per_file = []

    for tp in sorted(SRC.glob("*.json")):
        if tp.name.startswith("manifest"):
            continue
        ptr = tp.stem
        doc = json.loads(tp.read_text())
        lp = LABELS / f"{ptr}.json"
        regions = []
        if lp.exists():
            lab = json.loads(lp.read_text())
            regions = [tuple(r) for r in (lab.get("drucker_regions") or [])]

        counts = {"drucker": 0, "other": 0, "uncertain": 0}
        secs = {"drucker": 0.0, "other": 0.0, "uncertain": 0.0}
        out_segs = []
        lines = []
        for s in doc.get("segments", []):
            cov = coverage(s["start"], s["end"], regions)
            spk = "drucker" if cov >= 0.60 else ("other" if cov <= 0.30 else "uncertain")
            counts[spk] += 1
            secs[spk] += max(0.0, s["end"] - s["start"])
            seg = dict(s, speaker=spk, drucker_coverage=round(cov, 3))
            out_segs.append(seg)
            lines.append(f"[{TAG[spk]}] {s['text'].strip()}")
            rows.append([ptr, f"{s['start']:.2f}", f"{s['end']:.2f}", spk,
                         f"{cov:.2f}", s["text"].strip()])

        doc["segments"] = out_segs
        doc["speaker_pass"] = ("speaker_id.py regions; segment tagged drucker >=60% coverage, "
                               "other <=30%, uncertain between")
        (OUT / f"{ptr}.json").write_text(json.dumps(doc, indent=1))
        (OUT / f"{ptr}.txt").write_text("\n".join(lines) + "\n")
        per_file.append((ptr, counts, secs))

    with CSV_OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pointer", "start", "end", "speaker", "drucker_coverage", "text"])
        w.writerows(rows)

    tot_c = {k: sum(c[k] for _, c, _ in per_file) for k in ("drucker", "other", "uncertain")}
    tot_s = {k: sum(s[k] for _, _, s in per_file) for k in ("drucker", "other", "uncertain")}
    lines = ["# Transcript speaker annotation", "",
             f"{len(per_file)} recordings, {len(rows)} segments", "",
             "| class | segments | speech |", "|---|---:|---:|"]
    for k in ("drucker", "other", "uncertain"):
        lines.append(f"| {k} | {tot_c[k]:,} | {tot_s[k]/3600:.2f} h |")
    lines += ["", "## Recordings with the most non-Drucker speech", "",
              "| pointer | drucker | other | uncertain | other h |", "|---|---:|---:|---:|---:|"]
    ranked = sorted(per_file, key=lambda r: -(r[2]["other"]))
    for ptr, c, s in ranked[:15]:
        lines.append(f"| {ptr} | {c['drucker']} | {c['other']} | {c['uncertain']} | {s['other']/3600:.2f} |")
    MD_OUT.write_text("\n".join(lines) + "\n")

    print(f"wrote {len(per_file)} annotated transcript pairs -> {OUT.relative_to(PROJECT)}/")
    print(f"  {CSV_OUT.relative_to(PROJECT)}  ({len(rows):,} segments)")
    print(f"  {MD_OUT.relative_to(PROJECT)}")
    print(f"corpus totals: drucker {tot_c['drucker']:,} segs / {tot_s['drucker']/3600:.1f} h | "
          f"other {tot_c['other']:,} / {tot_s['other']/3600:.1f} h | "
          f"uncertain {tot_c['uncertain']:,} / {tot_s['uncertain']/3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
