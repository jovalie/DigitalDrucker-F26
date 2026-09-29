#!/usr/bin/env python3
"""Select clean Drucker-only voice clips from the labeled corpus.

Candidates are windows that sit fully inside a Drucker region, are speech-dense
according to the Drucker-only transcript, and are loud/clean (not clipped, not quiet).
Clips are ranked per recording by a blend of speaker similarity, source clarity and
speech density.  Two outputs:

  clips/               large pool (good for fine-tuning / further curation)
  voice_reference/     curated top clips, normalized to 44.1 kHz mono (for CosyVoice)

  python clip_select.py --out /mnt/raid/projects/drucker-speaker \
      --audio-root ~/drucker/audio --clarity <voice_clarity.csv> [--per-rec 12]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

B = Path("/mnt/raid/projects/drucker-speaker")
sys.path.insert(0, str(B))
from speaker_id import get_model, embed_windows, load_audio, l2norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(B))
ap.add_argument("--audio-root", default=str(Path.home() / "drucker" / "audio"))
ap.add_argument("--clarity", default=str(B / "voice_clarity.csv"))
ap.add_argument("--per-rec", type=int, default=12)
ap.add_argument("--dur-min", type=float, default=4.0)
ap.add_argument("--dur-max", type=float, default=12.0)
ap.add_argument("--sim-min", type=float, default=0.55)
ap.add_argument("--density-min", type=float, default=0.80)
ap.add_argument("--top-ref", type=int, default=8)
args = ap.parse_args()

out = Path(args.out)
(out / "clips").mkdir(parents=True, exist_ok=True)
(out / "voice_reference").mkdir(parents=True, exist_ok=True)

clarity = {}
cp = Path(args.clarity)
if cp.exists():
    for row in csv.DictReader(cp.open()):
        try:
            clarity[row["pointer"]] = {
                "snr": float(row["snr_db"]), "hf": float(row["hf_ratio"]),
                "flat": float(row["spectral_flatness"]), "centroid": float(row["centroid_hz"]),
                "ctype": row.get("content_type", ""), "year": row.get("year", ""),
            }
        except Exception:
            pass

model = get_model(B / "models" / "ecapa")
refs = sorted(glob.glob("/mnt/raid/shared/tts/voices/4796_*_normalized.wav"))
cent = l2norm(embed_windows(model, [load_audio(Path(p)) for p in refs]).mean(0, keepdims=True))

all_clips = []
for lab_path in sorted((out / "labels").glob("*.json")):
    ptr = lab_path.stem
    rec = json.loads(lab_path.read_text())
    regions = rec.get("drucker_regions") or []
    tr_path = out / "transcripts-drucker" / f"{ptr}.json"
    if not regions or not tr_path.exists():
        continue
    segs = json.loads(tr_path.read_text())["segments"]
    if len(segs) < 5:
        continue
    src = Path(args.audio_root) / f"{ptr}.flac"
    x = load_audio(src)
    dur = len(x) / 16000.0
    cl = clarity.get(ptr, {"snr": 20.0, "hf": 0.05, "flat": 0.02, "centroid": 1200.0})
    # window embeddings for similarity scoring (2 s / 1 s hop over the whole file)
    n = max(0, int(dur - 2.0) + 1)
    starts = np.arange(n, dtype=float)
    wavs = [x[int(s * 16000):int(s * 16000) + 32000] for s in starts]
    E = embed_windows(model, wavs)
    sim = (E @ cent.T).ravel()

    cands = []
    for a, b in regions:
        for d in (6.0, 8.0, 10.0, 12.0, 4.0):
            if b - a < d + 0.6:
                continue
            t = a + 0.3
            while t + d <= b - 0.3:
                seg_dur = sum(min(s["end"], t + d) - max(s["start"], t)
                              for s in segs if s["end"] > t and s["start"] < t + d)
                dens = seg_dur / d
                if dens < args.density_min:
                    t += 2.0
                    continue
                m = (starts >= t) & (starts <= t + d - 1.0)
                sm = float(sim[m].min()) if m.any() else 0.0
                y = x[int(t * 16000):int((t + d) * 16000)]
                rms = 20 * np.log10(max(float(np.sqrt((y ** 2).mean())), 1e-9))
                peak = 20 * np.log10(max(float(np.abs(y).max()), 1e-9))
                if sm < args.sim_min or peak > -1.0 or rms < -34 or rms > -10:
                    t += 2.0
                    continue
                txt = " ".join(s["text"] for s in segs if s["end"] > t and s["start"] < t + d)
                score = (sm * 2.0 + min(cl["snr"], 40) / 40.0 + dens
                         + min(cl["hf"] * 10, 1.0) - abs(d - 8.0) / 20.0)
                cands.append({"pointer": ptr, "start": round(t, 2), "dur": d,
                              "sim": round(sm, 3), "rms_db": round(rms, 1),
                              "density": round(dens, 3), "score": round(score, 3),
                              "snr_db": cl["snr"], "year": cl["year"],
                              "text": " ".join(txt.split())})
                t += 2.0
    # keep the best non-overlapping windows per recording
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if any(not (c["start"] + c["dur"] <= k["start"] + 0.5 or c["start"] >= k["start"] + k["dur"] - 0.5)
               for k in kept):
            continue
        kept.append(c)
        if len(kept) >= args.per_rec:
            break
    for i, c in enumerate(kept):
        y = x[int(c["start"] * 16000):int((c["start"] + c["dur"]) * 16000)]
        name = f"{ptr}_{int(c['start'])}s_{int(c['dur'])}s"
        sf.write(out / "clips" / f"{name}.wav", y, 16000, subtype="PCM_16")
        c["file"] = f"clips/{name}.wav"
    all_clips.extend(kept)
    print(f"[clip] {ptr}: {len(kept)} clips (best {kept[0]['score'] if kept else 0})", flush=True)

all_clips.sort(key=lambda c: -c["score"])
(out / "clips_manifest.json").write_text(json.dumps(all_clips, indent=1))
print(f"[clip] pool: {len(all_clips)} clips total across {len({c['pointer'] for c in all_clips})} recordings")

# curated reference set: widest-band, cleanest recordings first
pool = [c for c in all_clips if c["snr_db"] >= 25 and c["density"] >= 0.85]
pool.sort(key=lambda c: -(c["sim"] * 1.5 + min(c["snr_db"], 40) / 40 + c["density"]))
picked, refs_out = [], []
for c in pool:
    if any(k["pointer"] == c["pointer"] for k in picked):   # at most one per recording for variety
        continue
    picked.append(c)
    refs_out.append(c)
    if len(refs_out) >= args.top_ref:
        break
for c in refs_out:
    src = out / c["file"]
    dst = out / "voice_reference" / Path(c["file"]).name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "44100",
                    "-ac", "1", str(dst)], check=True)
    c["reference_file"] = f"voice_reference/{Path(c['file']).name}"
    print(f"[ref] {c['file']} sim={c['sim']} snr={c['snr_db']} :: {c['text'][:90]}", flush=True)
(out / "voice_reference" / "manifest.json").write_text(json.dumps(refs_out, indent=1))
print(f"[ref] wrote {len(refs_out)} curated reference clips")
