#!/usr/bin/env python3
"""Drucker-vs-others speaker ID over the 80-recording corpus.

Two decoupled stages so thresholds can be tuned without recomputing embeddings:

  embed : sliding-window ECAPA embeddings  -> embeddings/<pointer>.npz
  label : per-recording anchored 2-means   -> labels/<pointer>.json
          (Drucker/other regions + per-window posteriors)
  mask  : write Drucker-only audio         -> audio-drucker/<pointer>.flac

Anchoring: a global "Drucker centroid" is built from the known-good reference clips
(4796 clean reel).  Each recording then runs a 2-means in cosine space seeded by that
centroid, so channel/era differences are absorbed per recording while "which cluster is
Drucker" stays pinned to the reference voice.  A recording whose best cluster still sits
far from the centroid (e.g. 4794, Bill May only) ends up with no Drucker at all.

Usage (one process per GPU):
  python speaker_id.py embed --audio-root ~/drucker/audio --out . --shard 0 --shards 3
  python speaker_id.py label --out . --refs '/mnt/raid/shared/tts/voices/*normalized.wav'
  python speaker_id.py mask  --audio-root ~/drucker/audio --out . --shard 0 --shards 3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
WIN = 2.0
HOP = 0.5
MIN_RMS_DB = -45.0


# --------------------------------------------------------------------------- utils
def db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-12))


def l2norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-9)


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio

        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def get_model(savedir: Path):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # speechbrain < 1.0
        from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cuda"},
    )


def embed_windows(model, wavs: list[np.ndarray], device="cuda", batch=48) -> np.ndarray:
    """wavs: list of float32 mono 16k arrays -> L2-normalised (N,192)."""
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(wavs), batch):
            chunk = wavs[i : i + batch]
            T = max(len(w) for w in chunk)
            x = np.zeros((len(chunk), T), dtype="float32")
            for j, w in enumerate(chunk):
                x[j, : len(w)] = w
            t = torch.from_numpy(x).to(device)
            e = model.encode_batch(t).squeeze(1).float().cpu().numpy()
            out.append(e)
    return l2norm(np.concatenate(out, axis=0)) if out else np.zeros((0, 192), "float32")


def ref_centroid(model, refs: list[str]) -> np.ndarray:
    wavs = [load_audio(Path(p)) for p in refs]
    e = embed_windows(model, wavs)
    c = l2norm(e.mean(axis=0, keepdims=True))
    return c


# --------------------------------------------------------------------------- stage 1
def cmd_embed(args) -> int:
    out = Path(args.out)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)
    files = sorted(audio_root.glob("*.flac"))
    files = [f for i, f in enumerate(files) if i % args.shards == args.shard]
    model = get_model(out / "models" / "ecapa")
    print(f"[embed] shard {args.shard}/{args.shards}: {len(files)} files", flush=True)
    for fi, f in enumerate(files):
        dst = out / "embeddings" / f"{f.stem}.npz"
        if dst.exists() and not args.overwrite:
            print(f"[embed] skip {f.stem}", flush=True)
            continue
        t0 = time.time()
        x = load_audio(f)
        n = max(0, int((len(x) / SR - WIN) / HOP) + 1)
        starts = np.arange(n) * HOP
        wavs, keep, rms = [], [], np.zeros(n, dtype="float32")
        for k, s in enumerate(starts):
            a, b = int(s * SR), int((s + WIN) * SR)
            w = x[a:b]
            r = float(np.sqrt(np.mean(w**2))) if len(w) else 0.0
            rms[k] = db(r)
            if rms[k] >= MIN_RMS_DB and len(w) == int(WIN * SR):
                wavs.append(w)
                keep.append(k)
        t1 = time.time()
        emb = embed_windows(model, wavs)
        dt = time.time() - t0
        np.savez_compressed(
            dst, starts=starts, rms=rms, keep=np.asarray(keep, dtype=np.int32),
            emb=emb, duration_s=len(x) / SR, win=WIN, hop=HOP,
        )
        print(
            f"[embed] {fi + 1}/{len(files)} {f.stem} dur={len(x)/SR/60:.1f}min "
            f"win={len(wavs)} kept={len(keep)} embed={dt - (t1 - t0):.1f}s total={dt:.1f}s",
            flush=True,
        )
    return 0


# --------------------------------------------------------------------------- stage 2
def two_means(E: np.ndarray, c0: np.ndarray, iters: int = 25):
    """Cosine 2-means seeded with c0; returns (labels, centroids)."""
    c1 = E[np.argmin(E @ c0.T).item()] if len(E) else c0
    c0 = c0.copy()
    c1 = c1.copy()
    lab = np.zeros(len(E), dtype=np.int32)
    for _ in range(iters):
        lab = (E @ c1.T).ravel() > (E @ c0.T).ravel()
        lab = lab.astype(np.int32)
        n1 = lab.sum()
        if n1 == 0 or n1 == len(E):
            break
        c0 = l2norm(E[lab == 0].mean(axis=0, keepdims=True))
        c1 = l2norm(E[lab == 1].mean(axis=0, keepdims=True))
    return lab, np.vstack([c0, c1])


def runs_from_windows(starts: np.ndarray, is_spk: np.ndarray, hop: float, win: float,
                      min_region: float, merge_gap: float):
    regions = []
    for i, flag in enumerate(is_spk):
        if not flag:
            continue
        s, e = starts[i], starts[i] + win
        if regions and s - regions[-1][1] <= merge_gap:
            regions[-1][1] = e
        else:
            regions.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in regions if e - s >= min_region]


def cmd_label(args) -> int:
    out = Path(args.out)
    files = sorted((out / "embeddings").glob("*.npz"))
    if args.pointers:
        want = set(args.pointers.split(","))
        files = [f for f in files if f.stem in want]
    model = get_model(out / "models" / "ecapa")
    c_global = ref_centroid(model, glob.glob(args.refs))
    print(f"[label] reference centroid from {len(glob.glob(args.refs))} clips", flush=True)

    (out / "labels").mkdir(parents=True, exist_ok=True)
    summary = []
    for f in files:
        d = np.load(f)
        emb, keep, starts = d["emb"], d["keep"], d["starts"]
        if len(emb) < 8:
            continue
        lab, cents = two_means(emb, c_global)
        sim0 = float((cents[0] @ c_global.T).item())
        sim1 = float((cents[1] @ c_global.T).item())
        drucker_cluster = 0 if sim0 >= sim1 else 1
        s_best, s_other = max(sim0, sim1), min(sim0, sim1)
        # speech gate: a recording with almost no transcribed speech is music/noise, and any
        # "clusters" found in it are meaningless (the 2-means always finds two).
        speech_frac = 0.0
        op = Path(args.orig_transcripts) / f"{f.stem}.json"
        if op.exists():
            od = json.loads(op.read_text())
            sp = sum(x["end"] - x["start"] for x in od.get("segments", []))
            speech_frac = sp / od["duration_s"] if od.get("duration_s") else 0.0
        no_speech = speech_frac < args.min_speech_frac
        if no_speech:
            is_spk = np.zeros(len(keep), dtype=bool)
        # A recording only has a second speaker if the *other* cluster is clearly not a
        # Drucker-like voice.  Two well-separated clusters that both sit high (0.42-0.60)
        # are channel/EQ variants of the same voice (e.g. cassette side changes).
        other_is_speaker = s_other < args.abs_thr
        both_drucker = not other_is_speaker
        if no_speech:
            pass  # is_spk already all-False
        elif s_best < args.abs_thr:
            is_spk = np.zeros(len(keep), dtype=bool)  # no Drucker-like voice in this file
        elif both_drucker:
            is_spk = np.ones(len(keep), dtype=bool)
        else:
            # hysteresis on the per-window similarity gap
            sim_dr = emb @ cents[drucker_cluster].T
            sim_ot = emb @ cents[1 - drucker_cluster].T
            lo = (sim_dr.ravel() - sim_ot.ravel())
            is_spk = (lab == drucker_cluster) & (lo >= args.hyst_lo)
            # median-smooth over 5 windows (2.5 s)
            k = 5
            pad = np.pad(is_spk.astype(np.int8), k // 2, mode="edge")
            is_spk = (np.convolve(pad, np.ones(k), "valid") >= k / 2)
        regions = runs_from_windows(starts[keep] if len(keep) else starts, is_spk,
                                    HOP, WIN, args.min_region, args.merge_gap)
        dur = float(d["duration_s"])
        spk_dur = sum(e - s for s, e in regions)
        rec = {
            "pointer": f.stem,
            "duration_s": dur,
            "n_windows": int(len(emb)),
            "cluster_sim": [round(sim0, 4), round(sim1, 4)],
            "drucker_cluster": int(drucker_cluster),
            "both_drucker_guard": bool(both_drucker),
            "other_is_speaker": bool(other_is_speaker),
            "speech_frac": round(speech_frac, 4),
            "no_speech": bool(no_speech),
            "drucker_regions": regions,
            "drucker_s": round(spk_dur, 2),
            "drucker_frac": round(spk_dur / dur, 4) if dur else 0.0,
            "win": WIN, "hop": HOP,
        }
        (out / "labels" / f"{f.stem}.json").write_text(json.dumps(rec, indent=1))
        summary.append(rec)
        print(
            f"[label] {f.stem} frac={rec['drucker_frac']:.2f} regions={len(regions)} "
            f"sims=({sim0:.2f},{sim1:.2f}) guard={both_drucker}",
            flush=True,
        )
    import csv

    with (out / "labels" / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pointer", "duration_s", "n_windows", "sim_best", "sim_other",
                    "drucker_cluster", "other_is_speaker", "regions", "drucker_s",
                    "drucker_frac"])
        for r in sorted(summary, key=lambda r: -r["drucker_frac"]):
            hi, lo = max(r["cluster_sim"]), min(r["cluster_sim"])
            w.writerow([r["pointer"], r["duration_s"], r["n_windows"], hi, lo,
                        r["drucker_cluster"], r["other_is_speaker"],
                        len(r["drucker_regions"]), r["drucker_s"], r["drucker_frac"]])
    print(f"[label] wrote {len(summary)} label files", flush=True)
    return 0


# --------------------------------------------------------------------------- stage 3
def cmd_mask(args) -> int:
    import soundfile as sf

    out = Path(args.out)
    audio_root = Path(args.audio_root)
    (out / "audio-drucker").mkdir(parents=True, exist_ok=True)
    labels = sorted((out / "labels").glob("*.json"))
    labels = [p for i, p in enumerate(labels) if i % args.shards == args.shard]
    for p in labels:
        rec = json.loads(p.read_text())
        dst = out / "audio-drucker" / f"{rec['pointer']}.flac"
        if dst.exists() and not args.overwrite:
            continue
        x = load_audio(audio_root / f"{rec['pointer']}.flac")
        y = np.zeros_like(x)
        pad = 0.05
        for s, e in rec["drucker_regions"]:
            a = max(0, int((s - pad) * SR))
            b = min(len(x), int((e + pad) * SR))
            y[a:b] = x[a:b]
            # short fades to avoid clicks at the cuts
            n = min(int(0.02 * SR), (b - a) // 2)
            if n > 0:
                y[a:a + n] *= np.linspace(0, 1, n, dtype="float32")
                y[b - n:b] *= np.linspace(1, 0, n, dtype="float32")
        sf.write(str(dst), y, SR, subtype="PCM_16")
        print(f"[mask] {rec['pointer']} kept={rec['drucker_s']}s -> {dst.name}", flush=True)
    return 0


# --------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--audio-root", default=str(Path.home() / "drucker" / "audio"))
    e.add_argument("--out", required=True)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--shards", type=int, default=1)
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(func=cmd_embed)

    l = sub.add_parser("label")
    l.add_argument("--out", required=True)
    l.add_argument("--refs", default="/mnt/raid/shared/tts/voices/4796_*_normalized.wav")
    l.add_argument("--abs-thr", type=float, default=0.30,
                   help="min cosine to the Drucker reference centroid for a cluster to count")
    l.add_argument("--unify-delta", type=float, default=0.10,
                   help="if both clusters are within this and above abs-thr -> single speaker")
    l.add_argument("--hyst-lo", type=float, default=0.0,
                   help="min (sim_drucker - sim_other) for a window to stay Drucker")
    l.add_argument("--min-region", type=float, default=1.0)
    l.add_argument("--merge-gap", type=float, default=0.6)
    l.add_argument("--pointers", default="")
    l.add_argument("--orig-transcripts", default=str(Path.home() / "drucker" / "out"))
    l.add_argument("--min-speech-frac", type=float, default=0.15,
                   help="recordings with less transcribed speech than this are treated as non-speech")
    l.add_argument("--overwrite", action="store_true")
    l.set_defaults(func=cmd_label)

    m = sub.add_parser("mask")
    m.add_argument("--audio-root", default=str(Path.home() / "drucker" / "audio"))
    m.add_argument("--out", required=True)
    m.add_argument("--shard", type=int, default=0)
    m.add_argument("--shards", type=int, default=1)
    m.add_argument("--overwrite", action="store_true")
    m.set_defaults(func=cmd_mask)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
