#!/usr/bin/env python3
"""GPU transcription for the Laguna cluster (faster-whisper, large-v3).

Reads a queue file (pointer<TAB>media-path, relative to --media-root), transcribes each
recording on the GPU, and writes transcripts/<pointer>.{txt,srt,json} plus a manifest.
Resumable: recordings whose .json already exists are skipped unless --force.

Runs on a GPU node (inside an sbatch job) and also on CPU for smoke tests:
    python3 transcribe_gpu.py --queue laguna/queue.tsv --media-root media \
        --out transcripts-gpu --model large-v3 --device cuda --compute-type float16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cuda_env() -> None:
    """Point CTranslate2 at pip-installed CUDA libs (nvidia-cublas/cudnn wheels)."""
    try:
        import nvidia.cublas.lib  # type: ignore
        import nvidia.cudnn.lib  # type: ignore

        libs = [os.path.dirname(nvidia.cublas.lib.__file__),
                os.path.dirname(nvidia.cudnn.lib.__file__)]
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")
    except Exception:
        pass


def load_queue(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--media-root", default="media")
    ap.add_argument("--out", default="transcripts-gpu")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pointer", help="transcribe a single pointer")
    ap.add_argument("--shard", type=int, default=0, help="shard index (0-based) for multi-GPU runs")
    ap.add_argument("--shards", type=int, default=1, help="total shards (e.g. 4 GPUs)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    queue = load_queue(Path(args.queue))
    if args.pointer:
        queue = [q for q in queue if q[0] == args.pointer]
    if args.shards > 1:
        queue = [q for i, q in enumerate(queue) if i % args.shards == args.shard]
    if args.limit:
        queue = queue[: args.limit]
    if not queue:
        sys.exit("empty queue")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mname = "manifest.json" if args.shards == 1 else f"manifest-shard{args.shard}-of{args.shards}.json"
    manifest_path = out_dir / mname
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"items": {}}
    items = manifest.setdefault("items", {})

    if args.device == "cuda":
        cuda_env()
    from faster_whisper import WhisperModel
    print(f"loading {args.model} on {args.device}/{args.compute_type} …", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    media_root = Path(args.media_root)
    done = skipped = failed = 0
    t_batch = time.time()
    for n, (ptr, rel) in enumerate(queue, 1):
        src = media_root / rel
        if not src.exists():
            print(f"[{n}/{len(queue)}] {ptr}: MISSING {src}", file=sys.stderr, flush=True)
            failed += 1
            continue
        if (out_dir / f"{ptr}.json").exists() and not args.force:
            skipped += 1
            continue
        t0 = time.time()
        try:
            segments, info = model.transcribe(
                str(src), language=args.language, beam_size=args.beam_size,
                vad_filter=True, condition_on_previous_text=False,
                vad_parameters=dict(min_speech_duration_ms=100, speech_pad_ms=200,
                                    threshold=0.35, min_silence_duration_ms=700),
            )
            segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                    for s in segments if s.text.strip()]
            text = "\n".join(s["text"] for s in segs)
            (out_dir / f"{ptr}.txt").write_text(text + "\n")
            (out_dir / f"{ptr}.srt").write_text(
                "\n".join(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['text']}\n"
                          for i, s in enumerate(segs, 1)))
            wall = time.time() - t0
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5).stdout.strip() or None
            except Exception:
                gpu = None
            payload = {"pointer": ptr, "media": rel, "model": args.model,
                       "language": info.language, "duration_s": round(info.duration, 1),
                       "segments": segs, "host": os.uname().nodename,
                       "gpu": gpu,
                       "rtf": round(info.duration / wall, 2) if wall else None,
                       "created_utc": now()}
            tmp = out_dir / f"{ptr}.json.tmp"
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.rename(out_dir / f"{ptr}.json")
            items[ptr] = {"status": "ok", "duration_s": payload["duration_s"],
                          "wall_s": round(wall, 1), "rtf": payload["rtf"],
                          "segments": len(segs), "chars": len(text),
                          "model": args.model, "finished_utc": now()}
            done += 1
            print(f"[{n}/{len(queue)}] {ptr}: {len(segs)} segments, "
                  f"{payload['rtf']}x realtime ({info.duration/60:.0f} min audio)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            items[ptr] = {"status": "failed", "error": str(exc)[:300], "finished_utc": now()}
            print(f"[{n}/{len(queue)}] {ptr}: FAILED {exc}", file=sys.stderr, flush=True)
        manifest["updated_utc"] = now()
        manifest_path.write_text(json.dumps(manifest, indent=2))

    total = time.time() - t_batch
    print(f"\ndone: {done} transcribed, {skipped} skipped, {failed} failed in {total/60:.1f} min "
          f"→ {out_dir}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
