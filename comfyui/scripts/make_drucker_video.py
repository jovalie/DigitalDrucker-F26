#!/usr/bin/env python3
"""Produce a Drucker talking-head video through the ComfyUI HTTP API.

The graph is the shipped LTX-2.3 ID-LoRA workflow (int8 transformer + distilled LoRA + identity
LoRA), exported to API format, so no browser is needed. Everything that matters is a parameter:

    # one shot, speech synthesized by the Drucker TTS portal
    python3 make_drucker_video.py --image drucker.png \
        --speech "The effective executive focuses on contribution." \
        --duration 10 --out drucker_10s.mp4

    # long video from an existing waveform: render 13 s pieces and stitch them
    python3 make_drucker_video.py --image drucker.png --audio speech.wav \
        --duration 52 --chunk-seconds 13 --out drucker_52s.mp4

Long clips (>~15 s) run out of memory in the x2 upscale stage — attention cost grows with the
square of the frame count — so `--chunk-seconds` is the recommended way to make anything longer.

Other knobs: --fps, --width/--height (output size; the base latent is half of it), --seed,
--server, --template, --node-set NODE.inputs.FIELD=VALUE, --no-upload (files already on the
server), --dry-run, --timeout.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_TEMPLATE = (Path(__file__).resolve().parent.parent /
                    "workflows" / "api" / "ltx23_id_lora.template.api.json")


# ------------------------------------------------------------------ graph helpers
def find(prompt: dict, class_type: str, where=None) -> list[tuple[str, dict]]:
    out = []
    for nid, n in prompt.items():
        if isinstance(n, dict) and n.get("class_type") == class_type:
            if where is None or where(n):
                out.append((nid, n))
    return out


def source_of(prompt: dict, value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
        return prompt.get(value[0])
    return None


def walk_to_primitive(prompt: dict, value):
    n = source_of(prompt, value)
    for _ in range(6):
        if n is None:
            return None
        if "value" in (n.get("inputs") or {}):
            return n
        nxt = None
        for _, v in (n.get("inputs") or {}).items():
            cand = source_of(prompt, v)
            if cand is not None:
                nxt = cand
                break
        n = nxt
    return None


def patch(prompt: dict, args) -> list[str]:
    log = []
    for nid, n in find(prompt, "LoadImage"):
        if args.image_name:
            n["inputs"]["image"] = args.image_name
            log.append(f"image  -> {args.image_name} [{nid}]")
    for nid, n in find(prompt, "LoadAudio"):
        if args.audio_name:
            n["inputs"]["audio"] = args.audio_name
            log.append(f"audio  -> {args.audio_name} [{nid}]")

    if getattr(args, "speech", None):
        for nid, n in find(prompt, "PrimitiveStringMultiline",
                           where=lambda x: "[SPEECH]" in str(x.get("inputs", {}).get("value", ""))):
            head, rest = n["inputs"]["value"].split("[SPEECH]:", 1)
            tail = rest.split("\n\n[SOUNDS]", 1)
            sounds = ("\n\n[SOUNDS]" + tail[1]) if len(tail) > 1 else ""
            n["inputs"]["value"] = f"{head}[SPEECH]: {args.speech.strip()}{sounds}"
            log.append(f"speech -> {args.speech[:60]!r} [{nid}]")

    empty = find(prompt, "EmptyLTXVLatentVideo")
    if empty and any(v is not None for v in (args.width, args.height, args.duration, args.fps)):
        _, n = empty[0]
        for field, want in (("width", args.width), ("height", args.height)):
            if want is None:
                continue
            expr = source_of(prompt, n["inputs"].get(field))
            prim = walk_to_primitive(prompt, expr["inputs"].get("values.a")) if expr else None
            if prim is not None:
                prim["inputs"]["value"] = int(want)
                log.append(f"{field} -> {want}")
        if "length" in n["inputs"]:
            expr = source_of(prompt, n["inputs"]["length"])
            if expr is not None:
                if args.duration is not None:
                    prim = walk_to_primitive(prompt, expr["inputs"].get("values.a"))
                    if prim is not None:
                        prim["inputs"]["value"] = float(args.duration)
                        log.append(f"duration -> {args.duration:.2f}s")
                prim = walk_to_primitive(prompt, expr["inputs"].get("values.b"))
                if prim is not None:
                    prim["inputs"]["value"] = int(args.fps)
                    log.append(f"fps -> {args.fps}")

    if args.seed is not None:
        for i, (nid, n) in enumerate(find(prompt, "RandomNoise")):
            n["inputs"]["noise_seed"] = int(args.seed) + i
            log.append(f"seed -> {int(args.seed) + i} [{nid}]")
    if args.prefix:
        for nid, n in find(prompt, "SaveVideo"):
            n["inputs"]["filename_prefix"] = args.prefix
            log.append(f"prefix -> {args.prefix} [{nid}]")

    for expr in (args.node_set or []):
        nid, rest = expr.split(".", 1)
        field = rest.split(".", 1)[1] if rest.startswith("inputs.") else rest
        raw = expr.split("=", 1)[1]
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        prompt[nid].setdefault("inputs", {})[field] = val
        log.append(f"set {nid}.inputs.{field} = {val!r}")
    return log


# ------------------------------------------------------------------ http helpers
def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)


def upload(server: str, path: Path) -> str:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = io.BytesIO()

    def part(name: str, value: str):
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

    part("type", "input")
    part("overwrite", "true")
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode())
    body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
    body.write(path.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{server}/upload/image", data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=900) as r:
        info = json.load(r)
    sub = info.get("subfolder") or ""
    return f"{sub}/{info.get('name', path.name)}" if sub else info.get("name", path.name)


def synthesise(tts_url: str, text: str, voice: str, out: Path) -> Path:
    res = post_json(f"{tts_url.rstrip('/')}/api/tts", {"text": text, "voice": voice, "mode": "clone"})
    url = res.get("url")
    if not url:
        raise SystemExit(f"TTS returned no url: {res}")
    with urllib.request.urlopen(f"{tts_url.rstrip('/')}{url}", timeout=600) as r:
        out.write_bytes(r.read())
    print(f"  tts -> {out} ({out.stat().st_size/1024:.0f} KB)")
    return out


# ------------------------------------------------------------------ ffmpeg helpers
def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def split_audio(path: Path, seconds: float, work: Path) -> list[Path]:
    for p in work.glob("chunk_*.wav"):
        p.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "segment",
                    "-segment_time", str(seconds), "-c", "copy",
                    str(work / "chunk_%02d.wav")], check=True)
    return sorted(work.glob("chunk_*.wav"))


def motion_report(path: Path, fps: int = 25, threshold: float = 0.4) -> float:
    """Per-second inter-frame motion; LTX sometimes renders a near-static take.

    Measured as the mean |frame(t) - frame(t-1)| luma. Two takes of the same 21 s clip with
    different seeds scored median 0.34 (visibly frozen) vs 1.73 (lively), so it is worth
    checking every render.
    """
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          "tblend=all_mode=difference,signalstats,"
                          "metadata=print:key=lavfi.signalstats.YAVG:file=-",
                          "-f", "null", "-"], capture_output=True, text=True).stdout
    vals = [float(x) for x in re.findall(r"YAVG=([0-9.]+)", out)]
    if not vals:
        return float("nan")
    secs = [sum(vals[i:i + fps]) / len(vals[i:i + fps]) for i in range(0, len(vals), fps)]
    static = [i for i, v in enumerate(secs) if v < threshold]
    print("== motion check: " + " ".join(f"{v:.1f}" for v in secs))
    print(f"   median {sorted(secs)[len(secs)//2]:.2f} | near-static seconds {len(static)}/{len(secs)}"
          + ("   <-- re-run with a different --seed" if len(static) > len(secs) / 4 else ""))
    return sorted(secs)[len(secs) // 2]


def concat_videos(videos: list[Path], out: Path) -> Path:
    lst = out.parent / "concat.txt"
    lst.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-y", str(out)], check=True)
    return out


# ------------------------------------------------------------------ one render
def render_one(args, prompt: dict) -> int:
    print("== patching workflow")
    for line in patch(prompt, args):
        print("   " + line)
    if args.dry_run:
        print("== dry run: nothing queued")
        return 0

    if not args.no_upload:
        if args.image and Path(args.image).exists():
            name = upload(args.server, Path(args.image))
            print(f"== uploaded image -> {name}")
            for _, n in find(prompt, "LoadImage"):
                n["inputs"]["image"] = name
        if args.audio and Path(args.audio).exists():
            name = upload(args.server, Path(args.audio))
            print(f"== uploaded audio -> {name}")
            for _, n in find(prompt, "LoadAudio"):
                n["inputs"]["audio"] = name

    print(f"== queueing on {args.server}")
    try:
        res = post_json(f"{args.server}/prompt", {"prompt": prompt})
    except urllib.error.HTTPError as e:
        print("queue failed:", e.code, e.read().decode()[:900], file=sys.stderr)
        return 1
    pid = res["prompt_id"]
    print(f"   prompt_id {pid}")

    t0 = time.time()
    while True:
        if time.time() - t0 > args.timeout:
            print("timed out waiting for the prompt", file=sys.stderr)
            return 1
        time.sleep(5)
        try:
            hist = get_json(f"{args.server}/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            continue
        entry = hist[pid]
        status = (entry.get("status") or {}).get("status_str")
        if status == "error":
            for m in (entry.get("status") or {}).get("messages", []):
                if m[0] in ("execution_error", "execution_interrupted"):
                    info = m[1] if isinstance(m[1], dict) else {}
                    print(f"ERROR in {info.get('node_type')}: "
                          f"{str(info.get('exception_message'))[:200]}", file=sys.stderr)
            return 1
        files = [f for node_out in (entry.get("outputs") or {}).values()
                 for key in ("images", "gifs", "videos", "audio")
                 for f in node_out.get(key, [])]
        if not files:
            continue
        print(f"== done in {time.time() - t0:.0f}s, {len(files)} file(s)")
        if not args.out:
            print("   " + ", ".join(f["filename"] for f in files))
            return 0
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for idx, f in enumerate(files):
            q = (f"filename={urllib.parse.quote(f['filename'])}"
                 f"&subfolder={urllib.parse.quote(f.get('subfolder', ''))}"
                 f"&type={f.get('type', 'output')}")
            with urllib.request.urlopen(f"{args.server}/view?{q}", timeout=3600) as r:
                data = r.read()
            dst = out if len(files) == 1 else out.parent / f["filename"]
            dst.write_bytes(data)
            print(f"   saved {dst} ({len(data)/1e6:.2f} MB)")
            if args.check_motion:
                motion_report(dst)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--image", help="reference portrait (png/jpg)")
    ap.add_argument("--audio", help="speech wav; if omitted, synthesized from --speech")
    ap.add_argument("--speech", help="text to speak ([SPEECH] section of the prompt)")
    ap.add_argument("--tts-url", default="http://127.0.0.1:8190", help="Drucker TTS portal")
    ap.add_argument("--tts-voice", default="drucker_1981_reel_10s")
    ap.add_argument("--width", type=int, help="output width (base latent is half)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--duration", type=float, help="seconds of video")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prefix", default=None, help="SaveVideo filename_prefix")
    ap.add_argument("--node-set", action="append", metavar="NODE.inputs.FIELD=VALUE")
    ap.add_argument("--out", help="where to save the produced video")
    ap.add_argument("--normalize-audio", action="store_true",
                    help="loudness-normalize the speech first (CosyVoice output can contain quiet "
                         "passages, and quiet audio produces a nearly motionless render)")
    ap.add_argument("--chunk-seconds", type=float,
                    help="render in pieces of this many seconds and stitch (long videos OOM in "
                         "the x2 upscale stage; ~13 s is safe on a 45 GB L40S)")
    ap.add_argument("--no-upload", action="store_true",
                    help="files are already in the server's input dir")
    ap.add_argument("--timeout", type=float, default=14400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-motion", action="store_true",
                    help="after rendering, report per-second motion and flag a near-static take "
                         "(LTX motion varies a lot with the seed; needs ffmpeg locally)")
    args = ap.parse_args()

    tpl = Path(args.template)
    if not tpl.exists():
        print(f"template not found: {tpl}", file=sys.stderr)
        return 2
    template = tpl.read_text()
    if args.prefix is None:
        args.prefix = f"video/drucker_{time.strftime('%Y%m%d-%H%M%S')}"
    work = Path(args.out).parent if args.out else Path(".")
    work.mkdir(parents=True, exist_ok=True)

    # speech: synthesize if needed
    audio = Path(args.audio) if args.audio else None
    if audio is None and args.speech:
        audio = work / f"tts_{int(time.time())}.wav"
        print(f"== synthesizing speech via {args.tts_url}")
        synthesise(args.tts_url, args.speech, args.tts_voice, audio)
    args.audio = str(audio) if audio else None

    tmpl_prompt = json.loads(template)
    args.image_name = Path(args.image).name if args.image else None
    if args.image_name is None:
        keep = find(tmpl_prompt, "LoadImage")
        args.image_name = keep[0][1]["inputs"]["image"] if keep else None
        print(f"== keeping template image: {args.image_name}")
    args.audio_name = audio.name if audio else None
    if args.audio_name is None:
        keep = find(tmpl_prompt, "LoadAudio")
        args.audio_name = keep[0][1]["inputs"]["audio"] if keep else None
        print(f"== keeping template audio: {args.audio_name}")

    if args.normalize_audio and audio:
        norm = work / "speech_normalized.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-af",
                        "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", "24000", "-y", str(norm)],
                       check=True)
        print(f"== normalized speech -> {norm}")
        audio = norm
        args.audio = str(norm)

    if args.chunk_seconds and audio:
        parts = split_audio(audio, args.chunk_seconds, work)
        print(f"== chunked mode: {len(parts)} pieces of {args.chunk_seconds:.0f}s")
        videos = []
        for i, part in enumerate(parts):
            sub = argparse.Namespace(**vars(args))
            sub.duration = probe_duration(part)
            sub.seed = (args.seed + i) if args.seed is not None else None
            sub.out = str(work / f"part_{i:02d}.mp4")
            sub.audio = str(part)
            sub.audio_name = part.name
            sub.prefix = f"{args.prefix}_p{i:02d}"
            sub.chunk_seconds = None
            print(f"\n== chunk {i + 1}/{len(parts)} ({sub.duration:.1f}s) {part.name}")
            rc = render_one(sub, json.loads(template))
            if rc:
                print(f"chunk {i + 1} failed", file=sys.stderr)
                return rc
            if not args.dry_run:
                videos.append(Path(sub.out))
        if args.dry_run:
            print("== dry run: nothing queued")
            return 0
        concat_videos(videos, Path(args.out))
        print(f"\n== stitched {len(videos)} chunks -> {args.out} "
              f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")
        return 0

    return render_one(args, tmpl_prompt)


if __name__ == "__main__":
    sys.exit(main())
