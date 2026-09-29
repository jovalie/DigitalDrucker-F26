#!/usr/bin/env python3
"""Produce a Drucker talking-head video through the ComfyUI HTTP API.

The graph is the shipped LTX-2.3 ID-LoRA workflow (int8 transformer + distilled LoRA +
identity LoRA), exported to API format — no browser needed. Everything that matters is a
parameter:

    python3 make_drucker_video.py \
        --image drucker.png \
        --speech "The effective executive focuses on contribution." \
        --duration 20 --width 1280 --height 720 \
        --out drucker_20s.mp4

Speech is taken from `--audio <wav>` if given, otherwise synthesized from `--speech` via the
Drucker TTS portal (`--tts-url`, `--tts-voice`) so one command goes text -> voice -> video.

Other knobs: --fps, --seed, --server, --template, --node-set (raw overrides), --no-upload
(assume the server already has the files), --wait/--timeout, --dry-run (print what it would do).
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_TEMPLATE = (Path(__file__).resolve().parent.parent /
                    "workflows" / "api" / "ltx23_id_lora.template.api.json")


# ---------------------------------------------------------------- helpers
def node(prompt: dict, nid: str) -> dict:
    return prompt.get(nid) or {}


def source_of(prompt: dict, value):
    """Resolve an input that is either a literal or a ['node_id', slot] link."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
        return prompt.get(value[0])
    return None


def find(prompt: dict, class_type: str, where=None) -> list[tuple[str, dict]]:
    out = []
    for nid, n in prompt.items():
        if not isinstance(n, dict) or n.get("class_type") != class_type:
            continue
        if where is None or where(n):
            out.append((nid, n))
    return out


def walk_to_primitive(prompt: dict, value, key: str | None = None):
    """Follow link(s) until a node with a literal 'value' input is found."""
    n = source_of(prompt, value)
    for _ in range(6):
        if n is None:
            return None
        if "value" in (n.get("inputs") or {}):
            return n
        nxt = None
        for k, v in (n.get("inputs") or {}).items():
            if key is None or k == key:
                cand = source_of(prompt, v)
                if cand is not None:
                    nxt = cand
                    break
        n = nxt
    return None


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def upload(server: str, path: Path, subfolder: str = "") -> str:
    """Upload a file into the server's input directory (what the UI does for image/audio widgets)."""
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = io.BytesIO()

    def part(name: str, value: str):
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    part("type", "input")
    part("overwrite", "true")
    if subfolder:
        part("subfolder", subfolder)
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode())
    body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
    body.write(path.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(f"{server}/upload/image", data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        info = json.load(r)
    name = info.get("name", path.name)
    sub = info.get("subfolder") or ""
    return f"{sub}/{name}" if sub else name


def synthesise(tts_url: str, text: str, voice: str, out: Path) -> Path:
    """Ask the Drucker TTS portal for a wav of `text` (CosyVoice clone)."""
    res = post_json(f"{tts_url.rstrip('/')}/api/tts",
                    {"text": text, "voice": voice, "mode": "clone"})
    url = res.get("url")
    if not url:
        raise SystemExit(f"TTS returned no url: {res}")
    with urllib.request.urlopen(f"{tts_url.rstrip('/')}{url}", timeout=600) as r:
        out.write_bytes(r.read())
    print(f"  tts: {text[:60]!r} -> {out} ({out.stat().st_size/1024:.0f} KB)")
    return out


# ---------------------------------------------------------------- parameter patching
def set_input(prompt: dict, nid: str, field: str, value) -> str:
    prompt[nid].setdefault("inputs", {})[field] = value
    return f"{nid}.inputs.{field}"


def patch(prompt: dict, args) -> list[str]:
    log = []

    # image / audio
    for nid, n in find(prompt, "LoadImage"):
        n["inputs"]["image"] = args.image_name
        log.append(f"image  -> {args.image_name} [{nid}]")
    for nid, n in find(prompt, "LoadAudio"):
        n["inputs"]["audio"] = args.audio_name
        log.append(f"audio  -> {args.audio_name} [{nid}]")

    # speech text inside the [SPEECH] section of the big prompt primitive
    if args.speech:
        for nid, n in find(prompt, "PrimitiveStringMultiline",
                           where=lambda x: "[SPEECH]" in str(x.get("inputs", {}).get("value", ""))):
            txt = n["inputs"]["value"]
            head, rest = txt.split("[SPEECH]:", 1)
            tail = rest.split("\n\n[SOUNDS]", 1)
            sounds = ("\n\n[SOUNDS]" + tail[1]) if len(tail) > 1 else ""
            n["inputs"]["value"] = f"{head}[SPEECH]: {args.speech.strip()}{sounds}"
            log.append(f"speech -> {args.speech[:60]!r} [{nid}]")

    # resolution / fps / duration via the math nodes that feed EmptyLTXVLatentVideo
    empty = find(prompt, "EmptyLTXVLatentVideo")
    if empty and (args.width or args.height or args.duration or args.fps):
        nid, n = empty[0]
        for field, want, key in (("width", args.width, "values.a"),
                                 ("height", args.height, "values.a"),
                                 ("length", args.fps, "values.b")):
            if want is None:
                continue
            expr = source_of(prompt, n["inputs"].get(field))
            if expr is None:
                continue
            if field == "length":           # length = seconds * fps + 1
                if args.duration is not None:
                    prim = walk_to_primitive(prompt, expr["inputs"].get("values.a"))
                    if prim is not None:
                        prim["inputs"]["value"] = float(args.duration)
                        log.append(f"duration -> {args.duration}s [{prim.get('_meta', {}).get('title', '')}]")
                prim = walk_to_primitive(prompt, expr["inputs"].get("values.b"))
                if prim is not None:
                    prim["inputs"]["value"] = int(args.fps)
                    log.append(f"fps -> {args.fps}")
            else:
                prim = walk_to_primitive(prompt, expr["inputs"].get(key))
                if prim is not None:
                    prim["inputs"]["value"] = int(want)
                    log.append(f"{field} -> {want}")

    # seed + output name
    if args.seed is not None:
        for i, (nid, n) in enumerate(find(prompt, "RandomNoise")):
            n["inputs"]["noise_seed"] = int(args.seed) + i
            log.append(f"seed -> {int(args.seed) + i} [{nid}]")
    if args.prefix:
        for nid, n in find(prompt, "SaveVideo"):
            n["inputs"]["filename_prefix"] = args.prefix
            log.append(f"prefix -> {args.prefix} [{nid}]")

    for expr in args.node_set or []:
        nid, rest = expr.split(".", 1)
        field = rest.split(".", 1)[1] if rest.startswith("inputs.") else rest
        raw = expr.split("=", 1)[1]
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        log.append("set " + set_input(prompt, nid, field, val))
    return log


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
    ap.add_argument("--width", type=int, help="output width (base latent is half this)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--duration", type=float, help="seconds of video")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prefix", default=None, help="SaveVideo filename_prefix")
    ap.add_argument("--node-set", action="append", metavar="NODE.inputs.FIELD=VALUE")
    ap.add_argument("--out", help="where to save the produced video")
    ap.add_argument("--no-upload", action="store_true",
                    help="files are already in the server's input dir")
    ap.add_argument("--timeout", type=float, default=14400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tpl = Path(args.template)
    if not tpl.exists():
        print(f"template not found: {tpl}", file=sys.stderr)
        return 2
    prompt = json.loads(tpl.read_text())

    work = Path(args.out).parent if args.out else Path(".")
    work.mkdir(parents=True, exist_ok=True)

    # speech audio: synthesize if needed, then upload
    audio_path = Path(args.audio) if args.audio else None
    if audio_path is None and args.speech:
        audio_path = work / f"tts_{int(time.time())}.wav"
        print(f"== synthesizing speech via {args.tts_url}")
        synthesise(args.tts_url, args.speech, args.tts_voice, audio_path)
    if audio_path is None:
        # no --audio and no --speech: keep whatever the template points at
        audio_path = None

    args.image_name = Path(args.image).name if args.image else None
    args.audio_name = Path(audio_path).name if audio_path else None
    if args.image_name is None:
        keep = find(prompt, "LoadImage")
        args.image_name = keep[0][1]["inputs"]["image"] if keep else None
        print(f"== keeping template image: {args.image_name}")
    if args.audio_name is None:
        keep = find(prompt, "LoadAudio")
        args.audio_name = keep[0][1]["inputs"]["audio"] if keep else None
        print(f"== keeping template audio: {args.audio_name}")
    if args.prefix is None:
        args.prefix = f"video/drucker_{time.strftime('%Y%m%d-%H%M%S')}"

    print("== patching workflow")
    for line in patch(prompt, args):
        print("   " + line)

    if args.dry_run:
        print("== dry run: not uploading, not queueing")
        return 0

    if not args.no_upload:
        if args.image and Path(args.image).exists():
            name = upload(args.server, Path(args.image))
            print(f"== uploaded image -> {name}")
            for _, n in find(prompt, "LoadImage"):
                n["inputs"]["image"] = name
        if audio_path and audio_path.exists():
            name = upload(args.server, audio_path)
            print(f"== uploaded audio -> {name}")
            for _, n in find(prompt, "LoadAudio"):
                n["inputs"]["audio"] = name

    print(f"== queueing on {args.server}")
    try:
        res = post_json(f"{args.server}/prompt", {"prompt": prompt})
    except urllib.error.HTTPError as e:
        print("queue failed:", e.code, e.read().decode()[:1000], file=sys.stderr)
        return 1
    pid = res.get("prompt_id")
    print(f"   prompt_id {pid}")

    t0 = time.time()
    last = ""
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
                    print("ERROR:", json.dumps(m[1])[:800], file=sys.stderr)
            return 1
        files = []
        for node_out in (entry.get("outputs") or {}).values():
            for key in ("images", "gifs", "videos", "audio"):
                for f in node_out.get(key, []):
                    files.append(f)
        if not files:
            continue
        print(f"== done in {time.time() - t0:.0f}s, {len(files)} file(s)")
        if not args.out:
            print("   " + ", ".join(f["filename"] for f in files))
            return 0
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for f in files:
            q = (f"filename={urllib.parse.quote(f['filename'])}"
                 f"&subfolder={urllib.parse.quote(f.get('subfolder', ''))}"
                 f"&type={f.get('type', 'output')}")
            with urllib.request.urlopen(f"{args.server}/view?{q}", timeout=1800) as r:
                data = r.read()
            if len(files) == 1:
                out.write_bytes(data)
                print(f"   saved {out} ({len(data)/1e6:.2f} MB)")
            else:
                p = out.parent / f["filename"]
                p.write_bytes(data)
                print(f"   saved {p} ({len(data)/1e6:.2f} MB)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
