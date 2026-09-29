#!/usr/bin/env python3
"""Minimal ComfyUI API client: queue a workflow, wait, download the results.

    # list what a workflow exposes, then set a value and run it
    python3 api-example.py --server http://localhost:8188 --workflow wf.json --describe
    python3 api-example.py --server http://localhost:8188 --workflow wf.json \
        --set '340.inputs.prompt=a management lecture' --set '276.inputs.audio=ref.mp3' \
        --download ./out

`--set` takes `NODE_ID.inputs.FIELD=VALUE`; the value is parsed as JSON when possible
(numbers, booleans, null) and used as a plain string otherwise.  Values are applied to the
API-format workflow, so no re-export is needed for a new prompt or seed.

The workflow must be in **API format** (ComfyUI web UI: Workflow → Export (API)).  A UI-format
workflow (with "definitions"/"nodes") cannot be posted to /prompt; this script detects that
and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(server: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{server}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(server: str, path: str):
    with urllib.request.urlopen(f"{server}{path}", timeout=60) as r:
        return json.load(r)


def apply_set(wf: dict, expr: str) -> str:
    node, rest = expr.split(".", 1)
    field = rest.split(".", 1)[1] if rest.startswith("inputs.") else rest
    if node not in wf:
        raise SystemExit(f"no node '{node}' in the workflow (have: {', '.join(sorted(wf)[:10])}…)")
    raw = expr.split("=", 1)[1]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    wf[node].setdefault("inputs", {})[field] = value
    return f"{node}.inputs.{field} = {value!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="NODE.inputs.FIELD=VALUE")
    ap.add_argument("--describe", action="store_true", help="print nodes + inputs, then exit")
    ap.add_argument("--download", metavar="DIR", help="save produced files here")
    ap.add_argument("--timeout", type=float, default=7200, help="seconds to wait for the prompt")
    args = ap.parse_args()

    wf = json.loads(Path(args.workflow).read_text())
    if "definitions" in wf or any(isinstance(v, dict) and v.get("class_type") is None for v in wf.values()):
        print("This looks like a UI-format workflow. Export it as API format first "
              "(ComfyUI web UI: Workflow → Export (API)).", file=sys.stderr)
        return 2

    if args.describe:
        for nid, node in sorted(wf.items(), key=lambda kv: int(kv[0])):
            print(f"[{nid}] {node.get('class_type')}  {node.get('_meta', {}).get('title', '')}")
            for k, v in (node.get("inputs") or {}).items():
                shown = f"← node {v[0]}" if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) else repr(v)
                print(f"      {k} = {shown}")
        return 0

    for expr in args.set:
        print("set", apply_set(wf, expr))

    try:
        res = post(args.server, "/prompt", {"prompt": wf})
    except urllib.error.HTTPError as e:
        print("POST /prompt failed:", e.code, e.read().decode()[:800], file=sys.stderr)
        return 1
    pid = res.get("prompt_id")
    print("queued prompt_id =", pid, "(number:", res.get("number"), ")")

    t0 = time.time()
    while True:
        if time.time() - t0 > args.timeout:
            print("timed out waiting", file=sys.stderr)
            return 1
        time.sleep(3)
        hist = get(args.server, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            print("status:", status.get("status_str"), "completed:", status.get("completed"))
            if status.get("status_str") == "error":
                for m in status.get("messages", []):
                    if m[0] in ("execution_error", "execution_interrupted"):
                        print("  error:", json.dumps(m[1])[:600])
                return 1
            if args.download:
                out = Path(args.download)
                out.mkdir(parents=True, exist_ok=True)
                n = 0
                for node_out in (entry.get("outputs") or {}).values():
                    for key in ("images", "videos", "gifs", "audio"):
                        for f in node_out.get(key, []):
                            q = (f"filename={urllib.parse.quote(f['filename'])}"
                                 f"&subfolder={urllib.parse.quote(f.get('subfolder', ''))}"
                                 f"&type={f.get('type', 'output')}")
                            dst = out / f["filename"]
                            with urllib.request.urlopen(f"{args.server}/view?{q}", timeout=300) as r, \
                                    dst.open("wb") as fh:
                                fh.write(r.read())
                            print("  saved", dst)
                            n += 1
                print(f"{n} file(s) → {out}")
            return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (used above)
    sys.exit(main())
