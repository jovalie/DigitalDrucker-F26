#!/usr/bin/env python3
"""Fetch the ComfyUI models listed in models.manifest.json.

Resumable (curl -C -) and sha256-verified, so it is safe to re-run after an interrupted
download or to repair a truncated file.

    python3 scripts/fetch-models.py --models-dir /path/to/ComfyUI/models [--only ltx]
    python3 scripts/fetch-models.py --list
    python3 scripts/fetch-models.py --verify            # check what is already there
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "models.manifest.json"
HF = "https://huggingface.co"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: int) -> str:
    return f"{n / 2**30:.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", help="ComfyUI models/ directory")
    ap.add_argument("--only", choices=["ltx", "minimax"], help="fetch just one group")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only check existing files")
    args = ap.parse_args()

    models = json.loads(MANIFEST.read_text())["models"]
    if args.only:
        models = [m for m in models if m["group"] == args.only]

    if args.list:
        for m in models:
            print(f"{human(m['size']):>10}  {m['path']}   [{m['group']}]")
        print(f"total {human(sum(m['size'] for m in models))} over {len(models)} files")
        return 0

    if not args.models_dir:
        ap.error("--models-dir is required unless --list is used")
    root = Path(args.models_dir)
    ok = bad = missing = 0
    for m in models:
        dst = root / m["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == m["size"]:
            print(f"  present  {m['path']}")
            ok += 1
            continue
        if args.verify:
            print(f"  MISSING  {m['path']}")
            missing += 1
            continue
        url = f"{HF}/{m['repo']}/resolve/main/{m['file']}"
        print(f"  fetching {m['path']}  ({human(m['size'])})")
        rc = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
                             "--fail", "-o", str(dst), url]).returncode
        if rc != 0 or not dst.exists():
            print(f"    download FAILED (curl {rc})")
            bad += 1
            continue
        got = sha256(dst)
        if got != m["sha256"]:
            print(f"    sha256 MISMATCH\n      got  {got}\n      want {m['sha256']}")
            bad += 1
        else:
            print(f"    ok ({human(dst.stat().st_size)})")
            ok += 1

    print(f"\n{ok} ok, {bad} failed, {missing} missing "
          f"(of {len(models)} files, {human(sum(m['size'] for m in models))})")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
