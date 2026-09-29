#!/usr/bin/env python3
"""Word error rate of the synthesized eval samples (does fine-tuning hurt intelligibility?)."""
import glob
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel

S = Path("/mnt/raid/projects/drucker-speaker/sft/eval")
texts = json.loads((S / "texts.json").read_text())
model = WhisperModel("large-v3", device="cuda", compute_type="float16")


def norm(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


out = {}
for d in sorted(glob.glob(str(S / "wav_*"))):
    tag = Path(d).name.replace("wav_", "")
    errs, hyps = [], []
    for i, w in enumerate(sorted(glob.glob(d + "/*.wav"))):
        segs, _ = model.transcribe(w, language="en", beam_size=5, vad_filter=False)
        hyp = " ".join(s.text for s in segs).strip()
        e = wer(texts[min(i, len(texts) - 1)], hyp)
        errs.append(round(e, 3))
        hyps.append(hyp[:60])
    out[tag] = {"wer_mean": round(sum(errs) / len(errs), 4) if errs else None, "wers": errs,
                "sample_hyp": hyps[0] if hyps else ""}
    print(f"{tag:>6s} WER {out[tag]['wer_mean']} :: {out[tag]['sample_hyp']}")

(S / "wer.json").write_text(json.dumps(out, indent=1))
