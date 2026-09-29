#!/usr/bin/env python3
"""Drucker TTS webapp — CosyVoice 2 voice cloning with intonation prompting.

Runs on qclgpu (GPU1), listens on the tailnet, and turns text + an intonation prompt into
speech in Peter Drucker's archived voice.

    GET  /                 simple web UI
    GET  /api/voices       available reference voices (+ transcripts)
    POST /api/tts          {"text": "...", "voice": "...", "mode": "instruct|clone",
                            "instruct": "measured, deliberate...", "speed": 1.0}
    GET  /out/<file>.wav   generated audio
    GET  /health           model/GPU status

Prototype use only: the archive audio is "all rights retained by The Drucker Institute"
(see governance/RIGHTS.md, G9).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

BASE = Path(os.environ.get("DRUCKER_TTS_BASE", "/mnt/raid/shared/tts"))
COSY = Path(os.environ.get("COSYVOICE_DIR", str(BASE / "CosyVoice")))
sys.path.insert(0, str(COSY))
sys.path.insert(0, str(COSY / "third_party" / "Matcha-TTS"))

import torch  # noqa: E402
import torchaudio  # noqa: E402

MODEL_DIR = BASE / "pretrained_models" / "CosyVoice2-0.5B"
VOICES_DIR = BASE / "voices"
OUT_DIR = BASE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_model = None
_model_error: str | None = None


def load_model():
    """Load CosyVoice2 once (takes ~10-30 s and ~3 GB of VRAM)."""
    global _model, _model_error
    if _model is not None:
        return _model
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        _model = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
        _model_error = None
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        _model_error = f"{type(exc).__name__}: {exc}"
    return _model


def voices() -> dict:
    """Voice catalog: voices.json if present, else every wav in voices/."""
    catalog = VOICES_DIR / "voices.json"
    if catalog.exists():
        return json.loads(catalog.read_text())
    out = {}
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        out[wav.stem] = {"file": wav.name, "transcript": "", "note": ""}
    return out


def synth(text: str, voice_id: str, mode: str, instruct: str, speed: float) -> tuple[Path, float]:
    model = load_model()
    if model is None:
        raise RuntimeError(_model_error or "model not loaded")

    cat = voices()
    if voice_id not in cat:
        raise ValueError(f"unknown voice '{voice_id}'")
    entry = cat[voice_id]
    prompt_wav = VOICES_DIR / entry["file"]
    if not prompt_wav.exists():
        raise FileNotFoundError(prompt_wav)

    t0 = time.time()
    chunks = []
    with _lock:  # CosyVoice is not thread-safe and holds ~3 GB of VRAM on GPU1
        if mode == "clone" and entry.get("transcript"):
            # zero-shot clone using the reference clip's own transcript
            gen = model.inference_zero_shot(text, entry["transcript"], str(prompt_wav), stream=False)
        elif instruct.strip():
            # instruct2: natural-language intonation / style control + clone from prompt audio.
            # This revision expects the instruction to end with the <|endofprompt|> token.
            instruct_text = instruct.strip()
            if not instruct_text.endswith("<|endofprompt|>"):
                instruct_text += "<|endofprompt|>"
            gen = model.inference_instruct2(text, instruct_text, str(prompt_wav), stream=False)
        else:
            # no transcript and no instruction -> cross-lingual copy of the voice
            gen = model.inference_cross_lingual(text, str(prompt_wav), stream=False)
        for out in gen:
            chunks.append(out["tts_speech"])
    wav = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]

    out_dir = OUT_DIR
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{voice_id}.wav"
    path = out_dir / name
    torchaudio.save(str(path), wav, model.sample_rate)
    return path, time.time() - t0


# ----------------------------------------------------------------------------- web app
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="Drucker TTS", docs_url="/api/docs")

PRESETS = [
    "",
    "measured and deliberate, with pauses for emphasis",
    "warm and amused, as if telling an anecdote",
    "emphatic and slightly impatient, stressing the key words",
    "slow, professorial, as if explaining to students",
    "quiet and reflective",
]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Drucker TTS</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root { color-scheme: dark; }
 body { font: 15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 0; background:#111; color:#eee; }
 main { max-width: 820px; margin: 0 auto; padding: 28px 20px 60px; }
 h1 { font-size: 20px; margin: 0 0 4px; } .sub { color:#888; margin-bottom:20px; font-size:13px; }
 label { display:block; margin:14px 0 6px; color:#bbb; font-size:13px; text-transform:uppercase; letter-spacing:.04em; }
 textarea, input, select { width:100%; box-sizing:border-box; background:#1c1c1c; color:#eee; border:1px solid #333; border-radius:8px; padding:10px; font:inherit; }
 textarea { min-height:120px; resize:vertical; }
 .row { display:flex; gap:12px; flex-wrap:wrap; } .row > div { flex:1 1 220px; }
 button { margin-top:16px; background:#2b6cb0; color:#fff; border:0; border-radius:8px; padding:11px 18px; font:inherit; font-weight:600; cursor:pointer; }
 button:disabled { opacity:.5; cursor:default; }
 audio { width:100%; margin-top:18px; }
 .status { margin-top:12px; color:#888; font-size:13px; min-height:1.2em; }
 .hist { margin-top:26px; font-size:13px; color:#999; } .hist a { color:#7ab7ff; }
 code { background:#1c1c1c; padding:1px 5px; border-radius:4px; }
</style></head><body><main>
<h1>Drucker TTS</h1>
<div class="sub">CosyVoice 2 · zero-shot clone from the Drucker archive · intonation via text prompt</div>
<label for="text">Text</label>
<textarea id="text" placeholder="Type what Drucker should say…"></textarea>
<div class="row">
  <div><label for="voice">Voice reference</label><select id="voice"></select></div>
  <div><label for="mode">Mode</label><select id="mode">
      <option value="instruct">instruct (prompt the intonation)</option>
      <option value="clone">clone (match the reference clip)</option>
  </select></div>
</div>
<label for="instruct">Intonation prompt</label>
<select id="instruct"></select>
<input id="instructText" placeholder="…or type your own, e.g. 'dryly amused, slower on the last sentence'">
<button id="go">Generate</button>
<div class="status" id="status"></div>
<audio id="player" controls hidden></audio>
<div class="hist" id="hist"></div>
<script>
const $ = (id) => document.getElementById(id);
let voices = {};
async function init(){
  const v = await (await fetch('/api/voices')).json();
  voices = v.voices;
  $('voice').innerHTML = Object.entries(voices).map(([id,o]) =>
    `<option value="${id}">${id} — ${o.note||''}</option>`).join('');
  $('instruct').innerHTML = v.presets.map(p => `<option value="${p}">${p||'(none — natural reading)'}</option>`).join('');
}
$('instruct').addEventListener('change', e => { if(e.target.value) $('instructText').value = e.target.value; });
$('go').addEventListener('click', async () => {
  const text = $('text').value.trim();
  if (!text) { $('status').textContent = 'Type some text first.'; return; }
  $('go').disabled = true; $('status').textContent = 'Generating… (first run loads the model, ~30 s)';
  try {
    const r = await fetch('/api/tts', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, voice: $('voice').value, mode: $('mode').value,
                            instruct: $('instructText').value.trim(), speed: 1.0})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'failed');
    $('player').src = j.url; $('player').hidden = false; $('player').play();
    $('status').textContent = `done in ${j.seconds.toFixed(1)} s · ${j.duration.toFixed(1)} s of audio`;
    $('hist').insertAdjacentHTML('afterbegin', `<div><a href="${j.url}">${j.url.split('/').pop()}</a></div>`);
  } catch (e) { $('status').textContent = 'Error: ' + e.message; }
  finally { $('go').disabled = false; }
});
init();
</script></main></body></html>
"""


class TTSRequest(BaseModel):
    text: str
    voice: str
    mode: str = "instruct"
    instruct: str = ""
    speed: float = 1.0


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/voices")
def api_voices():
    cat = voices()
    return {"voices": cat, "presets": PRESETS}


@app.get("/health")
def health():
    info = {
        "model_dir": str(MODEL_DIR),
        "model_loaded": _model is not None,
        "model_error": _model_error,
        "voices": len(voices()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    return info


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 2000:
        raise HTTPException(400, "text too long (max 2000 chars)")
    try:
        path, seconds = synth(text, req.voice, req.mode, req.instruct, req.speed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    try:
        info = torchaudio.info(str(path))
        duration = info.num_frames / info.sample_rate
    except Exception:
        duration = 0.0
    return {"url": f"/out/{path.name}", "seconds": round(seconds, 2), "duration": round(duration, 2)}


@app.get("/out/{name}")
def out(name: str):
    p = (OUT_DIR / name).resolve()
    if p.parent != OUT_DIR.resolve() or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="audio/wav", filename=name)


def main() -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Drucker TTS webapp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--preload", action="store_true", help="load the model at startup")
    args = ap.parse_args()

    print(f"model dir : {MODEL_DIR}")
    print(f"voices    : {len(voices())}")
    print(f"listening : http://{args.host}:{args.port}")
    if args.preload:
        t0 = time.time()
        load_model()
        print(f"model loaded in {time.time() - t0:.1f}s (error={_model_error})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
