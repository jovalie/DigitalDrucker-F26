"""Drucker TTS — CosyVoice 2 voice clone demo.

Type a line, pick one of the reference voices taken from Peter Drucker's 1981 reel-to-reel
seminars, and hear it spoken in his voice. Zero-shot cloning from a reference clip, with the
CosyVoice 2 LLM fine-tuned on 12 h of speaker-verified Drucker speech.
"""
import spaces  # noqa: F401  — must be imported before torch/CUDA-touching imports

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
COSYVOICE = Path("/tmp/CosyVoice")
MODEL_DIR = Path("/tmp/cosyvoice2-drucker")
BASE_REPO = "FunAudioLLM/CosyVoice2-0.5B"
FINETUNED_REPO = os.environ.get("DRUCKER_MODEL_REPO", "jovalie/Drucker-CosyVoice2-0.5B")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

VOICES = json.loads((HERE / "voices.json").read_text())
DEFAULT_VOICE = next(iter(VOICES))

EXAMPLES = [
    "The best way to predict the future is to create it.",
    "Management is doing things right; leadership is doing the right things.",
    "The most important thing in communication is to hear what is not being said.",
    "There is nothing so useless as doing efficiently that which should not be done at all.",
    "Knowledge has to be improved, challenged, and increased constantly, or it vanishes.",
]


def _stub_optional_deps() -> None:
    """Put this Space's stand-ins for heavyweight optional deps on sys.path.

    cosyvoice2.yaml is resolved with hyperpyyaml, which imports modules such as
    cosyvoice.dataset.processor and cosyvoice.tokenizer.tokenizer just to look up class names —
    and those import openai-whisper (a ~2 GB build chain) and pyworld (source build) at module
    level. Neither is installed here; the stubs/ directory ships instead. See stubs/README.md.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))


def _patch_torchaudio_io() -> None:
    """Let torchaudio.load work without torchcodec.

    Recent torchaudio delegates file I/O to torchcodec, which the ZeroGPU image does not ship.
    CosyVoice only ever loads 16 kHz wavs, so read them with soundfile and hand back the same
    (channels, samples) tensor torchaudio.load would.
    """
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio, "_drucker_patched", False):
        return

    def _load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr

    torchaudio.load = _load
    torchaudio._drucker_patched = True


def _setup() -> None:
    """Clone the CosyVoice code and fetch the weights (network + disk only, no CUDA)."""
    _stub_optional_deps()
    _patch_torchaudio_io()
    if not (COSYVOICE / "cosyvoice").exists():
        subprocess.run(["git", "clone", "--recursive", "--depth", "1",
                        "https://github.com/FunAudioLLM/CosyVoice.git", str(COSYVOICE)], check=True)
    for path in (COSYVOICE, COSYVOICE / "third_party" / "Matcha-TTS"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # frontend.py hard-imports openai-whisper for the CosyVoice **1** tokenizer path; CosyVoice 2
    # uses the ONNX speech tokenizer instead, so make that import optional rather than pulling a
    # ~2 GB dependency (and its build chain) into the Space.
    frontend = COSYVOICE / "cosyvoice" / "cli" / "frontend.py"
    if frontend.exists():
        src = frontend.read_text()
        if "import whisper" in src and "whisper = None" not in src:
            src = src.replace(
                "import whisper\n",
                "try:\n    import whisper\nexcept ModuleNotFoundError:"
                "  # CosyVoice-1 path only\n    whisper = None\n", 1)
            frontend.write_text(src)

    # transformers >= 5 instantiates from_pretrained() in the checkpoint's saved dtype, and the
    # Qwen2.5 config behind CosyVoice's LLM says bfloat16 — while CosyVoice builds every
    # activation in float32, so generation dies on "mat1 and mat2 must have the same dtype".
    # Force the LLM back to float32 (which is also the dtype of the fine-tuned llm.pt).
    llm_src = COSYVOICE / "cosyvoice" / "llm" / "llm.py"
    if llm_src.exists():
        src = llm_src.read_text()
        old_line = "self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)"
        if old_line in src and "_drucker_dtype" not in src:
            src = src.replace(old_line, (
                "import transformers as _tf\n"
                "        _drucker_dtype = \"dtype\" if int(_tf.__version__.split(\".\")[0]) >= 5 "
                "else \"torch_dtype\"\n"
                "        self.model = Qwen2ForCausalLM.from_pretrained("
                "pretrain_path, **{_drucker_dtype: torch.float32})"), 1)
            llm_src.write_text(src)
            print("[setup] forced the Qwen2 LLM to float32", flush=True)

    # transformers >= 5 requires the attention mask to span the whole (cached + new) sequence.
    # CosyVoice's step-by-step decode passes a mask covering only the current chunk, so after the
    # first step the model attended to the wrong keys and produced garbled speech (transformers 4
    # used to expand that mask for you). Rebuild it as all-visible before each cached step.
    llm_py = COSYVOICE / "cosyvoice" / "llm" / "llm.py"
    if llm_py.exists():
        src = llm_py.read_text()
        old = """    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model("""
        new = """    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        # _drucker_patch: keep the cached prefix visible on every step
        _past = 0
        if cache is not None:
            _past = cache.get_seq_length() if hasattr(cache, "get_seq_length") else (
                cache[0][0].shape[-2] if len(cache) else 0)
        if input_masks.shape[1] != _past + xs.shape[1]:
            input_masks = torch.ones((xs.shape[0], _past + xs.shape[1]),
                                     dtype=torch.bool, device=xs.device)
        outs = self.model("""
        if old in src and "_drucker_patch" not in src:
            llm_py.write_text(src.replace(old, new, 1))
            print("[setup] patched the cached-decode attention mask", flush=True)

    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(BASE_REPO, local_dir=str(MODEL_DIR), token=HF_TOKEN)
    # the fine-tuned LLM ships in this repo (private Space) — it is the only file that differs
    local_llm = HERE / "llm.pt"
    if local_llm.exists() and not (MODEL_DIR / "llm.pt").exists():
        import shutil

        shutil.copy2(local_llm, MODEL_DIR / "llm.pt")
    if FINETUNED_REPO:
        # only the fine-tuned LLM differs from the base model
        snapshot_download(FINETUNED_REPO, local_dir=str(MODEL_DIR), token=HF_TOKEN,
                          allow_patterns=["llm.pt", "*.json", "*.yaml"])


_setup()

_model = None


def _get_model():
    """Load CosyVoice 2 once. Weights are pulled into VRAM on the first @spaces.GPU entry."""
    global _model
    if _model is None:
        import numpy
        import torch
        import transformers

        from cosyvoice.cli.cosyvoice import CosyVoice2

        import gradio
        import huggingface_hub
        import spaces as _spaces

        print(f"[drucker] torch={torch.__version__} transformers={transformers.__version__} "
              f"numpy={numpy.__version__} gradio={gradio.__version__} "
              f"hub={huggingface_hub.__version__} spaces={getattr(_spaces, '__version__', '?')}",
              flush=True)
        _model = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    return _model


def _gpu_seconds(text, voice, speed=1.0):
    """Reserve only as much GPU time as this request needs (free ZeroGPU quota is ~300 s/day).

    The first call of a session also pays for loading CosyVoice 2 into VRAM, hence the floor.
    """
    return int(min(150, 60 + len(str(text)) // 8))


@spaces.GPU(duration=_gpu_seconds)
def synthesize(text: str, voice: str, speed: float = 1.0) -> str:
    """Synthesize `text` in Peter Drucker's voice, cloned from the reference clip `voice`.

    Args:
        text: what he should say (English).
        voice: which reference recording to clone (see the dropdown).
        speed: playback rate multiplier; 1.0 is the recorded pace.
    """
    import torch

    text = (text or "").strip()
    if not text:
        raise gr.Error("Type something for Drucker to say.")
    if len(text) > 600:
        raise gr.Error("Keep it under 600 characters so the take stays short.")

    model = _get_model()
    ref = VOICES[voice]
    prompt_wav = HERE / "voices" / ref["file"]
    chunks = [out["tts_speech"] for out in
              model.inference_zero_shot(text, ref["transcript"], str(prompt_wav), stream=False)]
    wav = torch.cat(chunks, dim=1)

    if abs(speed - 1.0) > 1e-3:
        import torchaudio.functional as F  # resampling only; no file I/O

        wav = F.resample(wav, model.sample_rate, int(model.sample_rate / speed))

    out = Path(tempfile.mkdtemp()) / "drucker.wav"
    # write with soundfile: newer torchaudio delegates file I/O to torchcodec, which the ZeroGPU
    # image does not ship (and we do not need it for anything else here)
    import soundfile as sf

    sf.write(str(out), wav.squeeze(0).float().cpu().numpy(), model.sample_rate)
    return str(out)


with gr.Blocks(title="Drucker TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Drucker TTS\n"
        "Speak in Peter Drucker's voice. The clone is zero-shot from a clip of his 1981 "
        "reel-to-reel seminars, with a CosyVoice 2 model fine-tuned on 12 hours of "
        "speaker-verified Drucker speech.\n\n"
        "*Prototype: the source recordings are The Drucker Institute material, used here for "
        "internal research only.*"
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="What should Drucker say?", lines=4,
                              value=EXAMPLES[1], max_lines=8)
            voice = gr.Dropdown(choices=list(VOICES), value=DEFAULT_VOICE, label="Reference voice",
                                info="All clips are 1981 reel-to-reel unless noted.")
            speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Speed")
            button = gr.Button("Synthesize", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Drucker", type="filepath", autoplay=True)
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text])

    button.click(synthesize, inputs=[text, voice, speed], outputs=audio, api_name="synthesize")
    text.submit(synthesize, inputs=[text, voice, speed], outputs=audio)

demo.launch(mcp_server=True)
