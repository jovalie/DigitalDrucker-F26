# ComfyUI on Laguna

Setup that was validated on `qclgpu` (4× V100-32GB). The two workflows here are:

| Workflow | What it does | Model set |
|---|---|---|
| `LTX 2.3 ID LoRA (int8, …)` | LTX-2.3 22B audio+video generation with the identity LoRA (3 variants, see below) | `--only ltx` (~46 GB) |
| `MiniMax H3 i2v (int8 TE)` | MiniMax H3 image-to-video with the int8 text encoder | `--only minimax` (~41 GB) |

## 1. Install

Validated combination — **the torch build matters on V100** (sm_70 must be in the wheel's arch list):

| Component | Version |
|---|---|
| ComfyUI | 0.37.0 (`comfyui_version.py`) |
| Python | 3.10 |
| PyTorch | **2.7.1+cu126** (`torch.cuda.get_arch_list()` includes `sm_70`) |
| comfy-kitchen | 0.2.35 — provides the native `int8_linear` / `convrot_w4a4` ops |
| comfy-aimdo | 0.5.5 |

```bash
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
python -m venv venv && . venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**No custom nodes are required** for these workflows. (`ComfyUI-GGUF` is only needed if you switch to the
GGUF model variants in the MiniMax notes.)

## 2. Fetch the models

`../models.manifest.json` lists every file with its Hugging Face repo, size and sha256. The fetcher is
resumable and verifies checksums:

```bash
python3 scripts/fetch-models.py --list                                  # what/when/how big
python3 scripts/fetch-models.py --models-dir <base>/models --only ltx   # LTX workflow set (~46 GB)
python3 scripts/fetch-models.py --models-dir <base>/models --only minimax
python3 scripts/fetch-models.py --models-dir <base>/models --verify     # integrity check
```

The individual `download-*.sh` scripts are kept for reference; the manifest fetcher supersedes them.

## 3. Run

```bash
python main.py --base-directory <base> --listen 0.0.0.0 --port 8188
```

`--base-directory` is what makes `models/`, `user/`, `output/` live together; copy
`workflows/*.json` into `<base>/user/default/workflows/` and they appear in the Workflows sidebar.

The HTTP API is always enabled on the same port (`/prompt`, `/history`, `/view`, `/object_info`,
`/system_stats`, and ComfyUI's own `/internal/logs/raw` which is handy for remote debugging).

## 4. Driving it over the API

`scripts/api-example.py` queues a workflow, waits, and downloads the resulting video:

```bash
python3 scripts/api-example.py --server http://laguna-host:8188 \
    --workflow my_api_workflow.json --set "340.inputs.prompt=Peter Drucker explains management" \
    --download ./out
```

**Important:** `/prompt` takes the **API format**, not the UI format. The workflows in `workflows/` are UI
format (they contain subgraph definitions). To get an API-format copy, open the workflow in the ComfyUI
web UI and use **Workflow → Export (API)**, then keep that JSON next to the workflow. `api-example.py`
also accepts a UI workflow and will warn you if it looks like one.

## 5. V100 memory notes (why three LTX variants)

The LTX-2.3 transformer is 22B parameters. The stock template loads the **fp8 checkpoint (30.7 GB
resident)**, which does not fit a 32 GB V100: ComfyUI offloads 17.8 GB per step (≈490 s/it) and then OOMs
when the second (×2 upscale) sampling pass reloads the model. The workflows here instead use Kijai's
**int8/convrot transformer (20 GB, native int8 on Volta → also faster than the emulated fp8 path)** plus
a standalone video VAE:

| Variant | Gemma encoder | Base latent | Use when |
|---|---|---|---|
| `LTX 2.3 ID LoRA (int8, V100)` | GPU | 768×512×97 | ≥ 40 GB VRAM, or you accept model offload |
| `LTX 2.3 ID LoRA (int8, V100, TE on CPU)` | **CPU** | 768×512×97 | 32 GB V100 — frees ~11 GB so the transformer is fully resident |
| `LTX 2.3 ID LoRA (int8, TE CPU, 640x384x49)` | **CPU** | 640×384×49 | 32 GB V100 and you want guaranteed headroom for the upscale pass |

Widget values (identical in all variants): `ckpt_name = ltx-2.3-22b-dev-fp8.safetensors` (kept only for
the text-encoder + audio-VAE weights — **do not** point it at the int8 file), `text_encoder =
gemma_3_12B_it_fp4_mixed.safetensors`, `distilled_lora = ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors`,
`id_lora = ltx-2.3-id-lora-talkvid-3k.safetensors`, `upscale_model = ltx-2.3-spatial-upscaler-x2-1.1.safetensors`.

On A100/H100 (80 GB) use the first variant — everything fits with room to spare.

## 6. MiniMax H3 notes

The shipped workflow uses `minimax_h3_fl2va_pruned_int8_convrot` + the nvfp4 text encoder. nvfp4 is
**emulated** on Volta (works, but ~53 s/it for a 20-step generation; the payload's `int8_convrot` text
encoder is native and much faster if you have the VRAM). A GGUF variant of the same pipeline was
benchmarked earlier and is documented in `../voice_cloning/README.md`'s sibling docs on QCL.

## 7. Multi-GPU

ComfyUI runs one model on one device — there is no tensor/model parallelism. Extra GPUs are useful only
as separate instances (`--cuda-device` / `CUDA_VISIBLE_DEVICES`), e.g. one worker per GPU on Laguna, or
with the community MultiGPU nodes to place the text encoder and transformer on different devices.
