# Runbook: ComfyUI on Laguna + port forwarding (for labmates)

Everything lives under `/project/JoaZheng_1796/drucker/` on Laguna, which is also the group's
shared folder — so the nodes, models and outputs are the same for everyone.

---

## 0. SSH setup (do this once)

**a. Make a key** 

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_laguna -C "your.name@cgu.edu"
# press Enter for no passphrase, or set one (recommended) and use ssh-agent
```

**b. Register the public key with CARC.** Two things have to be true before anything works:

1. Your CARC account is a **member of project `JoaZheng_1796`** (ask the PI — access is per project,
   not automatic with an account).
2. Your **public key** (`~/.ssh/id_ed25519_laguna.pub`) is added in the CARC account portal
   ("SSH keys" / "Manage keys" section). Copy the whole line:

```bash
cat ~/.ssh/id_ed25519_laguna.pub
```

CARC often needs a few minutes to propagate a new key, and some systems want the key added to
`~/.ssh/authorized_keys` on the login node as a fallback:

```bash
ssh laguna1.carc.usc.edu 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
cat ~/.ssh/id_ed25519_laguna.pub | ssh laguna1.carc.usc.edu 'cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

**c. Add a shortcut** so you don't retype the username every time — put this in `~/.ssh/config`:

```sshconfig
Host laguna
    HostName laguna1.carc.usc.edu
    User <username>@cgu.edu
    IdentityFile ~/.ssh/id_ed25519_laguna
    IdentitiesOnly yes
    ServerAliveInterval 60
    StrictHostKeyChecking accept-new
```

Then everything below gets shorter: `ssh laguna`, and
`ssh -N -L 8199:<NODE>:8188 laguna` for the tunnel.

**d. Check you're in:**

```bash
ssh laguna 'hostname; whoami; squeue -u $USER'
ls /project/JoaZheng_1796/drucker            # should list the project folders
```

Notes: first connect will ask you to accept the host key (type `yes`); a warning about
"post-quantum key exchange" is harmless. From off campus you may need the university VPN.
To leave the project cleanly, remove your key from the portal and delete the `laguna` block from
your config.

---

## 1. Before you start

| You need | Details |
|---|---|
| Laguna access | A USC CARC account that's a member of project `JoaZheng_1796` |
| SSH key | Your own key registered with CARC (an example below uses `~/.ssh/id_ed25519_laguna`) |
| Login node | `laguna1.carc.usc.edu` — the *hostname* for the GPU nodes only resolves from inside |
| Nothing else | No local ComfyUI install, no GPU on your laptop |

---

## 1. Start ComfyUI (on the login node)

```bash
ssh laguna1
sbatch /project/JoaZheng_1796/drucker/comfy.sbatch
squeue -u $USER -o "%.10i %.2t %.10M %.12R"     # PD = waiting, R = running
cat /project/JoaZheng_1796/drucker/logs/comfy_node.txt   # e.g. c22-11.carc.usc.edu
```

The batch script starts ComfyUI on **port 8188**, listening only on the compute node, and writes
the node it landed on to `logs/comfy_node.txt`. First start takes ~30–60 s.

**The job has an 8-hour wall limit.** When it ends, ComfyUI stops — resubmit and re-read the node
name. The node can change between jobs, so always check `comfy_node.txt` before opening a tunnel.

---

## 2. Port-forward it to your machine

Keep this terminal open (macOS/Linux):

```bash
ssh -N -L 8199:laguna1.carc.usc.edu:8188 -i ~/.ssh/id_ed25519_laguna JoaZheng@cmc.edu@laguna1.carc.usc.edu
```

- `-N` = forward only, no shell.
- Replace `<NODE>` with the short name from `comfy_node.txt` (e.g. `c22-11`, without `.carc.usc.edu`).
- Local port **8199** is a convention here (8188 is taken by another service on QCL).

Then open **http://localhost:8199** (plain `http`, not `https`).

Quick check that it's really up:

```bash
curl -s http://localhost:8199/system_stats | head -c 120     # shows the L40S + VRAM
```

---

## 3. What's in the install

```
/project/JoaZheng_1796/drucker/comfy/
├── main.py                       ComfyUI itself (launched by comfy.sbatch)
├── models/
│   ├── checkpoints/              ltx-2.3-22b-dev-fp8
│   ├── diffusion_models/         LTX-2.5 int8, MiniMax H3 (fl2va / ref2va), …
│   ├── loras/                    distilled, id-lora talkvid, natural-face v2, realism, Pixar_Toon, skin-hair
│   ├── text_encoders/            gemma-3-12b, gemma4 (LTX-2.5), qwen3vl (MiniMax)
│   ├── vae/                      LTX-2.3 / LTX-2.5 video+audio VAEs, MiniMax VAEs
│   └── latent_upscale_models/    LTX spatial upscalers
├── user/default/workflows/       our saved workflows (T2V/I2V/FLF2V Studio + Identity Lock,
│                                 MiniMax natural-face, realism, Pixar variants)
├── input/                        reference images and audio you select in LoadImage / LoadAudio
└── output/                       finished videos (also reachable through the web UI)
```

Workflows of interest: `LTX-2.5 Studio T2V/I2V/FLF2V + Identity Lock`, `video_ltx2_3_id_lora`
(+ `_pixar06/08/10`), `video_minimax_h3_naturalface[_speech|_realism]`, `drucker_video_ltx2_3_…`.

---

## 4. Everyday use

1. Open the workflow from the **Workflows** panel (reload the browser tab after new files appear).
2. Check the `LoadImage` / `LoadAudio` dropdowns — they list what's in `models`/`input`.
3. Queue, and watch the queue panel. Two-stage LTX runs at 1280×720 / 12 s take ~15–20 min on one
   L40S; MiniMax H3 at 20 steps takes ~20–30 min.
4. Download results from the UI's output viewer, or copy them from `comfy/output/`.

Useful habits:

- **Record the settings** behind anything worth keeping (frame rate, length, LoRA strengths, seed
  via *Extra options → control after generate = fixed*). Most of our wasted time came from
  mismatched settings, not from the models.
- **Match the model's native frame rate** — LTX-2.x expects 25 fps; rendering at 30 produces warped
  motion and lip-sync drift.
- **Keep clips short** (≈5–12 s) and heights multiples of 32 (704, not 720) to avoid odd sizes.

---

## 5. If something breaks

| Symptom | Fix |
|---|---|
| Browser can't connect | Tunnel closed → re-run step 2. Check `curl …/system_stats` first |
| `squeue` shows no job | The 8 h limit expired → resubmit (step 1) and use the new node name |
| `Address already in use` | An old tunnel holds 8199 → close it (`pkill -f "8199:"`) and re-open |
| Workflow opens but a model is missing | The file isn't in `models/` — ask before downloading; check the loader's dropdown |
| Job stuck at "pending (Resources)" | The GPU partition is full — wait; `sinfo -p gpu` shows availability |
| Something else | Check `logs/comfy-<jobid>.out` for the traceback and share it |

---

## 6. House rules

- **Archival Drucker material stays on approved storage.** Don't upload his recordings or stills to
  external/cloud services, and don't use the paid partner/API nodes (Kling, Sync.so, ElevenLabs,
  Hailuo …) for his likeness.
- **Everything generated is AI-assisted** and must be labelled as such wherever it's shared.
- Long jobs belong on the GPU, not the login node — start big work through `sbatch`, never directly
  on the login node.
- Only one of us needs to run ComfyUI at a time; if it's already running, just tunnel to the
  existing node instead of submitting a second job.
