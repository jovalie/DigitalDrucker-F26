# Fine-tuning dataset — Drucker-only speech (12 h)

Contents: `drucker_sft_12h.tar` (Git LFS) — the CosyVoice SFT dataset built by
`../tts/finetune/build_sft_data.py` from the speaker-verified, masked Drucker audio.

```
drucker_sft_12h.tar
└── drucker/
    ├── wav/                 5,712 clips, 16 kHz mono PCM16, 3–30 s, segment-aligned
    ├── wav.scp              <utt_id> <abs path>   (contains the *absolute* paths of the build machine)
    ├── text                 <utt_id> <verbatim transcript, fillers kept>
    ├── utt2spk              all utts → speaker `drucker`
    ├── spk2utt
    └── utt_index.json       per-clip provenance: pointer, start/end in the source recording,
                             duration, source SNR, split (train/dev), text
```

## Unpack and use

```bash
git lfs install && git lfs pull            # after cloning
tar -xf drucker_sft_12h.tar -C /scratch/$USER/
D=/scratch/$USER/drucker
# wav.scp holds absolute paths from the machine that built the archive — rewrite them:
awk -v d="$D" '{print $1, d"/wav/"$1".wav"}' $D/wav.scp > $D/wav.scp.new && mv $D/wav.scp.new $D/wav.scp
```

Then follow `../tts/finetune/run_train.sh` (CosyVoice2-0.5B LLM-only SFT). The dataset expects the
CosyVoice tools' layout (`wav.scp`, `text`, `utt2spk`, `spk2utt`, then `extract_embedding.py`,
`extract_speech_token.py`, `make_parquet_list.py`). `utt_index.json` lets you rebuild any subset —
e.g. keep only `snr_db >= 30`, or hold out different recordings for the dev split (the shipped split
holds out `4941` and `4997`, ≈0.35 h).

## Provenance and limits

- Every clip sits inside a region attributed to Drucker by the speaker-ID pass, and is cut from the
  **masked** audio, so no other speaker can leak into training (`../docs/SPEAKER_ID.md`).
- 42 of the 80 recordings were used: `voice_clarity.csv` SNR ≥ 20 dB, capped at 20 min per recording.
  The remaining recordings are the noisier 1974–78 cassettes (SNR 17–19) — 46.5 h of Drucker speech exist
  in total if a broader run is wanted.
- The stock CosyVoice2-0.5B model overfits this single-speaker set after ~5 epochs (CV loss turns up from
  epoch 1); epoch 5 was the best by speaker similarity (0.734 vs 0.697) and WER (0.027 vs 0.009).
- Rights: derived from The Drucker Institute archival audio — internal project use only.
