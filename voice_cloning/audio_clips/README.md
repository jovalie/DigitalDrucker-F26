# Drucker audio clips (5 × 2 minutes)

Five 120 s clips of Drucker speaking, cut from **inside a single Drucker region** as bounded by the
speaker-ID pass — so each clip is 100 % Drucker, with no other speaker anywhere in it.

| Clip | Year | Source recording | Window | SNR | Spectral centroid |
|---|---|---|---|---|---|
| `drucker_4793_1981_433s_120s.wav` | 1981 | symposium, *managing the increasing complexity…* reel I | 7:13–9:13 | 33 dB | 1642 Hz |
| `drucker_4997_1979_61s_120s.wav` | 1979 | symposium III (with Worth Loomis / William Dill) | 1:01–3:01 | 37 dB | 584 Hz |
| `drucker_4943_1987_1047s_120s.wav` | 1987 | symposium day 1, luncheon tape 3 side a | 17:27–19:27 | 42 dB | 429 Hz |
| `drucker_3561_1980_381s_120s.wav` | 1980 | lecture series, cassette 2 side 1 | 6:21–8:21 | 33 dB | 680 Hz |
| `drucker_4938_1977_2771s_120s.wav` | 1977 | *Drucker on managing effectively* | 46:11–48:11 | 32 dB | 389 Hz |

Format: 44.1 kHz mono PCM16, `highpass=f=70`, loudness-normalised to −18 LUFS / −1.5 dBTP
(mean level −20 to −23 dBFS). Cut from the original media, not the 16 kHz analysis copies.

Fidelity ranking is by SNR + spectral centroid + HF ratio (`catalog/voice_clarity.csv`), with a
tie-break on year so the set spans 1977–1987. The 1981 recordings are the wideband reel-to-reel
transfers (centroid ~1400–1800 Hz); the 1977/1987 tapes are narrower (400–680 Hz), which is
audible as a duller timbre.

Regenerate or pick different windows:

```bash
python3 scripts/make_drucker_audio_clips.py --count 5 --seconds 120
python3 scripts/make_drucker_audio_clips.py --pointers 4796,5129 --seconds 90   # any others
```

Rights: derived from The Drucker Institute archival audio — internal project use only.
