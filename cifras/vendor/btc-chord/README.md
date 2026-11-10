---
license: mit
tags:
  - audio
  - music-information-retrieval
  - chord-recognition
  - btc
  - pytorch
---

# BTC — Chord Recognition

> **Unofficial HuggingFace packaging** of [BTC](https://github.com/jayg996/BTC-ISMIR19)
> (Bi-directional Transformer for Chord Recognition, ISMIR 2019). Model weights and
> architecture are unchanged from the official release. This repo adds a `from_pretrained`
> + `predict` interface and **trims the dependencies to a modern, conflict-free minimum**.

Recognizes the **chord progression** of a music track. Outputs a chord timeline — a list of
`{start, end, chord}` segments, where `chord` is [Harte notation](https://ismir2005.ismir.net/proceedings/1080.pdf)
(`A#`, `G:min`, `C:maj7`, `N` = no-chord).

Two vocabularies are bundled:

| `large_voca` | Vocabulary | Checkpoint |
|--------------|-----------|------------|
| `True` *(default)* | 170 chords — maj / min / dim / aug / 6 / 7 / min7 / maj7 / sus, etc. | `btc_model_large_voca.pt` |
| `False` | 25 chords — major / minor only | `btc_model.pt` |

## Quick start

```bash
pip install torch numpy librosa transformers huggingface_hub
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("puar-playground/btc-chord", trust_remote_code=True)
chords = model.predict("song.wav")
# [{'start': 0.0,   'end': 1.481, 'chord': 'N'},
#  {'start': 1.481, 'end': 3.981, 'chord': 'A#'},
#  {'start': 3.981, 'end': 7.037, 'chord': 'D#'}, ...]
```

`predict` accepts a file path or a 1-D `float32` numpy array. Audio is resampled internally
to 22050 Hz mono and turned into a log-CQT (144 bins, 24 per octave). Works on any format
`librosa` can read (wav, mp3, flac, …).

Major/minor vocabulary instead of the large one:

```python
model = AutoModel.from_pretrained("puar-playground/btc-chord", trust_remote_code=True,
                                  large_voca=False)
```

## Device selection

```python
# Auto-detect: CUDA → CPU
model = AutoModel.from_pretrained("puar-playground/btc-chord", trust_remote_code=True)

# Explicit
model = AutoModel.from_pretrained("puar-playground/btc-chord", trust_remote_code=True,
                                  device="mps")   # or "cpu", "cuda", "cuda:1"
```

### Apple Silicon (MPS)

The Metal (MPS) backend works — pass `device="mps"` — but for **this** model it is
**slower than CPU**, so `device="auto"` deliberately picks CPU on Macs. The model is tiny
(hidden 128, 8 layers) and runs the audio chunk-by-chunk, so MPS kernel-dispatch and
CPU↔GPU transfer overhead outweighs the GPU speedup.

Measured on an M4 (218 s song, feature extraction included):

| device | `predict()` | real-time factor |
|--------|-------------|------------------|
| **CPU** *(auto)* | **~0.6 s** | ~370× |
| MPS | ~0.9 s | ~250× |

Both are effectively instant; on Apple Silicon just use the default (CPU).

## Dependencies — deliberately minimal

Inference needs only **`torch`, `numpy`, `librosa`** (plus `transformers` / `huggingface_hub`
for the loader). No pinned versions — compatible with **NumPy 2.x** and **PyTorch 2.x**.

The original repo depended on `mir_eval`, `pretty_midi`, `pandas`, `pyrubberband`, and pinned
old NumPy/PyYAML; those are **removed** here (they were only used for training / evaluation /
MIDI export, not inference). Modern-stack fixes are baked in: `np.float→float`,
`yaml.load(Loader=…)`, `torch.load(weights_only=False)`. This avoids the version conflicts you
hit installing the upstream repo alongside a current audio/ML stack.

## Output format

```python
[
  {"start": 0.0,   "end": 1.481, "chord": "N"},      # seconds; "N" = no chord
  {"start": 1.481, "end": 3.981, "chord": "A#"},     # Harte notation
  {"start": 3.981, "end": 7.037, "chord": "D#:maj7"},
  ...
]
```

## Repository layout

```
btc-chord/
├── modeling_btc.py           # BTCChordConfig + BTCForChordRecognition (trust_remote_code)
├── config.json               # auto_map → AutoModel / AutoConfig
├── requirements.txt
├── btc_model_large_voca.pt   # 170-chord weights
├── btc_model.pt              # maj/min weights
└── btc_src/                  # vendored BTC source (model + CQT features + chord maps)
```

## Credits

- Original repository: [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) (MIT)
- Paper: *A Bi-Directional Transformer for Musical Chord Recognition* (Park et al., ISMIR 2019)
