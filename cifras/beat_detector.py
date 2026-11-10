"""Grade neural de pulsações e tempos fortes com Beat This!."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHECKPOINT = Path(__file__).resolve().parent / "vendor" / "beat-this" / "final0.ckpt"
FPS = 50
# Custo de um compasso mais curto ou mais longo que o modal. Com ele, um tempo
# forte isolado no tempo 3 não move a fase; dois compassos seguidos movem.
METER_CHANGE = 1e-3
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.98
NEUTRAL_PROBABILITY = 0.5
MIN_BEATS = 8
MIN_DOWNBEATS = 2
HEAD_EXTEND_S = 0.04
BAR_LENGTH_MIN = 2
BAR_LENGTH_MAX = 12
LOGIT_WINDOW = 2

_frames = None
_postprocessor = None


def available() -> bool:
    return CHECKPOINT.is_file()


def frames_model():
    global _frames, _postprocessor
    if _frames is None:
        if not available():
            raise FileNotFoundError(f"Pesos Beat This! ausentes: {CHECKPOINT}")
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor

        _frames = Audio2Frames(checkpoint_path=str(CHECKPOINT), device="cpu")
        _postprocessor = Postprocessor(type="minimal", fps=FPS)
    return _frames, _postprocessor


def downbeat_probability(logits: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    frames = np.round(beat_times * FPS).astype(int)
    probabilities = np.full(beat_times.size, NEUTRAL_PROBABILITY)
    for index, frame in enumerate(frames):
        if 0 <= frame < logits.size:
            window = logits[max(0, frame - LOGIT_WINDOW) : frame + LOGIT_WINDOW + 1]
            probabilities[index] = 1.0 / (1.0 + np.exp(-float(np.max(window))))
    return probabilities


def decode_bar_positions(probabilities: np.ndarray, beats_per_bar: int) -> np.ndarray:
    """Viterbi da posição de cada pulso no compasso; devolve os índices dos tempos 1.

    Avançar uma posição é o normal. Voltar ao tempo 1 antes do fim, ou segurar o
    último tempo, custa ``METER_CHANGE``; nenhum compasso fica com menos de dois
    tempos. A emissão é a probabilidade de tempo
    forte que a rede dá ao pulso.
    """
    size = probabilities.size
    positions = beats_per_bar
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, PROBABILITY_CEILING)
    emission = np.tile(np.log(1.0 - clipped), (positions, 1))
    emission[0] = np.log(clipped)
    stay = np.log(1.0 - METER_CHANGE)
    change = np.log(METER_CHANGE)
    score = emission[:, 0] + np.log(1.0 / positions)
    back = np.zeros((positions, size), dtype=int)
    for beat in range(1, size):
        nxt = np.full(positions, -np.inf)
        for position in range(positions):
            candidates = []
            if position == 0:
                candidates.append((score[positions - 1] + stay, positions - 1))
                candidates.extend(
                    (score[early] + change, early) for early in range(1, positions - 1)
                )
            else:
                candidates.append((score[position - 1] + stay, position - 1))
                if position == positions - 1:
                    candidates.append((score[position] + change, position))
            value, origin = max(candidates)
            nxt[position] = value + emission[position, beat]
            back[position, beat] = origin
        score = nxt
    state = int(np.argmax(score))
    path = np.zeros(size, dtype=int)
    for beat in range(size - 1, -1, -1):
        path[beat] = state
        state = back[state, beat]
    return np.flatnonzero(path == 0)


def recognize(samples: np.ndarray, sample_rate: int) -> dict:
    import torch

    duration = samples.size / sample_rate
    model, postprocessor = frames_model()
    beat_logits, downbeat_logits = model(samples, sample_rate)
    with torch.no_grad():
        beats, downbeats = postprocessor(beat_logits, downbeat_logits)
    logits = downbeat_logits.cpu().numpy().astype(float)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if beats.size < MIN_BEATS or downbeats.size < MIN_DOWNBEATS:
        raise ValueError("Beat This! não encontrou uma grade suficiente.")

    interval = float(np.median(np.diff(beats)))
    original = beats.copy()
    before = []
    cursor = float(beats[0] - interval)
    while cursor > HEAD_EXTEND_S:
        before.append(cursor)
        cursor -= interval
    if before:
        beats = np.r_[np.asarray(before[::-1]), beats]
    offset = len(before)
    after = 0
    while beats[-1] + interval < duration + interval * 0.5:
        beats = np.r_[beats, beats[-1] + interval]
        after += 1

    matched = np.asarray(
        [int(np.argmin(np.abs(original - value))) + offset for value in downbeats], dtype=int
    )
    matched = np.unique(matched)
    differences = np.diff(matched)
    valid = differences[(differences >= BAR_LENGTH_MIN) & (differences <= BAR_LENGTH_MAX)]
    if valid.size == 0:
        raise ValueError("Beat This! não determinou quantos tempos cabem no compasso.")
    beats_per_bar = int(np.argmax(np.bincount(valid)))

    probabilities = downbeat_probability(logits, beats)
    probabilities[:offset] = NEUTRAL_PROBABILITY
    if after:
        probabilities[-after:] = NEUTRAL_PROBABILITY
    downbeat_indexes = decode_bar_positions(probabilities, beats_per_bar)
    phase = int(np.bincount(downbeat_indexes % beats_per_bar, minlength=beats_per_bar).argmax())
    lengths = np.diff(downbeat_indexes)
    inferred = int(np.sum(probabilities[downbeat_indexes] < 0.5))
    bpm = 60.0 / float(np.median(np.diff(original)))
    return {
        "bpm": round(bpm, 1),
        "beat_times": beats,
        "downbeat_indexes": downbeat_indexes,
        "beats_per_bar": beats_per_bar,
        "phase": phase,
        "collapsed_eighths": False,
        "source": "beat_this",
        "extrapolated_beats": offset + after,
        "detected_downbeats": int(matched.size),
        "inferred_downbeats": inferred,
        "rejected_downbeats": int(np.setdiff1d(matched, downbeat_indexes).size),
        "irregular_bars": int(np.sum(lengths != beats_per_bar)),
    }
