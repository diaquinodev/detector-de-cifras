"""Pulsação, fórmula de compasso e grade de tempos fortes."""

from __future__ import annotations

import numpy as np
from audio import magnitude
from constants import (
    BAR_OFFSET_LAG_S,
    BAR_OFFSET_LEAD_S,
    BAR_OFFSET_MIN_ERRORS,
    BAR_OFFSET_RADIUS_S,
    BAR_OFFSET_SKIP,
    BAR_OFFSET_SMOOTH_S,
    BASS_ENERGY_WINDOW_S,
    BASS_FIR_HZ,
    BASS_FIR_TAPS,
    BEAT_LAG_NEAR,
    BEAT_LAG_WIDE,
    BEAT_TIGHTNESS,
    COLLAPSE_ACCENT_RATIO,
    COLLAPSE_MIN_BEATS,
    COLLAPSE_TEMPO_BPM,
    COMPOUND_DUPLE_RATIO,
    COMPOUND_WHOLE_RATIO,
    DOUBLE_LAG_BPM,
    DOUBLE_LAG_RATIO,
    EPS,
    HARMONY_PHASE_EXTRA,
    HARMONY_PHASE_RATIO,
    HOP,
    MAX_BEAT_FRAMES,
    METER_BASS_WEIGHT,
    METER_CANDIDATES,
    METER_CHANGE_WEIGHT,
    METER_MIN_HITS,
    MIN_BEATS_FOR_GRID,
    ONSET_LOG_GAIN,
    ONSET_SMOOTH_FRAMES,
    ONSET_WEIGHT_HIGH,
    ONSET_WEIGHT_LOW,
    QUIET_BAR_GATE,
    SAMPLE_RATE,
    TEMPO_MAX_BPM,
    TEMPO_MIN_BPM,
)
from scipy import signal
from scipy.ndimage import uniform_filter1d


def onset_envelope(magnitude_spectrum: np.ndarray) -> np.ndarray:
    logged = np.log1p(magnitude_spectrum * ONSET_LOG_GAIN)
    flux = np.maximum(0.0, np.diff(logged, axis=1, prepend=logged[:, :1]))
    weights = np.linspace(ONSET_WEIGHT_HIGH, ONSET_WEIGHT_LOW, flux.shape[0])[:, None]
    envelope = np.sum(flux * weights, axis=0)
    return uniform_filter1d(envelope, size=ONSET_SMOOTH_FRAMES)


def track_pulse(envelope: np.ndarray) -> dict:
    tempo, period = tempo_from_onset(envelope)
    beat_frames = beat_track(envelope, period)
    if beat_frames.size < MIN_BEATS_FOR_GRID:
        raise SystemExit("A faixa não tem tempos suficientes para uma grade.")
    beat_times = beat_frames * HOP / SAMPLE_RATE
    accent = envelope[np.clip(beat_frames, 0, envelope.size - 1)]
    beat_times, tempo, collapsed = prefer_musical_pulse(beat_times, accent, tempo)
    return {"bpm": round(tempo, 1), "beat_times": beat_times, "collapsed_eighths": collapsed}


def tempo_from_onset(envelope: np.ndarray) -> tuple[float, int]:
    centered = envelope - np.mean(envelope)
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    min_lag = max(1, int(60 / TEMPO_MAX_BPM * SAMPLE_RATE / HOP))
    max_lag = min(autocorr.size - 1, int(60 / TEMPO_MIN_BPM * SAMPLE_RATE / HOP))
    window = autocorr[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(window))
    doubled = lag * 2
    if (
        doubled <= max_lag
        and autocorr[doubled] > autocorr[lag] * DOUBLE_LAG_RATIO
        and lag < int(60 / DOUBLE_LAG_BPM * SAMPLE_RATE / HOP)
    ):
        lag = doubled
    bpm = 60.0 * SAMPLE_RATE / HOP / lag
    return bpm, lag


def beat_track(envelope: np.ndarray, period: int) -> np.ndarray:
    size = envelope.size
    score = np.full(size, -1e18)
    previous = np.zeros(size, dtype=int)
    tightness = BEAT_TIGHTNESS
    for index in range(size):
        start = max(0, int(index - period * BEAT_LAG_WIDE))
        stop = max(0, int(index - period * BEAT_LAG_NEAR))
        if stop <= start:
            score[index] = envelope[index]
            previous[index] = index
            continue
        lags = np.arange(start, stop)
        penalty = tightness * (np.log((index - lags) / period) ** 2)
        values = score[lags] - penalty
        choice = int(np.argmax(values))
        previous[index] = int(lags[choice])
        score[index] = envelope[index] + values[choice]
    tail = max(0, size - period * 2)
    cursor = tail + int(np.argmax(score[tail:]))
    frames = []
    while cursor > 0 and len(frames) < MAX_BEAT_FRAMES:
        frames.append(cursor)
        nxt = int(previous[cursor])
        if nxt >= cursor:
            break
        cursor = nxt
    frames.reverse()
    return np.asarray(frames, dtype=int)


def prefer_musical_pulse(
    beat_times: np.ndarray, accent: np.ndarray, tempo: float
) -> tuple[np.ndarray, float, bool]:
    if tempo < COLLAPSE_TEMPO_BPM or beat_times.size < COLLAPSE_MIN_BEATS:
        return beat_times, tempo, False
    even = float(np.mean(accent[0::2]))
    odd = float(np.mean(accent[1::2]))
    if min(even, odd) <= EPS or max(even, odd) < min(even, odd) * COLLAPSE_ACCENT_RATIO:
        return beat_times, tempo, False
    phase = 0 if even >= odd else 1
    kept = beat_times[phase::2]
    if kept.size < MIN_BEATS_FOR_GRID:
        return beat_times, tempo, False
    return kept, tempo / 2.0, True


def normalize(values: np.ndarray) -> np.ndarray:
    span = float(np.max(values) - np.min(values))
    if span < EPS:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def estimate_meter(samples: np.ndarray, beat_times: np.ndarray, beat_chroma: np.ndarray) -> dict:
    change = np.zeros(beat_times.size)
    if beat_chroma.shape[1] > 1:
        change[1:] = np.linalg.norm(np.diff(beat_chroma, axis=1), axis=0)
    low = bass_energy(samples, beat_times)
    accent = METER_BASS_WEIGHT * normalize(low) + METER_CHANGE_WEIGHT * normalize(change)
    best = None
    scores = []
    for period in METER_CANDIDATES:
        phase_scores = []
        for phase in range(period):
            hits = (np.arange(beat_times.size) - phase) % period == 0
            if int(np.sum(hits)) < METER_MIN_HITS:
                continue
            score = float(np.mean(accent[hits]))
            phase_scores.append(score)
            if best is None or score > best["score"]:
                best = {"period": period, "phase": phase, "score": score}
        if phase_scores:
            scores.append(max(phase_scores))
    margin = 0.0
    if len(scores) >= 2:
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1]
    compound = subdivision_is_compound(samples, beat_times)
    numerator, denominator = signature_for(best["period"], compound)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "beats_per_bar": best["period"],
        "phase": best["phase"],
        "compound_subdivision": compound,
        "margin": round(float(margin), 3),
        "phase_from_harmony": False,
        "formula_confirmed": False,
    }


def neural_meter(pulse: dict) -> dict:
    """A rede conta tempos entre tempos fortes; ela não distingue 6/8 de 2/4."""
    beats_per_bar = int(pulse["beats_per_bar"])
    numerator, denominator = signature_for(beats_per_bar, False)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "beats_per_bar": beats_per_bar,
        "phase": int(pulse["phase"]),
        "downbeat_indexes": np.asarray(pulse["downbeat_indexes"], dtype=int),
        "compound_subdivision": False,
        "margin": None,
        "phase_from_harmony": False,
        "formula_confirmed": False,
        "source": "beat_this",
    }


def bar_starts(
    steps: int, beats_per_bar: int, phase: int, downbeats: np.ndarray | None = None
) -> np.ndarray:
    if downbeats is not None:
        starts = np.asarray(downbeats, dtype=int)
        starts = starts[(starts >= 0) & (starts < steps)]
    else:
        starts = np.flatnonzero((np.arange(steps) - phase) % beats_per_bar == 0)
    if starts.size == 0 or int(starts[0]) != 0:
        starts = np.unique(np.r_[0, starts])
    return starts


def harmony_downbeat(
    beat_chroma: np.ndarray, energy: np.ndarray, beats_per_bar: int, accent_phase: int
) -> int:
    """Entre as fases do 4/4, prefere a que segura uma fundamental por compasso.

    O grave continua mandando quando a diferença é pequena. Assim um compasso
    que o baixo completa, como o Em7, não é partido para ganhar dois acordes.
    """
    if beats_per_bar != 4:
        return accent_phase
    from chords import chord_templates

    rolled, _suffixes = chord_templates()
    kinds = rolled.shape[0]
    steps = beat_chroma.shape[1]
    emission = np.zeros((12 * kinds, steps))
    for root in range(12):
        rotated = np.roll(beat_chroma, -root, axis=0)
        emission[root * kinds : (root + 1) * kinds] = rolled @ rotated
    quiet = max(float(np.median(energy)) * QUIET_BAR_GATE, EPS)
    scores = []
    for phase in range(4):
        starts = np.flatnonzero((np.arange(steps) - phase) % 4 == 0)
        same = 0
        for start in starts:
            stop = int(start + 4)
            if stop > steps or float(np.mean(energy[start:stop])) < quiet:
                continue
            mid = start + 2
            head = int(np.argmax(np.mean(emission[:, start:mid], axis=1))) // kinds
            tail = int(np.argmax(np.mean(emission[:, mid:stop], axis=1))) // kinds
            if head == tail:
                same += 1
        scores.append(same)
    chosen = int(np.argmax(scores))
    if (
        scores[chosen] < scores[accent_phase] * HARMONY_PHASE_RATIO
        or scores[chosen] < scores[accent_phase] + HARMONY_PHASE_EXTRA
    ):
        return accent_phase
    return chosen


def signature_for(beats_per_bar: int, compound: bool) -> tuple[int, int]:
    if compound and beats_per_bar == 2:
        return 6, 8
    if compound and beats_per_bar == 3:
        return 9, 8
    if compound and beats_per_bar == 4:
        return 12, 8
    return beats_per_bar, 4


def subdivision_is_compound(samples: np.ndarray, beat_times: np.ndarray) -> bool:
    _, spectrum = magnitude(samples)
    envelope = onset_envelope(spectrum)
    centered = envelope - np.mean(envelope)
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    beat_frames = max(1, int(round(np.median(np.diff(beat_times)) * SAMPLE_RATE / HOP)))
    duple = frame_peak(autocorr, beat_frames / 2)
    triple = frame_peak(autocorr, beat_frames / 3)
    whole = frame_peak(autocorr, beat_frames)
    return triple > duple * COMPOUND_DUPLE_RATIO and triple > whole * COMPOUND_WHOLE_RATIO


def frame_peak(autocorr: np.ndarray, lag: float) -> float:
    center = int(round(lag))
    if center <= 0 or center >= autocorr.size:
        return 0.0
    return float(np.max(autocorr[max(0, center - 1) : min(autocorr.size, center + 2)]))


def bass_energy(samples: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    kernel = signal.firwin(BASS_FIR_TAPS, BASS_FIR_HZ, fs=SAMPLE_RATE)
    bass = signal.lfilter(kernel, [1.0], samples)
    squared = bass**2
    window = int(BASS_ENERGY_WINDOW_S * SAMPLE_RATE)
    energy = uniform_filter1d(squared, size=max(1, window))
    indexes = np.clip((beat_times * SAMPLE_RATE).astype(int), 0, energy.size - 1)
    return energy[indexes]


def bar_line_offset(samples: np.ndarray, beat_times: np.ndarray) -> float:
    """O grave ataca antes do agudo. A troca do compasso vai até esse ataque."""
    if beat_times.size < MIN_BEATS_FOR_GRID:
        return 0.0
    from chroma import bass_bandpass

    low = bass_bandpass(samples)
    energy = uniform_filter1d(low**2, size=max(1, int(BAR_OFFSET_SMOOTH_S * SAMPLE_RATE)))
    radius = int(BAR_OFFSET_RADIUS_S * SAMPLE_RATE)
    errors = []
    for beat in beat_times[BAR_OFFSET_SKIP:-BAR_OFFSET_SKIP]:
        center = int(round(float(beat) * SAMPLE_RATE))
        left = max(1, center - radius)
        right = min(energy.size - 2, center + radius)
        if right - left < MIN_BEATS_FOR_GRID:
            continue
        peak = left + int(np.argmax(energy[left:right]))
        if peak <= left + 2 or peak >= right - 2:
            continue
        errors.append((peak - center) / SAMPLE_RATE)
    if len(errors) < BAR_OFFSET_MIN_ERRORS:
        return 0.0
    return float(np.clip(np.median(errors), BAR_OFFSET_LEAD_S, BAR_OFFSET_LAG_S))


def detect_pulse(samples: np.ndarray, spectrum: np.ndarray) -> dict:
    try:
        from beat_detector import recognize as recognize_beats

        return recognize_beats(samples.astype(np.float32), SAMPLE_RATE)
    except Exception as error:
        import sys

        sys.stderr.write(f"Beat This! indisponível; usando a grade local: {error}\n")
        pulse = track_pulse(onset_envelope(spectrum))
        pulse["source"] = "local"
        return pulse
