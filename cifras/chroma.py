"""Croma NNLS, grave e afinação."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.optimize import nnls

from constants import (
    A4_HZ,
    A4_MIDI,
    BASS_BAND_HIGH,
    BASS_BAND_LOW,
    BASS_SPAN_MIN_S,
    BASS_WINDOW_END,
    BASS_WINDOW_START,
    EPS,
    FUNDAMENTAL_MAX_HZ,
    FUNDAMENTAL_MIN_AMP,
    FUNDAMENTAL_MIN_HZ,
    FUNDAMENTAL_MIN_SAMPLES,
    FUNDAMENTAL_PEAK,
    FUNDAMENTAL_WEAK,
    HARMONICS,
    HARMONY_MIDI_MIN,
    HOP,
    NOTE_HIGH,
    NOTE_LOW,
    OCTAVE_LAG_RATIO,
    OCTAVE_STRENGTH_RATIO,
    SAMPLE_RATE,
    SEMITONE_FREQ_FLOOR,
    SPEC_HIGH,
    SPEC_LOW,
    TUNING_CENTS_CLIP,
    TUNING_CONFIDENCE,
    TUNING_MIN_SAMPLES,
)


def bass_bandpass(samples: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, [BASS_BAND_LOW, BASS_BAND_HIGH], btype="band", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, samples)


def harmonic_dictionary() -> tuple[np.ndarray, np.ndarray]:
    """Cada coluna é uma nota e os harmônicos dela, em semitons afinados."""
    notes = np.arange(NOTE_LOW, NOTE_HIGH + 1)
    rows = SPEC_HIGH - SPEC_LOW + 1
    dictionary = np.zeros((rows, notes.size))
    for index, midi in enumerate(notes):
        for offset, weight in HARMONICS:
            row = int(midi) + offset - SPEC_LOW
            if 0 <= row < rows:
                dictionary[row, index] += weight
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True) + EPS
    return dictionary, notes


def semitone_spectrogram(freqs: np.ndarray, magnitude_spectrum: np.ndarray, reference: float) -> np.ndarray:
    rows = SPEC_HIGH - SPEC_LOW + 1
    spec = np.zeros((rows, magnitude_spectrum.shape[1]))
    valid = np.flatnonzero(freqs >= SEMITONE_FREQ_FLOOR)
    midi = np.round(A4_MIDI + 12.0 * np.log2(freqs[valid] / reference)).astype(int)
    offsets = midi - SPEC_LOW
    for row in range(rows):
        bins = valid[offsets == row]
        if bins.size:
            spec[row] = np.sum(magnitude_spectrum[bins], axis=0)
    return spec


def chroma_at_beats(
    freqs: np.ndarray,
    magnitude_spectrum: np.ndarray,
    beat_times: np.ndarray,
    reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NNLS em duas passagens. O grave explica os harmônicos e sai do espectro.

    O que sobra, da região média para cima, é o croma do acorde.
    """
    spec = semitone_spectrogram(freqs, magnitude_spectrum, reference)
    dictionary, notes = harmonic_dictionary()
    frames = np.clip(np.round(beat_times * SAMPLE_RATE / HOP).astype(int), 0, spec.shape[1] - 1)
    chroma = np.zeros((12, beat_times.size))
    energy = np.zeros(beat_times.size)
    bass_dict = dictionary[:, notes < HARMONY_MIDI_MIN]
    harmony_dict = dictionary[:, notes >= HARMONY_MIDI_MIN]
    harmony_notes = notes[notes >= HARMONY_MIDI_MIN]
    for beat, frame in enumerate(frames):
        stop = int(frames[beat + 1]) if beat + 1 < frames.size else frame + 1
        stop = min(spec.shape[1], max(int(frame) + 1, stop))
        column = np.mean(spec[:, int(frame) : stop], axis=1)
        energy[beat] = float(np.sum(column))
        peak = float(np.max(column)) if column.size else 0.0
        if peak < EPS:
            continue
        target = column / peak
        bass_coeff, _ = nnls(bass_dict, target)
        residual = np.maximum(target - bass_dict @ bass_coeff, 0.0)
        if float(np.max(residual)) < EPS:
            continue
        harmony_coeff, _ = nnls(harmony_dict, residual)
        for pitch in range(12):
            chroma[pitch, beat] = float(np.sum(harmony_coeff[harmony_notes % 12 == pitch]))
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + EPS
    return chroma / norms, energy


def tuning_from_bass(frequencies: np.ndarray, confidences: np.ndarray) -> tuple[float, float]:
    deviations = []
    for freq, confidence in zip(frequencies, confidences):
        if confidence < TUNING_CONFIDENCE or freq <= 0:
            continue
        semitone = 12 * np.log2(freq / A4_HZ)
        deviations.append((semitone - round(semitone)) * 100)
    cents = float(np.median(deviations)) if len(deviations) >= TUNING_MIN_SAMPLES else 0.0
    cents = float(np.clip(cents, -TUNING_CENTS_CLIP, TUNING_CENTS_CLIP))
    return A4_HZ * (2 ** (cents / 1200.0)), round(cents, 1)


def pitch_class(freq: float, reference: float) -> int:
    return int(np.round(12 * np.log2(freq / reference)) + 9) % 12


def estimate_bass(samples: np.ndarray, beat_times: np.ndarray, names: list[str]) -> list[dict]:
    grave = bass_bandpass(samples)
    interval = float(np.median(np.diff(beat_times))) if beat_times.size > 1 else 0.5
    labels = []
    confidences = []
    pitches = []
    for index, start in enumerate(beat_times):
        end = float(beat_times[index + 1]) if index + 1 < beat_times.size else float(start) + interval
        span = max(BASS_SPAN_MIN_S, end - float(start))
        left = int((float(start) + BASS_WINDOW_START * span) * SAMPLE_RATE)
        right = int((float(start) + BASS_WINDOW_END * span) * SAMPLE_RATE)
        left = min(max(0, left), grave.size - 1)
        right = min(grave.size, max(left + 8, right))
        freq, confidence = fundamental(grave[left:right])
        confidences.append(confidence)
        pitches.append(freq)
        labels.append(note_name(freq, names) if freq > 0 else "?")
    return [
        {"beat": index, "label": label, "confidence": round(confidences[index], 3), "hz": round(pitches[index], 1)}
        for index, label in enumerate(labels)
    ]


def fundamental(chunk: np.ndarray) -> tuple[float, float]:
    if chunk.size < FUNDAMENTAL_MIN_SAMPLES or float(np.max(np.abs(chunk))) < FUNDAMENTAL_MIN_AMP:
        return 0.0, 0.0
    windowed = chunk * np.hanning(chunk.size)
    size = 1 << (windowed.size - 1).bit_length()
    spectrum = np.fft.rfft(windowed, size)
    acf = np.fft.irfft(np.abs(spectrum) ** 2, size)[: chunk.size // 2]
    if acf.size < 8 or acf[0] <= 1e-10:
        return 0.0, 0.0
    acf = acf / acf[0]
    min_lag = max(1, int(SAMPLE_RATE / FUNDAMENTAL_MAX_HZ))
    max_lag = min(acf.size - 2, int(SAMPLE_RATE / FUNDAMENTAL_MIN_HZ))
    if max_lag <= min_lag + 2:
        return 0.0, 0.0
    segment = acf[min_lag:max_lag]
    peaks = [
        min_lag + index
        for index in range(1, segment.size - 1)
        if segment[index] >= segment[index - 1]
        and segment[index] > segment[index + 1]
        and segment[index] >= FUNDAMENTAL_PEAK
    ]
    if not peaks:
        lag = min_lag + int(np.argmax(segment))
        if acf[lag] < FUNDAMENTAL_WEAK:
            return 0.0, float(acf[lag])
    else:
        strongest = max(peaks, key=lambda lag: acf[lag])
        octaves = [
            lag for lag in peaks if lag < strongest * OCTAVE_LAG_RATIO and acf[lag] >= acf[strongest] * OCTAVE_STRENGTH_RATIO
        ]
        lag = min(octaves) if octaves else strongest
    if 1 <= lag < acf.size - 1:
        alpha, beta, gamma = float(acf[lag - 1]), float(acf[lag]), float(acf[lag + 1])
        denom = alpha - 2 * beta + gamma
        if abs(denom) > EPS:
            lag = lag + 0.5 * (alpha - gamma) / denom
    return float(SAMPLE_RATE / lag), float(acf[int(round(lag))])


def note_name(freq: float, names: list[str]) -> str:
    semitone = int(np.round(12 * np.log2(freq / A4_HZ)))
    return names[(semitone + 9) % 12]
