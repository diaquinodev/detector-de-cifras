"""Decodifica o áudio e calcula o espectro de magnitude."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

from constants import (
    AUDIO_SUFFIXES,
    HOP,
    MAX_AUDIO_SECONDS,
    N_FFT,
    ROOT,
    SAMPLE_RATE,
    SKIPPED_AUDIO_DIRS,
)


def find_default_audio() -> Path:
    found = [
        path
        for path in ROOT.rglob("*")
        if path.suffix.lower() in AUDIO_SUFFIXES and not any(part in SKIPPED_AUDIO_DIRS for part in path.parts)
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit("Nenhum MP3, WAV ou FLAC fora das pastas desempacotadas.")
    names = "\n".join(str(path) for path in found)
    raise SystemExit(f"Há mais de um áudio. Passe o caminho:\n{names}")


def decode_audio(path: Path) -> np.ndarray:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(MAX_AUDIO_SECONDS + 1),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"Não consegui decodificar {path.name}. {detail}")
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if samples.size < SAMPLE_RATE:
        raise SystemExit(f"{path.name} tem menos de um segundo de áudio.")
    if samples.size > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        raise SystemExit(f"{path.name} passa de {MAX_AUDIO_SECONDS // 60} minutos.")
    return samples


def magnitude(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, spectrum = signal.stft(
        samples,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        boundary=None,
        padded=False,
    )
    return freqs, np.abs(spectrum)


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    samples = decode_audio(path)
    duration = float(samples.size / SAMPLE_RATE)
    freqs, spectrum = magnitude(samples)
    return samples, freqs, spectrum, duration
