"""Seções da All-In-One, alinhadas aos compassos da grade de tempos fortes."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LABELS = {
    "start": "silêncio",
    "end": "silêncio",
    "intro": "introdução",
    "outro": "outro",
    "break": "instrumental",
    "bridge": "ponte",
    "inst": "instrumental",
    "solo": "solo",
    "verse": "verso",
    "chorus": "refrão",
}

_SESSION = None
_SESSION_LOCK = threading.Lock()
CACHE = Path(__file__).resolve().parent / "cache" / "structure"
VOCAL_PEAK_PERCENTILE = 90
VOCAL_FLOOR_PERCENTILE = 25
VOCAL_BLEND = 0.20
VOCAL_HOLD_BARS = 3


def _session():
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            from allin1_infer import AllInOneSession

            _SESSION = AllInOneSession(model="harmonix-fold0", device="cpu")
            _SESSION.load()
        return _SESSION


def _label_at(segments, moment: float) -> str:
    for segment in segments:
        if float(segment.start) <= moment < float(segment.end):
            return LABELS.get(str(segment.label), str(segment.label))
    if segments:
        nearest = min(
            segments,
            key=lambda segment: min(
                abs(moment - float(segment.start)),
                abs(moment - float(segment.end)),
            ),
        )
        return LABELS.get(str(nearest.label), str(nearest.label))
    return "instrumental"


def _vocal_onset(vocal_path: Path, bars: list[dict]) -> int | None:
    """Find the first sustained vocal bar from the separated vocal stem."""
    if not vocal_path.is_file() or len(bars) < 2:
        return None
    import soundfile as sf

    audio, sample_rate = sf.read(str(vocal_path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    energies = []
    for bar in bars:
        start = max(0, int(float(bar["start"]) * sample_rate))
        end = min(mono.size, int(float(bar["end"]) * sample_rate))
        part = mono[start:end]
        if part.size:
            energies.append(float(np.sqrt(np.mean(part * part) + 1e-12)))
        else:
            energies.append(0.0)
    values = np.asarray(energies)
    peak = float(np.percentile(values, VOCAL_PEAK_PERCENTILE))
    floor = float(np.percentile(values[: min(4, values.size)], VOCAL_FLOOR_PERCENTILE))
    threshold = floor + VOCAL_BLEND * max(0.0, peak - floor)
    for index in range(values.size):
        window = values[index : index + VOCAL_HOLD_BARS]
        if values[index] >= threshold and np.sum(window >= threshold) >= min(2, window.size):
            return index
    return None


@contextmanager
def _decoded_input(path: Path):
    if path.suffix.lower() in {".wav", ".flac"}:
        yield path
        return

    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="cifra-estrutura-") as directory:
        fingerprint = hashlib.sha1(
            f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode()
        ).hexdigest()[:10]
        target = Path(directory) / f"{path.stem}-{fingerprint}.wav"
        completed = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(path),
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Falha ao preparar o áudio para o mapa: {detail}")
        yield target


def cached_ai_labels(
    path: Path, bars: list[dict], run_if_missing: bool = False
) -> tuple[list[str] | None, int | None]:
    """Retorna os rótulos por compasso e início vocal do All-In-One se já estiverem em cache.

    Evita bloquear 25 minutos na CPU quando o mapeamento por ciclos harmônicos e
    teoria musical já está ativo, mas aproveita o resultado neural se existir.
    """
    if not bars:
        return None, None
    fingerprint = hashlib.sha1(
        f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:10]
    stem_name = (
        f"{path.stem}-{fingerprint}" if path.suffix.lower() not in {".wav", ".flac"} else path.stem
    )
    result_json = CACHE / "results" / f"{stem_name}.json"
    vocal_path = CACHE / "demix" / "htdemucs" / stem_name / "vocals.wav"

    predicted = None
    if result_json.is_file():
        import json as _json
        from types import SimpleNamespace

        raw = _json.loads(result_json.read_text(encoding="utf-8"))
        predicted = [SimpleNamespace(**seg) for seg in (raw.get("segments") or [])]
    elif run_if_missing:
        with _decoded_input(path) as source:
            CACHE.mkdir(parents=True, exist_ok=True)
            result = _session().infer(
                str(source),
                multiprocess=False,
                keep_byproducts=True,
                demix_dir=CACHE / "demix",
                spec_dir=CACHE / "spectrograms",
                out_dir=CACHE / "results",
            )
            predicted = list(result.segments or [])
            vocal_path = CACHE / "demix" / "htdemucs" / source.stem / "vocals.wav"

    vocal_onset = _vocal_onset(vocal_path, bars) if vocal_path.is_file() else None
    if not predicted:
        return None, vocal_onset
    labels = []
    for index, bar in enumerate(bars):
        midpoint = (float(bar["start"]) + float(bar["end"])) / 2.0
        name = _label_at(predicted, midpoint)
        if vocal_onset is not None and index < vocal_onset and name in {"verso", "refrão"}:
            name = "introdução"
        labels.append(name)
    return labels, vocal_onset


def recognize(path: Path, bars: list[dict]) -> tuple[list[dict], dict]:
    """Return model sections snapped to the existing musical bars."""
    if not bars:
        return [], {"detector": "all-in-one"}

    with _decoded_input(path) as source:
        CACHE.mkdir(parents=True, exist_ok=True)
        result = _session().infer(
            str(source),
            multiprocess=False,
            keep_byproducts=True,
            demix_dir=CACHE / "demix",
            spec_dir=CACHE / "spectrograms",
            out_dir=CACHE / "results",
        )
    predicted = list(result.segments or [])
    if not predicted:
        raise RuntimeError("O modelo de estrutura não retornou segmentos.")

    vocal_path = CACHE / "demix" / "htdemucs" / source.stem / "vocals.wav"
    vocal_onset = _vocal_onset(vocal_path, bars)

    groups: list[dict] = []
    for index, bar in enumerate(bars):
        midpoint = (float(bar["start"]) + float(bar["end"])) / 2.0
        name = _label_at(predicted, midpoint)
        if vocal_onset is not None and index < vocal_onset and name in {"verso", "refrão"}:
            name = "introdução"
        if groups and groups[-1]["name"] == name:
            groups[-1]["end"] = bar["end"]
            groups[-1]["bars"].append(bar)
            continue
        groups.append(
            {
                "name": name,
                "start": bar["start"],
                "end": bar["end"],
                "bars": [bar],
            }
        )

    sections = []
    for group in groups:
        sections.append(
            {
                "name": group["name"],
                "start": group["start"],
                "end": group["end"],
                "repeats": len(group["bars"]),
                "bars": [
                    {
                        "bar": bar["bar"],
                        "chords": bar["chords"],
                        "confidence": bar["confidence"],
                    }
                    for bar in group["bars"]
                ],
            }
        )

    verse = next((section["start"] for section in sections if section["name"] == "verso"), None)
    chorus = next((section["start"] for section in sections if section["name"] == "refrão"), None)
    voice = {
        "entrada_verso_s": None if verse is None else round(float(verse), 3),
        "entrada_refrao_s": None if chorus is None else round(float(chorus), 3),
        "refrao_pela_dinamica": False,
        "detector": "all-in-one/harmonix-fold0",
    }
    return sections, voice
