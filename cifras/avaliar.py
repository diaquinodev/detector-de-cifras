"""Compara as cifras salvas com anotações de referência, usando mir_eval.

Para cada ``saida/<faixa>.json`` procura, na pasta de anotações:

- ``<faixa>.lab``: acordes em notação Harte, linhas ``início fim acorde``;
- ``<faixa>.beats``: uma linha por pulso, ``tempo [posição]``; posição 1 é o tempo forte.

Acordes são medidos duas vezes: a cifra final e só o modelo, sem as regras de
ciclo. A diferença entre as duas é o efeito das regras.

    pip install mir_eval
    python cifras/avaliar.py caminho/das/anotacoes

Nenhum número sai sem anotação. Use só anotações cuja licença permita esse uso
e áudio obtido legalmente, e não avalie em músicas usadas para ajustar as regras.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "saida"
ROOTS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = {
    "": "maj",
    "m": "min",
    "7": "7",
    "maj7": "maj7",
    "m7": "min7",
    "dim": "dim",
    "aug": "aug",
    "m6": "min6",
    "6": "maj6",
    "mMaj7": "minmaj7",
    "dim7": "dim7",
    "m7b5": "hdim7",
    "sus2": "sus2",
    "sus4": "sus4",
    "°": "dim",
    "ø": "hdim7",
    "9": "9",
    "maj9": "maj9",
    "m9": "min9",
    "11": "11",
    "m11": "min11",
    "13": "13",
    "5": "5",
    "7sus4": "sus4(b7)",
    "add9": "maj(9)",
    "madd9": "min(9)",
}
DEGREES = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
CHORD_METRICS = ("root", "majmin", "sevenths", "tetrads", "mirex")
TEMPO_TOLERANCE = 0.04


def to_harte(symbol: str) -> str:
    text = (symbol or "N").strip()
    if text in {"N", "N.C.", ""}:
        return "N"
    if text == "X":
        return "X"
    head, _, bass = text.partition("/")
    root_name = head[:2] if len(head) > 1 and head[1] in "#b" else head[:1]
    quality = QUALITIES.get(head[len(root_name) :])
    if root_name not in ROOTS or quality is None:
        return "X"
    root = ROOTS[root_name]
    harte = f"{SHARPS[root]}:{quality}"
    if bass in ROOTS and ROOTS[bass] != root:
        harte += "/" + DEGREES[(ROOTS[bass] - root) % 12]
    return harte


def chord_intervals(bars: list[dict], model_only: bool) -> tuple[np.ndarray, list[str]]:
    intervals, labels = [], []
    for bar in bars:
        events = (
            bar.get("events_modelo") if model_only and "events_modelo" in bar else bar.get("events")
        )
        if not events:
            events = [
                {"chord": chord, "start": bar["start"], "end": bar["end"]}
                for chord in bar.get("chords") or ["N"]
            ][:1]
        for event in events:
            if float(event["end"]) > float(event["start"]):
                intervals.append([float(event["start"]), float(event["end"])])
                labels.append(to_harte(str(event["chord"])))
    return np.asarray(intervals), labels


def read_beats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    beats, downbeats = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        moment = float(parts[0])
        beats.append(moment)
        if len(parts) > 1 and parts[1].split(".")[0] == "1":
            downbeats.append(moment)
    return np.asarray(beats), np.asarray(downbeats)


def tempo_accuracy(estimated: float, reference: float) -> tuple[bool, bool]:
    """Acc1: dentro de 4%. Acc2: idem, aceitando os fatores 2, 3, 1/2 e 1/3."""

    def close(value: float) -> bool:
        return abs(value - reference) <= TEMPO_TOLERANCE * reference

    return close(estimated), any(close(estimated * factor) for factor in (1, 2, 3, 1 / 2, 1 / 3))


def evaluate_chords(mir_eval, reference: Path, bars: list[dict]) -> dict:
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(reference))
    result = {}
    for name, model_only in (("final", False), ("so_modelo", True)):
        est_intervals, est_labels = chord_intervals(bars, model_only)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals,
            est_labels,
            ref_intervals.min(),
            ref_intervals.max(),
            mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD,
        )
        kept = est_intervals[:, 1] > est_intervals[:, 0]
        est_intervals, est_labels = (
            est_intervals[kept],
            [label for label, keep in zip(est_labels, kept) if keep],
        )
        merged, ref_merged, est_merged = mir_eval.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        durations = mir_eval.util.intervals_to_durations(merged)
        scores = {}
        for metric in CHORD_METRICS:
            comparison = getattr(mir_eval.chord, metric)(ref_merged, est_merged)
            scores[metric] = round(
                float(mir_eval.chord.weighted_accuracy(comparison, durations)), 4
            )
        scores["sobressegmentacao"] = round(
            float(mir_eval.chord.overseg(ref_intervals, est_intervals)), 4
        )
        scores["subsegmentacao"] = round(
            float(mir_eval.chord.underseg(ref_intervals, est_intervals)), 4
        )
        result[name] = scores
    return result


def evaluate_beats(mir_eval, reference: Path, data: dict) -> dict:
    ref_beats, ref_downbeats = read_beats(reference)
    pulse = data.get("pulsacao") or {}
    est_beats = np.asarray(pulse.get("tempos_s") or [])
    est_downbeats = np.asarray(pulse.get("tempos_fortes_s") or [])
    if est_beats.size and est_downbeats.size:
        nearest = np.abs(est_downbeats[:, None] - est_beats[None, :]).argmin(axis=1)
        est_downbeats = est_beats[nearest]
    ref_beats = mir_eval.beat.trim_beats(ref_beats)
    est_beats = mir_eval.beat.trim_beats(est_beats)
    scores = {
        "f_measure": mir_eval.beat.f_measure(ref_beats, est_beats, f_measure_threshold=0.07),
        "cemgil": mir_eval.beat.cemgil(ref_beats, est_beats, cemgil_sigma=0.04)[0],
        "p_score": mir_eval.beat.p_score(ref_beats, est_beats, p_score_threshold=0.2),
    }
    _cmlc, cmlt, _amlc, amlt = mir_eval.beat.continuity(ref_beats, est_beats)
    scores.update({"cmlt": cmlt, "amlt": amlt})
    if ref_downbeats.size and est_downbeats.size:
        scores["downbeat_f_measure"] = mir_eval.beat.f_measure(
            mir_eval.beat.trim_beats(ref_downbeats), mir_eval.beat.trim_beats(est_downbeats), 0.07
        )
    if ref_beats.size > 2 and pulse.get("bpm"):
        reference_bpm = 60.0 / float(np.median(np.diff(ref_beats)))
        acc1, acc2 = tempo_accuracy(float(pulse["bpm"]), reference_bpm)
        scores.update(
            {"tempo_acc1": acc1, "tempo_acc2": acc2, "bpm_referencia": round(reference_bpm, 2)}
        )
    return {
        key: (round(float(value), 4) if isinstance(value, (float, np.floating)) else value)
        for key, value in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia as cifras salvas contra anotações de referência."
    )
    parser.add_argument("anotacoes", type=Path)
    args = parser.parse_args()
    try:
        import mir_eval
    except ImportError:
        raise SystemExit("Instale o mir_eval primeiro: pip install mir_eval")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    report = {
        "parametros": {
            "beat_f_measure_janela_s": 0.07,
            "cemgil_sigma_s": 0.04,
            "p_score_limiar": 0.2,
            "tempo_tolerancia": TEMPO_TOLERANCE,
            "tempo_fatores_acc2": [2, 3, 0.5, 1 / 3],
            "mir_eval": getattr(mir_eval, "__version__", "?"),
        },
        "faixas": {},
    }
    for path in sorted(OUT.glob("*.json")):
        chords_ref = args.anotacoes / f"{path.stem}.lab"
        beats_ref = args.anotacoes / f"{path.stem}.beats"
        if not chords_ref.is_file() and not beats_ref.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = {}
        if chords_ref.is_file():
            entry["acordes"] = evaluate_chords(mir_eval, chords_ref, data.get("compassos") or [])
        if beats_ref.is_file():
            entry["ritmo"] = evaluate_beats(mir_eval, beats_ref, data)
        report["faixas"][path.stem] = entry
    if not report["faixas"]:
        raise SystemExit(f"Nenhuma anotação em {args.anotacoes} casa com as faixas de {OUT}.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
