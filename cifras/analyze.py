"""Lê um MP3 e escreve pulsação, fórmula de compasso, cifras e seções.

A pulsação e o compasso saem da grade de tempos medida no áudio.
O croma de cada tempo é NNLS: o grave explica os próprios harmônicos e sai
do espectro; o acorde sai do que sobra na região média. Se essas notas cabem
no grave, o grave é a fundamental. Se não cabem, o grave entra com barra.
Cada compasso fica com o acorde que os tempos dele sustentam. No tempo 3 a
mesma fundamental pode ganhar a sétima, quando essa nota aparece só nessa
metade. O sus4 só substitui a tríade quando ganha folgado. Se o grave coroa
o dominante, o perfil de presença devolve a tônica uma quinta abaixo. O verso começa quando a voz entra. O refrão começa quando essa
voz empurra e o empurro se sustenta num acorde novo. Um trecho cujo
áudio repete outro herda, no miolo, a cifra da volta que esse áudio confirma.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from audio import decode_audio, find_default_audio, load_spectrum, magnitude
from btc_align import (
    BTC_INTERVALS,
    BTC_QUALITIES,
    CHORD_ONSET_SLACK,
    SLASH_BASS_CONFIDENCE,
    SLASH_BASS_SHARE,
    btc_bars,
    btc_parts,
    btc_root_support,
    interval_bass,
    key_from_btc,
    model_change_time,
    spell_btc,
    timeline_vote,
)
from chroma import (
    chroma_at_beats,
    estimate_bass,
    fundamental,
    harmonic_dictionary,
    note_name,
    pitch_class,
    semitone_spectrogram,
    tuning_from_bass,
)
from chords import (
    bar_presence,
    bars_from,
    bass_spells,
    chord_root_name,
    chord_templates,
    demote_sus4,
    estimate_key,
    fold_bar,
    half_bar_seventh,
    kind_index,
    label_chords,
    major_triad_root,
    profile_scores,
    relative_minor_seventh,
    rootless_sus4,
    scale_steps,
)
from constants import (
    CHORD_KINDS,
    HARMONICS,
    HARMONY_MIDI_MIN,
    HOP,
    MAJOR_PROFILE,
    MAX_AUDIO_SECONDS,
    METER_CANDIDATES,
    MINOR_PROFILE,
    N_FFT,
    NOTE_HIGH,
    NOTE_LOW,
    PITCHES_FLAT,
    PITCHES_SHARP,
    ROOT,
    SAMPLE_RATE,
    SPEC_HIGH,
    SPEC_LOW,
    TEMPERLEY_MAJOR,
    TEMPERLEY_MINOR,
)
from form import (
    band_envelope,
    bar_level,
    chest_pushes,
    chorus_from_push,
    clock,
    estimate_sections,
    fallback_chorus,
    final_rest,
    first_sustained,
    instrumental_floor,
    push_lands,
    render,
    singing_threshold,
    vocal_profile,
    voice_reentries,
)
from repeats import chroma_by_bar
from pulse import (
    bar_line_offset,
    bar_starts,
    bass_energy,
    beat_track,
    detect_pulse,
    estimate_meter,
    frame_peak,
    harmony_downbeat,
    neural_meter,
    normalize,
    onset_envelope,
    prefer_musical_pulse,
    signature_for,
    subdivision_is_compound,
    tempo_from_onset,
    track_pulse,
)


def detect_harmony(samples, freqs, spectrum, pulse):
    bass = estimate_bass(samples, pulse["beat_times"], PITCHES_FLAT)
    bass_hz = np.array([item["hz"] for item in bass])
    bass_conf = np.array([item["confidence"] for item in bass])
    reference, cents = tuning_from_bass(bass_hz, bass_conf)
    bass_pc = np.array([pitch_class(freq, reference) if freq > 0 else -1 for freq in bass_hz])
    beat_chroma, energy = chroma_at_beats(freqs, spectrum, pulse["beat_times"], reference)
    if pulse.get("source") == "beat_this":
        meter = neural_meter(pulse)
    else:
        meter = estimate_meter(samples, pulse["beat_times"], beat_chroma)
        accent_phase = meter["phase"]
        meter["phase"] = harmony_downbeat(beat_chroma, energy, meter["beats_per_bar"], accent_phase)
        meter["phase_from_harmony"] = meter["phase"] != accent_phase
    downbeats = meter.get("downbeat_indexes")
    key = estimate_key(
        beat_chroma,
        bass_pc,
        bass_conf,
        energy,
        meter["beats_per_bar"],
        meter["phase"],
        downbeats,
    )
    key["reference"] = reference
    key["cents"] = cents
    return beat_chroma, energy, bass_pc, bass_conf, meter, key, downbeats


def label_local_bars(
    pulse, beat_chroma, energy, bass_pc, bass_conf, key, meter, downbeats, samples
):
    chords = label_chords(
        beat_chroma,
        energy,
        bass_pc,
        bass_conf,
        key,
        meter["beats_per_bar"],
        meter["phase"],
        downbeats,
    )
    offset = bar_line_offset(samples, pulse["beat_times"])
    shown = np.maximum(pulse["beat_times"] + offset, 0.0)
    pulse["compasso_ms"] = int(round(offset * 1000))
    bars = bars_from(
        shown,
        chords,
        meter["beats_per_bar"],
        meter["phase"],
        downbeats,
    )
    return bars, chords, shown


def overlay_btc(samples, bars, shown, bass_pc, bass_conf, beat_chroma, chords, key, cents):
    try:
        from btc_detector import recognize

        timeline = recognize(samples.astype(np.float32), tuning_cents=cents)
        if timeline:
            key = key_from_btc(timeline, key)
            bars = btc_bars(
                bars,
                timeline,
                shown,
                bass_pc,
                bass_conf,
                key["names"],
                beat_chroma,
                chords,
            )
            return key, bars, "btc"
    except Exception as error:
        sys.stderr.write(f"BTC indisponível; usando NNLS local: {error}\n")
    return key, bars, "nnls"


def map_form(path, samples, bars, key, bar_chroma=None):
    section_detector = "teoria-musical+ciclos"
    try:
        from structure_detector import cached_ai_labels

        ai_labels, vocal_onset_bar = cached_ai_labels(path, bars)
        sections, voice, patterns = estimate_sections(
            samples,
            bars,
            key=key,
            ai_bar_labels=ai_labels,
            vocal_onset_bar=vocal_onset_bar,
            bar_chroma=bar_chroma,
        )
        if ai_labels:
            section_detector = "teoria-musical+ciclos+all-in-one"
        return sections, voice, patterns, section_detector
    except Exception as error:
        sys.stderr.write(f"Estrutura combinada indisponível; usando ciclos locais: {error}\n")
        sections, voice, patterns = estimate_sections(samples, bars, key=key, bar_chroma=bar_chroma)
        return sections, voice, patterns, section_detector


def chart_document(
    path,
    duration,
    pulse,
    meter,
    key,
    bars,
    detector,
    sections,
    voice,
    patterns,
    section_detector,
    stages,
    total,
):
    ruled = sum(1 for bar in bars if bar.get("origem") == "regra")
    return {
        "arquivo": path.name,
        "duracao_s": round(duration, 3),
        "tom": {
            "nota": key["name"],
            "modo": key["mode"],
            "afinidade": key["score"],
            "afinacao_hz": round(key["reference"], 2),
            "cents": key["cents"],
            "modulacao": key.get("modulacao"),
            "correcao_tom": key.get("correcao_tom"),
        },
        "pulsacao": {
            "bpm": pulse["bpm"],
            "tempos_s": [round(float(value), 3) for value in pulse["beat_times"]],
            "tempos_fortes_s": [
                round(float(pulse["beat_times"][index]), 3)
                for index in meter.get("downbeat_indexes", [])
                if 0 <= int(index) < pulse["beat_times"].size
            ],
            "deslocamento_barras_ms": pulse["compasso_ms"],
            "detector": pulse.get("source", "local"),
            "grade_recolhida_da_colcheia": pulse["collapsed_eighths"],
            "tempos_estendidos": int(pulse.get("extrapolated_beats", 0)),
            "tempos_fortes_detectados": pulse.get("detected_downbeats"),
            "tempos_fortes_inferidos": pulse.get("inferred_downbeats"),
            "tempos_fortes_ignorados": pulse.get("rejected_downbeats"),
            "compassos_irregulares": pulse.get("irregular_bars"),
        },
        "compasso": {
            "formula": f"{meter['numerator']}/{meter['denominator']}",
            "formula_confirmada": meter["formula_confirmed"],
            "numerador": meter["numerator"],
            "denominador": meter["denominator"],
            "tempos_por_compasso": meter["beats_per_bar"],
            "subdivisao_ternaria": meter["compound_subdivision"],
            "margem": meter["margin"],
        },
        "compassos": bars,
        "compassos_ajustados_por_regra": ruled,
        "detector_acordes": detector,
        "detector_estrutura": section_detector,
        "padroes_harmonicos": patterns,
        "voz": voice,
        "secoes": sections,
        "desempenho": {
            "etapas_s": stages,
            "total_s": round(total, 2),
            "fator_tempo_real": round(total / max(duration, 1e-6), 3),
            "dispositivo": "cpu",
            "processador": platform.processor() or platform.machine(),
            "nucleos_logicos": os.cpu_count(),
        },
        "cifra": render(path.stem, pulse, meter, key, sections, voice, patterns, ruled),
    }


def analyze(path: Path) -> dict:
    stages: dict[str, float] = {}
    clock_start = time.perf_counter()
    mark = clock_start

    def lap(name: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        stages[name] = round(now - mark, 2)
        mark = now

    samples, freqs, spectrum, duration = load_spectrum(path)
    lap("decodificacao_e_stft")
    pulse = detect_pulse(samples, spectrum)
    lap("pulsacao")
    beat_chroma, energy, bass_pc, bass_conf, meter, key, downbeats = detect_harmony(
        samples, freqs, spectrum, pulse
    )
    bars, chords, shown = label_local_bars(
        pulse, beat_chroma, energy, bass_pc, bass_conf, key, meter, downbeats, samples
    )
    lap("baixo_croma_e_tom")
    key, bars, detector = overlay_btc(
        samples, bars, shown, bass_pc, bass_conf, beat_chroma, chords, key, key["cents"]
    )
    key["detector"] = detector
    lap("acordes")
    bar_chroma = chroma_by_bar(
        beat_chroma,
        meter["beats_per_bar"],
        meter["phase"],
        downbeats,
    )
    sections, voice, patterns, section_detector = map_form(path, samples, bars, key, bar_chroma)
    lap("estrutura")
    total = time.perf_counter() - clock_start
    return chart_document(
        path,
        duration,
        pulse,
        meter,
        key,
        bars,
        detector,
        sections,
        voice,
        patterns,
        section_detector,
        stages,
        total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mapeia pulsação, compasso, cifras e seções de um áudio."
    )
    parser.add_argument("audio", nargs="?", type=Path)
    args = parser.parse_args()
    audio = (args.audio if args.audio else find_default_audio()).expanduser().resolve()
    if not audio.is_file():
        raise SystemExit(f"Não encontrei {audio}")
    result = analyze(audio)
    out_dir = ROOT / "cifras" / "saida"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "cifra"}
    (out_dir / f"{audio.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{audio.stem}.txt").write_text(result["cifra"], encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(result["cifra"])
    sys.stdout.write(f"\nSalvo em {out_dir}\n")


if __name__ == "__main__":
    main()
