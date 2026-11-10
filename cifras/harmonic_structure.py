"""Mapa estrutural baseado em repetição harmônica, anacruse, modulação e teoria musical.

Descobre o centro tonal diatônico real (evitando que uma abertura no IV, como
Fmaj7 | G/F | Em7 | Am7, seja confundida com Fá maior quando a harmonia é
IV – V – iii – vi em Dó maior), identifica modulações de meio-tom ou tom
inteiro no terço final da música (ex.: Dó maior → Réb/Fá# maior), alinha
compassos de pickup/anacruse e turnarounds, estabiliza variações ruidosas dentro
de padrões repetidos e delimita seções coerentes.
"""

from __future__ import annotations

from constants import (
    DIATONIC_MAJOR,
    DIATONIC_MINOR,
    KEY_FIT_MARGIN,
    LETTER_PCS,
    LETTERS,
    MAJOR_FUNCTIONS,
    MAJOR_TONIC_NAMES,
    MINOR_TONIC_NAMES,
    PC_NAMES_FLAT,
    PC_NAMES_SHARP,
    PITCH_TO_PC,
    UNUSUAL_NAMES,
)
from cycles import (
    _window_repeats_score,
    canonical_chord_choice,
    classify_cadence,
    discover_repeating_cycles,
)
from section_map import map_sections_with_theory
from spelling import (
    bar_signature,
    bar_similarity,
    detect_key_modulation,
    diatonic_fit,
    infer_diatonic_key_from_bars,
    key_fitting_the_chords,
    key_spelling,
    override_bar,
    parse_chord_symbol,
    respell_bars,
    respell_chord_symbol,
    roman_and_function,
    spell_with,
)

__all__ = [
    "DIATONIC_MAJOR",
    "DIATONIC_MINOR",
    "KEY_FIT_MARGIN",
    "LETTER_PCS",
    "LETTERS",
    "MAJOR_FUNCTIONS",
    "MAJOR_TONIC_NAMES",
    "MINOR_TONIC_NAMES",
    "PC_NAMES_FLAT",
    "PC_NAMES_SHARP",
    "PITCH_TO_PC",
    "UNUSUAL_NAMES",
    "_window_repeats_score",
    "bar_signature",
    "bar_similarity",
    "canonical_chord_choice",
    "classify_cadence",
    "detect_key_modulation",
    "diatonic_fit",
    "discover_repeating_cycles",
    "infer_diatonic_key_from_bars",
    "key_fitting_the_chords",
    "key_spelling",
    "map_sections_with_theory",
    "override_bar",
    "parse_chord_symbol",
    "respell_bars",
    "respell_chord_symbol",
    "roman_and_function",
    "spell_with",
]


if __name__ == "__main__":
    import copy
    import json
    import sys
    from pathlib import Path

    from audio import decode_audio
    from constants import ROOT
    from form import render, vocal_profile

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = Path(__file__).resolve().parent / "saida"
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        tom = d["tom"]
        key = {
            "name": tom["nota"],
            "mode": tom["modo"],
            "tonic": PITCH_TO_PC[tom["nota"]],
            "reference": tom.get("afinacao_hz", 440.0),
            "cents": tom.get("cents", 0.0),
            "detector": d.get("detector_acordes", "btc"),
        }
        bars = copy.deepcopy(d["compassos"])
        audio_candidate = ROOT / d.get("arquivo", f"{p.stem}.mp3")
        presence = None
        chest = None
        if audio_candidate.is_file():
            samples = decode_audio(audio_candidate)
            prof = vocal_profile(samples, bars)
            presence = prof["presence"]
            chest = prof["chest"]
        secs, voice, cat = map_sections_with_theory(
            bars,
            key,
            vocal_presence=presence,
            vocal_chest=chest,
        )
        d["tom"]["nota"] = key["name"]
        d["tom"]["modo"] = key["mode"]
        if key.get("modulacao"):
            d["tom"]["modulacao"] = key["modulacao"]
        d["compassos"] = bars
        d["secoes"] = secs
        d["voz"] = voice
        d["padroes_harmonicos"] = cat
        d["detector_estrutura"] = "teoria-musical+ciclos"
        pulsacao = d["pulsacao"]
        pulse_dict = {
            "bpm": pulsacao["bpm"],
            "collapsed_eighths": pulsacao.get("grade_recolhida_da_colcheia", False),
            "extrapolated_beats": pulsacao.get("tempos_estendidos", 0),
            "detected_downbeats": pulsacao.get("tempos_fortes_detectados") or 0,
            "inferred_downbeats": pulsacao.get("tempos_fortes_inferidos") or 0,
            "rejected_downbeats": pulsacao.get("tempos_fortes_ignorados") or 0,
            "irregular_bars": pulsacao.get("compassos_irregulares") or 0,
        }
        meter_dict = {
            "numerator": d["compasso"]["numerador"],
            "denominator": d["compasso"]["denominador"],
            "beats_per_bar": d["compasso"]["tempos_por_compasso"],
            "source": "beat_this" if pulsacao.get("detector") == "beat_this" else "local",
        }
        ruled = sum(1 for bar in bars if bar.get("origem") == "regra")
        d["compassos_ajustados_por_regra"] = ruled
        cifra_txt = render(p.stem, pulse_dict, meter_dict, key, secs, voice, cat, ruled)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"{p.stem}.txt").write_text(cifra_txt, encoding="utf-8")
        print("===", p.name, "Tom:", key["name"], key["mode"], "Mod:", key.get("modulacao"), "===")
        for c in cat[:4]:
            print("  PADRAO:", c)
        for s in secs:
            print(
                f"  SECAO: {s['name']:16s} {s['start']:6.2f}s–{s['end']:6.2f}s "
                f"({s['repeats']:2d} c.) | {s['padrao']} | {s['graus']} | {s['cadencia']}"
            )
