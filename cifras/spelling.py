"""Grafia, encaixe tonal, modulação e assinatura de compasso."""

from __future__ import annotations

import numpy as np

from constants import (
    DEGREE_OF_INTERVAL,
    DIATONIC_MAJOR,
    DIATONIC_MINOR,
    INFER_KEY_RATIO,
    KEY_FIT_MARGIN,
    LETTER_PCS,
    LETTERS,
    MAJOR_FUNCTIONS,
    MAJOR_TONIC_NAMES,
    MINOR_TONIC_NAMES,
    MODULATION_MIN_BARS,
    MODULATION_NEW_FIT_FLOOR,
    MODULATION_NEW_FIT_RATIO,
    MODULATION_SEARCH_FRACTION,
    MODULATION_SHIFT_CANDIDATES,
    MODULATION_TAIL_BARS,
    PC_NAMES_FLAT,
    PC_NAMES_SHARP,
    PITCH_TO_PC,
    TONIC_DEGREES,
    UNUSUAL_NAMES,
)


def key_spelling(tonic: int, mode: str) -> list[str]:
    """Nome de cada classe de altura pelo grau que ela ocupa no tom.

    Em Dó maior o bVII é Bb, não A#; em Lá menor o VI elevado é F#, não Gb.
    """
    minor = str(mode).lower().startswith("men")
    tonic_name = (MINOR_TONIC_NAMES if minor else MAJOR_TONIC_NAMES)[tonic % 12]
    fallback = (
        PC_NAMES_FLAT
        if "b" in tonic_name or (minor and tonic_name in {"C", "D", "F", "G"})
        else PC_NAMES_SHARP
    )
    tonic_letter = LETTERS.index(tonic_name[0])
    names = []
    for pc in range(12):
        letter_index = (tonic_letter + DEGREE_OF_INTERVAL[(pc - tonic) % 12]) % 7
        offset = (pc - LETTER_PCS[letter_index] + 6) % 12 - 6
        name = LETTERS[letter_index] + {0: "", 1: "#", -1: "b"}.get(offset, "?")
        names.append(fallback[pc] if "?" in name or name in UNUSUAL_NAMES else name)
    return names


def spell_with(chord: str, names: list[str]) -> str:
    info = parse_chord_symbol(chord)
    if info["root"] is None:
        return info["raw"]
    out = names[info["root"]] + info["suffix"]
    if info["bass_name"] is not None and info["bass"] is not None and info["bass"] != info["root"]:
        out = f"{out}/{names[info['bass']]}"
    return out


def respell_bars(bars: list[dict], tonic: int, mode: str, modulation: dict | None) -> None:
    """Reescreve as cifras com a grafia do tom final, e do tom novo depois da modulação."""
    home = key_spelling(tonic, mode)
    moved = (
        key_spelling(modulation["new_tonic"], modulation.get("new_mode", "maior"))
        if modulation
        else home
    )
    start = modulation["start_bar"] if modulation else len(bars)
    for index, bar in enumerate(bars):
        names = moved if index >= start else home
        for field in ("chords", "chords_modelo"):
            if bar.get(field):
                bar[field] = [spell_with(chord, names) for chord in bar[field]]
        for field in ("events", "events_modelo"):
            for event in bar.get(field) or []:
                event["chord"] = spell_with(str(event["chord"]), names)
        if bar.get("chords"):
            bar["root"] = parse_chord_symbol(bar["chords"][0])["root_name"]


def parse_chord_symbol(chord: str) -> dict:
    """Extrai fundamental, qualidade, família funcional e baixo de uma cifra."""
    text = (chord or "N").strip()
    if text in {"N", "N.C.", "X", "?"} or not text:
        return {
            "raw": "N",
            "root_name": "N",
            "root": None,
            "suffix": "",
            "family": "none",
            "bass_name": None,
            "bass": None,
        }
    head, _, bass_name = text.partition("/")
    root_name = head[:2] if len(head) >= 2 and head[1] in {"#", "b"} else head[:1]
    suffix = head[len(root_name) :]
    root = PITCH_TO_PC.get(root_name)
    bass = PITCH_TO_PC.get(bass_name) if bass_name else root
    if suffix.startswith(("m7b5", "dim")):
        family = "dim"
    elif suffix.startswith("m") and not suffix.startswith("maj"):
        family = "min"
    elif suffix == "7" or suffix.startswith("7"):
        family = "dom"
    elif suffix.startswith("sus"):
        family = "sus"
    else:
        family = "maj"
    return {
        "raw": text,
        "root_name": root_name,
        "root": root,
        "suffix": suffix,
        "family": family,
        "bass_name": bass_name or None,
        "bass": bass,
    }


def respell_chord_symbol(chord: str, prefer_sharp: bool = True) -> str:
    """Converte uma cifra para notação em sustenidos (padrão Chordify) ou bemóis."""
    text = (chord or "N").strip()
    if text in {"N", "N.C.", "X", "?"} or not text:
        return "N"
    info = parse_chord_symbol(text)
    if info["root"] is None:
        return text
    names = PC_NAMES_SHARP if prefer_sharp else PC_NAMES_FLAT
    out = names[info["root"]] + info["suffix"]
    if info["bass"] is not None and info["bass_name"] is not None and info["bass"] != info["root"]:
        out = f"{out}/{names[info['bass']]}"
    return out


def infer_diatonic_key_from_bars(bars: list[dict], fallback_key: dict) -> dict:
    """Verifica pela teoria musical se os acordes dos primeiros compassos pertencem
    a uma tonalidade cujo centro foi deslocado por abertura em IV (ex.: Fmaj7 | G/F | Em7 | Am7
    é IV – V – iii – vi em Dó maior, não Fá maior)."""
    early_bars = [b for b in bars[:32] if (b.get("chords") or ["N"]) != ["N"]]
    if len(early_bars) < 4:
        return fallback_key

    scores: list[tuple[float, int, str]] = []
    for cand_tonic in range(12):
        maj_score = 0.0
        min_score = 0.0
        for bar in early_bars:
            weight = max(0.2, float(bar.get("confidence", 0.7)))
            for ch in bar.get("chords") or []:
                info = parse_chord_symbol(ch)
                if info["root"] is None:
                    continue
                rel = (info["root"] - cand_tonic) % 12
                fam = "maj" if info["family"] == "sus" else info["family"]
                if (rel, fam) in DIATONIC_MAJOR:
                    maj_score += weight * (1.25 if rel in {0, 5, 7, 9} else 1.0)
                elif (rel, fam) in MAJOR_FUNCTIONS:
                    maj_score += weight * 0.45
                if (rel, fam) in DIATONIC_MINOR:
                    min_score += weight * (1.25 if rel in {0, 7, 8, 10} else 1.0)
        scores.append((maj_score, cand_tonic, "maior"))
        scores.append((min_score, cand_tonic, "menor"))

    best_score, best_tonic, best_mode = max(scores, key=lambda item: item[0])
    fb_tonic = int(fallback_key.get("tonic", PITCH_TO_PC.get(fallback_key.get("name", "C"), 0)))
    fb_mode = str(fallback_key.get("mode", "maior"))
    current_score = next(
        (sc for sc, t, m in scores if t == fb_tonic and m == fb_mode),
        0.0,
    )
    if best_tonic == (fb_tonic + 7) % 12 and best_score > current_score * INFER_KEY_RATIO:
        updated = dict(fallback_key)
        names = key_spelling(best_tonic, best_mode)
        updated.update(
            {
                "tonic": best_tonic,
                "name": names[best_tonic],
                "mode": best_mode,
                "names": names,
            }
        )
        return updated
    return fallback_key


def diatonic_fit(bars: list[dict], tonic: int, mode: str) -> float:
    """Fração dos acordes da música que são diatônicos no tom."""
    table = DIATONIC_MINOR if str(mode).lower().startswith("men") else DIATONIC_MAJOR
    total = 0
    inside = 0
    for bar in bars:
        for chord in bar.get("chords") or []:
            info = parse_chord_symbol(chord)
            if info["root"] is None:
                continue
            rel = (info["root"] - tonic) % 12
            family = "maj" if info["family"] == "sus" else info["family"]
            if family == "dom" and (rel, "dom") not in table:
                family = "maj"
            total += 1
            inside += (rel, family) in table
    return inside / total if total else 0.0


def key_fitting_the_chords(bars: list[dict], key: dict) -> dict | None:
    """Troca o tom só quando outro explica os acordes por uma margem larga.

    O relativo fica de fora: os dois partilham os acordes, e maior contra menor
    já é decidido pelo perfil e pelo baixo.
    """
    tonic = int(key.get("tonic", PITCH_TO_PC.get(key.get("name", "C"), 0)))
    mode = str(key.get("mode", "maior"))
    minor = mode.lower().startswith("men")
    relative = ((tonic + 3) % 12, "maior") if minor else ((tonic + 9) % 12, "menor")
    current = diatonic_fit(bars, tonic, mode)
    best_fit, best_tonic, best_mode = max(
        (diatonic_fit(bars, candidate, candidate_mode), candidate, candidate_mode)
        for candidate in range(12)
        for candidate_mode in ("maior", "menor")
        if (candidate, candidate_mode) != relative
    )
    if best_fit - current < KEY_FIT_MARGIN:
        return None
    names = key_spelling(best_tonic, best_mode)
    return {
        "tonic": best_tonic,
        "mode": best_mode,
        "name": names[best_tonic],
        "names": names,
        "correcao_tom": {
            "antes": f"{key_spelling(tonic, mode)[tonic]} {mode}",
            "encaixe_antes": round(current, 3),
            "encaixe_depois": round(best_fit, 3),
        },
    }


def detect_key_modulation(bars: list[dict], tonic: int, mode: str) -> dict | None:
    """Detecta se a música modula (ex.: sobe meio-tom ou 1 tom no terço final,
    como C maior / Fmaj7 → C# maior / F#maj7 no compasso 49 de Leave the Door Open)."""
    n = len(bars)
    if n < MODULATION_MIN_BARS:
        return None
    start_search = int(n * MODULATION_SEARCH_FRACTION)
    minor = str(mode).lower().startswith("men")
    diatonic = (
        DIATONIC_MINOR
        if minor
        else {(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (7, "dom"), (9, "min")}
    )

    best_mod = None
    for split_idx in range(start_search, n - MODULATION_TAIL_BARS):
        tail_bars = [b for b in bars[split_idx:] if (b.get("chords") or ["N"]) != ["N"]]
        if len(tail_bars) < MODULATION_TAIL_BARS:
            continue
        for shift in MODULATION_SHIFT_CANDIDATES:
            new_tonic = (tonic + shift) % 12
            orig_fit = 0.0
            new_fit = 0.0
            for b in tail_bars:
                for ch in b.get("chords") or []:
                    info = parse_chord_symbol(ch)
                    if info["root"] is None:
                        continue
                    fam = "maj" if info["family"] == "sus" else info["family"]
                    rel_old = (info["root"] - tonic) % 12
                    rel_new = (info["root"] - new_tonic) % 12
                    if (rel_old, fam) in diatonic:
                        orig_fit += 1.0
                    elif not minor and (rel_old, fam) in MAJOR_FUNCTIONS:
                        orig_fit += 0.35
                    if (rel_new, fam) in diatonic:
                        new_fit += 1.0
                    elif not minor and (rel_new, fam) in MAJOR_FUNCTIONS:
                        new_fit += 0.45
            if new_fit >= max(MODULATION_NEW_FIT_FLOOR, orig_fit * MODULATION_NEW_FIT_RATIO):
                cand = (new_fit - orig_fit, -split_idx, split_idx, new_tonic, shift)
                if best_mod is None or cand > best_mod:
                    best_mod = cand

    if best_mod is None:
        return None
    _, _, split_idx, new_tonic, shift = best_mod
    while split_idx > start_search:
        prev_chords = bars[split_idx - 1].get("chords") or ["N"]
        info = parse_chord_symbol(prev_chords[0])
        if info["root"] is not None and (info["root"] - new_tonic) % 12 in TONIC_DEGREES:
            split_idx -= 1
        else:
            break
    return {
        "start_bar": split_idx,
        "start_s": bars[split_idx]["start"],
        "new_tonic": new_tonic,
        "new_name_sharp": PC_NAMES_SHARP[new_tonic],
        "new_name_flat": PC_NAMES_FLAT[new_tonic],
        "new_mode": mode,
        "semitones": shift,
    }


def roman_and_function(chord_info: dict, tonic: int, mode: str) -> tuple[str, str]:
    """Retorna o grau romano e a função harmônica (T, SD, D, D2) em relação à tônica."""
    root = chord_info.get("root")
    family = chord_info.get("family")
    if root is None or family == "none":
        return "N", "N"
    rel = (int(root) - int(tonic)) % 12
    is_minor_key = str(mode).lower().startswith("men")
    if is_minor_key:
        if rel == 0 and family == "min":
            return "i", "T"
        if rel == 2 and family in {"dim", "min"}:
            return "ii°", "SD"
        if rel == 3 and family in {"maj", "dom"}:
            return ("bIII7" if family == "dom" else "bIII"), "T"
        if rel == 5 and family in {"min", "maj", "dom"}:
            return ("iv" if family == "min" else ("IV7" if family == "dom" else "IV")), "SD"
        if rel == 7 and family in {"maj", "dom", "sus"}:
            return ("V7" if family == "dom" else "V"), "D"
        if rel == 7 and family == "min":
            return "v", "D"
        if rel == 8 and family in {"maj", "dom"}:
            return ("bVI7" if family == "dom" else "bVI"), "SD"
        if rel == 10 and family in {"maj", "dom"}:
            return ("bVII7" if family == "dom" else "bVII"), "SD"
        if rel == 0 and family in {"maj", "dom"}:
            return "V7/iv", "D2"
    else:
        bass = chord_info.get("bass")
        if rel == 7 and bass is not None and (int(bass) - int(tonic)) % 12 == 5:
            return "V/IV", "D"
        if rel == 2 and bass is not None and (int(bass) - int(tonic)) % 12 == 0:
            return "V/V", "D"
        mapped = MAJOR_FUNCTIONS.get((rel, "maj" if family == "sus" else family))
        if mapped is not None:
            return mapped

    degree_names = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
    deg = degree_names[rel]
    if family == "min":
        deg = deg.lower()
    elif family == "dim":
        deg = deg.lower() + "°"
    elif family == "dom":
        deg = deg + "7"
    func = "T" if rel in {0, 3, 4, 9} else ("D" if rel in {7, 11} else "SD")
    return deg, func


def bar_signature(bar: dict, tonic: int, mode: str) -> dict:
    """Resumo harmônico de um compasso, considerando todos os acordes tocados nele."""
    chords = [c for c in (bar.get("chords") or ["N"]) if c]
    parsed = [parse_chord_symbol(c) for c in chords]
    primary = parsed[0] if parsed else parse_chord_symbol("N")
    if len(parsed) >= 2 and parsed[0]["family"] == "sus" and parsed[-1]["root"] is not None:
        primary = parsed[-1]
    roots_in_bar = {p["root"] for p in parsed if p["root"] is not None}
    roman, func = roman_and_function(primary, tonic, mode)
    last = parsed[-1] if parsed else primary
    end_roman, end_func = roman_and_function(last, tonic, mode)
    return {
        "chords": chords,
        "parsed": parsed,
        "primary": primary,
        "last": last,
        "root": primary["root"],
        "roots": roots_in_bar,
        "family": primary["family"],
        "roman": roman,
        "function": func,
        "end_roman": end_roman,
        "end_function": end_func,
        "confidence": float(bar.get("confidence", 0.5)),
    }


def bar_similarity(sig_a: dict, sig_b: dict) -> float:
    """Similaridade harmônica entre dois compassos em [0, 1]."""
    if sig_a["root"] is None and sig_b["root"] is None:
        return 1.0
    if sig_a["root"] is None or sig_b["root"] is None:
        return 0.0
    if sig_a["chords"] == sig_b["chords"]:
        return 1.0
    if sig_a["root"] == sig_b["root"]:
        if sig_a["family"] == sig_b["family"]:
            return 0.94
        if {sig_a["family"], sig_b["family"]} <= {"maj", "dom", "sus"}:
            return 0.88
        return 0.68
    if sig_a["roots"] & sig_b["roots"]:
        return 0.82
    return 0.0


def override_bar(bar: dict, chords: list[str], rule: str) -> None:
    """Troca a cifra do compasso por uma regra e mantém a grade e o texto iguais.

    O que o modelo disse fica em ``events_modelo``. Sem tempos de troca medidos
    para a cifra nova, os acordes dividem o compasso em partes iguais.
    """
    chords = list(chords) or ["N"]
    if list(bar.get("chords") or []) == chords:
        return
    if "events_modelo" not in bar:
        bar["events_modelo"] = list(bar.get("events") or [])
        bar["chords_modelo"] = list(bar.get("chords") or [])
    start, end = float(bar["start"]), float(bar["end"])
    step = (end - start) / len(chords)
    bar["events"] = [
        {
            "chord": chord,
            "start": round(start + step * index, 3),
            "end": round(end if index + 1 == len(chords) else start + step * (index + 1), 3),
            "confidence": bar.get("confidence", 0.0),
            "origem": "regra",
        }
        for index, chord in enumerate(chords)
    ]
    bar["chords"] = chords
    bar["root"] = parse_chord_symbol(chords[0])["root_name"]
    bar["origem"] = "regra"
    bar["regra"] = rule
