"""Ciclos harmônicos repetidos, anacruse e consenso de padrão."""

from __future__ import annotations

from collections import Counter

import numpy as np
from constants import (
    CONSECUTIVE_SIM,
    CYCLE_REPEAT_SIM,
    FAMILY_SIM,
    OUTLIER_CONFIDENCE,
    OUTLIER_SIM,
    PICKUP_SIM_DELTA,
    QUIET_LAP_WEIGHT,
    STRONG_SLOT_SIM,
)
from spelling import (
    bar_signature,
    bar_similarity,
    key_spelling,
    override_bar,
    parse_chord_symbol,
    respell_chord_symbol,
    roman_and_function,
)


def canonical_chord_choice(
    bars_at_slot: list[dict], sigs_at_slot: list[dict], weights: list[float] | None = None
) -> list[str]:
    """Escolhe o acorde representativo de uma posição dentro de um ciclo repetido.

    Cada volta vota com a confiança do modelo. ``weights`` multiplica o voto de cada
    volta: a volta sem voz vale mais, porque ali a harmonia soa limpa.
    """
    root_weights: Counter[int] = Counter()
    chord_weights: Counter[tuple[str, ...]] = Counter()
    for index, (bar, sig) in enumerate(zip(bars_at_slot, sigs_at_slot, strict=False)):
        if sig["root"] is None:
            continue
        weight = _vote_weight(bar, weights[index] if weights else 1.0)
        root_weights[sig["root"]] += weight
        chord_weights[tuple(bar.get("chords") or ["N"])] += weight
    if not root_weights:
        return ["N"]
    best_root = root_weights.most_common(1)[0][0]
    matching = [
        (chords_tuple, score)
        for chords_tuple, score in chord_weights.items()
        if best_root in {parse_chord_symbol(c)["root"] for c in chords_tuple}
    ]
    if not matching:
        return list(chord_weights.most_common(1)[0][0])
    matching.sort(key=lambda item: (item[1], -len(item[0])), reverse=True)
    return list(matching[0][0])


def _vote_weight(bar: dict, factor: float = 1.0) -> float:
    return max(0.15, float(bar.get("confidence", 0.6))) * factor


def _lap_weights(instances: list[dict], slot: int, quiet_until: int) -> list[float]:
    """Peso de cada volta numa posição: maior onde a voz ainda não entrou."""
    return [
        QUIET_LAP_WEIGHT if inst["start_bar"] + slot < quiet_until else 1.0 for inst in instances
    ]


def _has_majority(target: list[str], bars_at_slot: list[dict], weights: list[float]) -> bool:
    """A cifra do ciclo soma mais voto do que todas as outras leituras juntas."""
    agree = against = 0.0
    for bar, factor in zip(bars_at_slot, weights, strict=True):
        if list(bar.get("chords") or ["N"]) == target:
            agree += _vote_weight(bar, factor)
        else:
            against += _vote_weight(bar, factor)
    return agree > against


def _window_repeats_score(sigs: list[dict], start: int, period: int = 4) -> tuple[int, float]:
    """Conta quantas vezes a janela `sigs[start:start+period]` se repete na música,
    bonificando ciclos que começam na tônica (I/i) ou subdominante forte (IV/bVI)."""
    n = len(sigs)
    if start + period > n:
        return 0, 0.0
    if any(sigs[k]["root"] is None for k in range(start, start + period)):
        return 0, 0.0
    matches = 0
    sims = []
    pos = 0
    while pos <= n - period:
        if abs(pos - start) < period:
            pos += 1
            continue
        sim = float(
            np.mean([bar_similarity(sigs[start + s], sigs[pos + s]) for s in range(period)])
        )
        if sim >= CYCLE_REPEAT_SIM:
            matches += 1
            sims.append(sim)
            pos += period
        else:
            pos += 1
    base_sim = float(np.mean(sims)) if sims else 0.0
    first_rom = sigs[start]["roman"]
    last_func = sigs[start + period - 1]["function"]
    theory_bonus = 0.0
    if first_rom in {"I", "i", "IV"}:
        theory_bonus += 0.10
    if last_func in {"D", "D2", "SD"} or sigs[start + period - 1]["roman"] != first_rom:
        theory_bonus += 0.04
    return matches, base_sim + theory_bonus


def _block(start: int, end: int, kind: str, sigs: list[dict]) -> dict:
    silent = kind == "silence"
    return {
        "start_bar": start,
        "end_bar": end,
        "period": end - start,
        "kind": kind,
        "roots": () if silent else tuple(sigs[k]["root"] for k in range(start, end)),
        "romans": ("N",) * (end - start)
        if silent
        else tuple(sigs[k]["roman"] for k in range(start, end)),
        "functions": ("N",) * (end - start)
        if silent
        else tuple(sigs[k]["function"] for k in range(start, end)),
    }


def _leading_silence(sigs: list[dict]) -> tuple[list[dict], int]:
    idx = 0
    n = len(sigs)
    while idx < n and sigs[idx]["root"] is None:
        idx += 1
    if idx > 0:
        return [_block(0, idx, "silence", sigs)], idx
    return [], idx


def _silence_run(sigs: list[dict], idx: int) -> dict | None:
    if sigs[idx]["root"] is not None:
        return None
    end_sil = idx + 1
    while end_sil < len(sigs) and sigs[end_sil]["root"] is None:
        end_sil += 1
    return _block(idx, end_sil, "silence", sigs)


def _turnaround_block(sigs: list[dict], idx: int) -> dict | None:
    n = len(sigs)
    if not (
        idx + 2 <= n
        and sigs[idx]["function"] == "D"
        and sigs[idx + 1]["root"] == sigs[idx]["root"]
        and (idx + 2 == n or sigs[idx + 2]["root"] != sigs[idx]["root"])
    ):
        return None
    block = _block(idx, idx + 2, "turnaround", sigs)
    block["functions"] = ("D", "D")
    return block


def _pickup_or_turnaround(sigs: list[dict], idx: int) -> dict | None:
    n = len(sigs)
    if idx + 5 > n:
        return None
    here_matches, here_sim = _window_repeats_score(sigs, idx, 4)
    shift1_matches, shift1_sim = _window_repeats_score(sigs, idx + 1, 4)
    shift2_matches, shift2_sim = (
        _window_repeats_score(sigs, idx + 2, 4) if idx + 6 <= n else (0, 0.0)
    )
    if (
        shift1_matches >= 2
        and (
            shift1_matches > here_matches
            or (shift1_matches == here_matches and shift1_sim > here_sim + PICKUP_SIM_DELTA)
        )
        and sigs[idx]["roman"] not in {"I", "i", "IV"}
    ):
        return _block(idx, idx + 1, "pickup", sigs)
    if (
        shift2_matches >= 2
        and (
            shift2_matches > here_matches
            or (shift2_matches == here_matches and shift2_sim > here_sim + PICKUP_SIM_DELTA)
        )
        and sigs[idx]["roman"] not in {"I", "i", "IV"}
    ):
        return _block(idx, idx + 2, "turnaround", sigs)
    return None


def _consecutive_repeats(sigs: list[dict], idx: int) -> list[dict] | None:
    n = len(sigs)
    best_choice = None
    for period in (4, 2):
        if idx + period > n:
            continue
        repeats = 1
        sim_scores = []
        while idx + (repeats + 1) * period <= n:
            chunk_sims = [
                bar_similarity(
                    sigs[idx + slot],
                    sigs[idx + repeats * period + slot],
                )
                for slot in range(period)
            ]
            mean_sim = float(np.mean(chunk_sims))
            if mean_sim >= CONSECUTIVE_SIM:
                sim_scores.append(mean_sim)
                repeats += 1
            else:
                break
        if repeats >= 2:
            priority = (
                1 if period == 4 else 0,
                repeats * period,
                float(np.mean(sim_scores)),
            )
            if best_choice is None or priority > best_choice[0]:
                best_choice = (priority, period, repeats)
    if best_choice is None:
        return None
    _, period, repeats = best_choice
    return [_block(idx + r * period, idx + (r + 1) * period, "cycle", sigs) for r in range(repeats)]


def _remainder_block(sigs: list[dict], idx: int) -> dict:
    n = len(sigs)
    step = 4 if (n - idx) >= 4 else (n - idx)
    return _block(idx, idx + step, "cycle" if step == 4 else "phrase", sigs)


def _group_families(blocks: list[dict], sigs: list[dict]) -> list[dict]:
    families: list[dict] = []
    for block in blocks:
        if block["kind"] == "silence":
            block["pattern_id"] = "silêncio"
            continue
        b0, b1 = block["start_bar"], block["end_bar"]
        length = b1 - b0
        matched_family = None
        best_sim = 0.0
        for fam in families:
            if fam["period"] != length:
                continue
            ref_b0 = fam["instances"][0]["start_bar"]
            sim = float(
                np.mean([bar_similarity(sigs[b0 + s], sigs[ref_b0 + s]) for s in range(length)])
            )
            if sim >= FAMILY_SIM and sim > best_sim:
                best_sim = sim
                matched_family = fam
        if matched_family is None:
            label = f"P{len(families) + 1}"
            matched_family = {
                "id": label,
                "period": length,
                "instances": [block],
            }
            families.append(matched_family)
        else:
            matched_family["instances"].append(block)
        block["pattern_id"] = matched_family["id"]
    return families


def _is_passing_reading(bar: dict, sig: dict, target: list[str]) -> bool:
    """O compasso traz o acorde do ciclo e mais um de passagem (``F C`` onde o ciclo diz ``F``)."""
    if len(target) != 1 or len(bar.get("chords") or []) < 2:
        return False
    return parse_chord_symbol(target[0])["root"] in sig["roots"]


def _is_parallel_blink(bar: dict, target: list[str]) -> bool:
    """Dois acordes na fundamental do ciclo, o segundo menor, onde o ciclo diz maior."""
    if len(bar.get("chords") or []) != 2 or len(target) != 1:
        return False
    first = parse_chord_symbol(bar["chords"][0])
    second = parse_chord_symbol(bar["chords"][1])
    wanted = parse_chord_symbol(target[0])
    return (
        first["root"] == wanted["root"]
        and second["root"] == wanted["root"]
        and second["family"] == "min"
        and wanted["family"] == "maj"
    )


def _settle_laps(
    instances: list[dict],
    period: int,
    canonical_chords: list[list[str]],
    bars: list[dict],
    sigs: list[dict],
    mod_bar: int,
    tonic: int,
    mod_tonic: int,
    mode: str,
    quiet_until: int,
) -> None:
    """Escreve a cifra do ciclo nas voltas que saíram diferentes; o que fica, fica marcado.

    Só mexe numa volta que bate com o ciclo em todas as posições menos uma. Um compasso
    troca de cifra em três casos: a leitura dele discorda do ciclo e o modelo tinha pouca
    confiança; o ciclo é maior e o compasso pisca para o menor; ou o compasso traz o acorde
    do ciclo com outro de passagem e a maioria das voltas (a volta sem voz vale mais) não
    tem essa passagem. Diferença que sobra ganha ``variacao_do_ciclo``.
    """
    if len(instances) < 2 or period < 4:
        return
    majority = [
        _has_majority(
            canonical_chords[slot],
            [bars[inst["start_bar"] + slot] for inst in instances],
            _lap_weights(instances, slot, quiet_until),
        )
        for slot in range(period)
    ]
    for inst in instances:
        b0 = inst["start_bar"]
        inst_tonic = mod_tonic if b0 >= mod_bar else tonic
        slot_sims = [
            bar_similarity(
                sigs[b0 + s],
                bar_signature({"chords": canonical_chords[s], "confidence": 1.0}, inst_tonic, mode),
            )
            for s in range(period)
        ]
        strong_slots = sum(1 for sim in slot_sims if sim >= STRONG_SLOT_SIM)
        if strong_slots < period - 1:
            continue
        for s in range(period):
            bar = bars[b0 + s]
            target = canonical_chords[s]
            if list(bar.get("chords") or ["N"]) == target:
                continue
            unsure = (
                slot_sims[s] < OUTLIER_SIM
                and float(bar.get("confidence", 0.0)) < OUTLIER_CONFIDENCE
            )
            passing = majority[s] and _is_passing_reading(bar, sigs[b0 + s], target)
            if unsure or passing or _is_parallel_blink(bar, target):
                override_bar(bar, target, "consenso do ciclo")
            else:
                bar["variacao_do_ciclo"] = " ".join(target)


def _apply_secondary_dominant(
    period: int,
    canonical_chords: list[list[str]],
    canonical_romans: list[str],
    canonical_functions: list[str],
    instances: list[dict],
    bars: list[dict],
    local_tonic: int,
) -> None:
    if period != 4 or canonical_romans[:3] not in (["IV", "V", "I"], ["IV", "V7", "I"]):
        return
    fourth_chords = canonical_chords[3]
    fourth_info = parse_chord_symbol(fourth_chords[0])
    if fourth_info["suffix"] in {"dim", "dim7"} or fourth_info["bass"] == (local_tonic + 9) % 12:
        v7_ii_name = key_spelling(local_tonic, "maior")[(local_tonic + 9) % 12] + "7"
        canonical_chords[3] = [v7_ii_name]
        canonical_romans[3] = "V7/ii"
        canonical_functions[3] = "D2"
        for inst in instances:
            override_bar(bars[inst["start_bar"] + 3], [v7_ii_name], "V7/ii em IV–V–I")


def classify_cadence(romans: list[str], functions: list[str]) -> str:
    """Identifica a fórmula cadencial ou movimento harmônico do bloco."""
    if not functions or all(f == "N" for f in functions):
        return "silêncio"
    clean_r = [r for r in romans if r != "N"]
    clean_f = [f for f in functions if f != "N"]
    if not clean_f:
        return "estático"
    if len(clean_r) >= 4 and clean_r[:3] == ["i", "bVII", "bVI"]:
        return "ciclo descendente menor (i–bVII–bVI–bIII)"
    if len(clean_r) >= 4 and clean_r[:4] in (["IV", "V", "iii", "vi"], ["IV", "V/IV", "iii", "vi"]):
        return "progressão diatônica IV – V – iii – vi"
    if len(clean_r) >= 4 and clean_r[:3] in (["IV", "V", "I"], ["IV", "V7", "I"]):
        return "resolução maior com dominante secundário (IV – V – I – V7/ii)"
    if len(clean_r) >= 2 and clean_r[:2] == ["bIII", "bVI"]:
        return "empréstimo modal cromático (bIIIΔ – bVIΔ – ii – V)"
    joined_f = "–".join(clean_f)
    if all(f == "D" for f in clean_f):
        return "turnaround / preparação dominante (V7)"
    if clean_r.count("V7") >= 2 and "i" in clean_r:
        return "alternância tônica–dominante (i – V7)"
    if "D2" in clean_f:
        return "preparação com dominante secundário"
    if "SD–D–T" in joined_f or (clean_f[-1] == "T" and "D" in clean_f):
        return "cadência autêntica (SD–D–T)"
    if clean_f[-1] == "D":
        return "meia-cadência (suspensão no dominante)"
    if clean_f[0] == "SD" and "T" in clean_f:
        return "abertura subdominante (SD–T)"
    return f"ciclo funcional ({joined_f})"


def _catalog_families(
    families: list[dict],
    bars: list[dict],
    sigs: list[dict],
    tonic: int,
    mode: str,
    mod_bar: int,
    mod_tonic: int,
    harmonize_outliers: bool,
    quiet_until: int = 0,
) -> list[dict]:
    pattern_catalog: list[dict] = []
    for fam in families:
        period = fam["period"]
        instances = fam["instances"]
        canonical_chords: list[list[str]] = []
        canonical_romans: list[str] = []
        canonical_functions: list[str] = []
        ref_start = instances[0]["start_bar"]
        local_tonic = mod_tonic if ref_start >= mod_bar else tonic

        for slot in range(period):
            slot_bars = [bars[inst["start_bar"] + slot] for inst in instances]
            slot_sigs = [sigs[inst["start_bar"] + slot] for inst in instances]
            chosen_chords = canonical_chord_choice(
                slot_bars, slot_sigs, _lap_weights(instances, slot, quiet_until)
            )
            canonical_chords.append(chosen_chords)
            info = parse_chord_symbol(
                chosen_chords[-1]
                if len(chosen_chords) > 1 and chosen_chords[0].endswith("sus4")
                else chosen_chords[0]
            )
            rom, fn = roman_and_function(info, local_tonic, mode)
            canonical_romans.append(rom)
            canonical_functions.append(fn)

        if harmonize_outliers:
            _settle_laps(
                instances,
                period,
                canonical_chords,
                bars,
                sigs,
                mod_bar,
                tonic,
                mod_tonic,
                mode,
                quiet_until,
            )
        _apply_secondary_dominant(
            period,
            canonical_chords,
            canonical_romans,
            canonical_functions,
            instances,
            bars,
            local_tonic,
        )

        cadence = classify_cadence(canonical_romans, canonical_functions)
        for inst in instances:
            inst["canonical_chords"] = [" ".join(ch) for ch in canonical_chords]
            inst["romans"] = tuple(canonical_romans)
            inst["functions"] = tuple(canonical_functions)
            inst["cadence"] = cadence
            inst["repeat_count"] = len(instances)

        pattern_catalog.append(
            {
                "id": fam["id"],
                "compassos_por_ciclo": period,
                "repeticoes": len(instances),
                "acordes": [" ".join(ch) for ch in canonical_chords],
                "acordes_sustenido": [
                    " ".join(respell_chord_symbol(c, prefer_sharp=True) for c in ch)
                    for ch in canonical_chords
                ],
                "graus": canonical_romans,
                "funcoes": canonical_functions,
                "cadencia": cadence,
            }
        )
    return pattern_catalog


def _rename_cycles(pattern_catalog: list[dict], blocks: list[dict]) -> None:
    pattern_catalog.sort(
        key=lambda item: item["repeticoes"] * item["compassos_por_ciclo"], reverse=True
    )
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    id_map = {}
    for rank, item in enumerate(pattern_catalog):
        new_id = f"Ciclo {letters[rank]}" if rank < len(letters) else item["id"]
        id_map[item["id"]] = new_id
        item["id"] = new_id
    for block in blocks:
        pid = block.get("pattern_id")
        if pid in id_map:
            block["pattern_id"] = id_map[pid]


def discover_repeating_cycles(
    bars: list[dict],
    tonic: int,
    mode: str,
    harmonize_outliers: bool = True,
    modulation: dict | None = None,
    quiet_until: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Encontra ciclos harmônicos repetidos (com suporte a anacruse/turnaround e modulação).

    ``quiet_until`` é o compasso (contando de 0) em que a voz entra, quando se sabe:
    as voltas antes dele pesam mais no voto da cifra do ciclo.
    """
    n = len(bars)
    if n == 0:
        return [], []
    mod_bar = modulation["start_bar"] if modulation else n + 1
    mod_tonic = modulation["new_tonic"] if modulation else tonic

    sigs = [
        bar_signature(bar, mod_tonic if idx >= mod_bar else tonic, mode)
        for idx, bar in enumerate(bars)
    ]

    blocks, idx = _leading_silence(sigs)
    while idx < n:
        silence = _silence_run(sigs, idx)
        if silence is not None:
            blocks.append(silence)
            idx = silence["end_bar"]
            continue
        turnaround = _turnaround_block(sigs, idx)
        if turnaround is not None:
            blocks.append(turnaround)
            idx = turnaround["end_bar"]
            continue
        pickup = _pickup_or_turnaround(sigs, idx)
        if pickup is not None:
            blocks.append(pickup)
            idx = pickup["end_bar"]
            continue
        repeats = _consecutive_repeats(sigs, idx)
        if repeats is not None:
            blocks.extend(repeats)
            idx = repeats[-1]["end_bar"]
            continue
        remainder = _remainder_block(sigs, idx)
        blocks.append(remainder)
        idx = remainder["end_bar"]

    families = _group_families(blocks, sigs)
    pattern_catalog = _catalog_families(
        families,
        bars,
        sigs,
        tonic,
        mode,
        mod_bar,
        mod_tonic,
        harmonize_outliers,
        int(quiet_until or 0),
    )
    _rename_cycles(pattern_catalog, blocks)
    return blocks, pattern_catalog
