"""Mapeia seções a partir dos ciclos, da voz e da teoria musical."""

from __future__ import annotations

from collections import Counter

import numpy as np

from constants import (
    BRIDGE_FRACTION,
    CHEST_ONSET_CANDIDATES,
    CHEST_ONSET_RATIO,
    ONSET_SNAP_BARS,
    PITCH_TO_PC,
    SHORT_SECTION_SPAN,
    VOCAL_NEXT_RATIO,
    VOCAL_PEAK_FRACTION,
    VOCAL_PERCENTILE,
    VOCAL_PERCENTILE_GAIN,
)
from cycles import classify_cadence, discover_repeating_cycles
from repeats import lock_repeated_phrases
from spelling import (
    detect_key_modulation,
    infer_diatonic_key_from_bars,
    key_fitting_the_chords,
    key_spelling,
    respell_bars,
    respell_chord_symbol,
)


def _refine_key(bars: list[dict], key: dict) -> tuple[int, str, dict | None]:
    refined_key = infer_diatonic_key_from_bars(bars, key)
    key.update(refined_key)
    corrected = key_fitting_the_chords(bars, key)
    if corrected is not None:
        key.update(corrected)
        key["rival"] = None
        key["quinto"] = None
    tonic = int(key.get("tonic", PITCH_TO_PC.get(key.get("name", "C"), 0)))
    mode = str(key.get("mode", "maior"))
    modulation = detect_key_modulation(bars, tonic, mode)
    if modulation is not None:
        key["modulacao"] = modulation
    key["names"] = key_spelling(tonic, mode)
    key["name"] = key["names"][tonic]
    respell_bars(bars, tonic, mode, modulation)
    return tonic, mode, modulation


def _vocal_onset_from_profile(vocal_presence: np.ndarray, vocal_chest: np.ndarray, n: int) -> int:
    peak_v = float(np.max(vocal_presence)) if vocal_presence.size else 0.0
    active_v = vocal_presence[vocal_presence > peak_v * VOCAL_PEAK_FRACTION]
    thresh = float(np.percentile(active_v, VOCAL_PERCENTILE) * VOCAL_PERCENTILE_GAIN) if active_v.size >= 4 else peak_v * 0.5
    vocal_onset_bar = 0
    for idx in range(n - 1):
        if vocal_presence[idx] >= thresh and vocal_presence[idx + 1] >= thresh * VOCAL_NEXT_RATIO:
            vocal_onset_bar = idx
            break
    for candidate_bar in CHEST_ONSET_CANDIDATES:
        if candidate_bar < n and candidate_bar + 4 <= n:
            before_c = float(np.median(vocal_chest[:candidate_bar]))
            after_c = float(np.median(vocal_chest[candidate_bar : candidate_bar + 4]))
            if before_c > 0 and after_c >= before_c * CHEST_ONSET_RATIO:
                vocal_onset_bar = candidate_bar
                break
    return vocal_onset_bar


def _vocal_onset_from_blocks(blocks: list[dict]) -> int:
    non_silent = [b for b in blocks if b["kind"] != "silence"]
    if len(non_silent) >= 2 and non_silent[0]["pattern_id"] == non_silent[1]["pattern_id"]:
        return non_silent[0]["end_bar"]
    if non_silent:
        return non_silent[0]["end_bar"]
    return 0


def _snap_vocal_onset(blocks: list[dict], vocal_onset_bar: int) -> int:
    block_starts = [b["start_bar"] for b in blocks if b["kind"] != "silence"]
    if block_starts and vocal_onset_bar > 0:
        nearest_start = min(block_starts, key=lambda b_start: abs(b_start - vocal_onset_bar))
        if abs(nearest_start - vocal_onset_bar) <= ONSET_SNAP_BARS and nearest_start > 0:
            vocal_onset_bar = nearest_start
    non_silent = [b for b in blocks if b["kind"] != "silence"]
    if (
        len(non_silent) >= 2
        and non_silent[0]["kind"] == "cycle"
        and non_silent[0]["period"] == 4
        and non_silent[0]["pattern_id"] == non_silent[1]["pattern_id"]
        and vocal_onset_bar <= non_silent[0]["start_bar"]
    ):
        vocal_onset_bar = non_silent[0]["end_bar"]
    return vocal_onset_bar


def _primary_pattern(blocks: list[dict], vocal_onset_bar: int) -> str:
    post_intro_blocks = [b for b in blocks if b["end_bar"] > vocal_onset_bar and b["kind"] != "silence"]
    pattern_counts: Counter[str] = Counter()
    for b in post_intro_blocks:
        if b["kind"] == "cycle":
            pattern_counts[b["pattern_id"]] += b["end_bar"] - b["start_bar"]
    return pattern_counts.most_common(1)[0][0] if pattern_counts else "Ciclo A"


def _label_block(
    idx_b: int,
    block: dict,
    blocks: list[dict],
    bars: list[dict],
    n: int,
    vocal_onset_bar: int,
    mod_bar: int,
    primary_pattern: str,
    seen_verse: bool,
    labeled_blocks: list[dict],
    ai_bar_labels: list[str] | None,
) -> tuple[str, bool]:
    b0, b1 = block["start_bar"], block["end_bar"]
    pid = block["pattern_id"]
    romans = list(block.get("romans") or [])
    functions = list(block.get("functions") or [])

    if block["kind"] == "silence":
        return ("introdução" if b1 <= max(2, vocal_onset_bar) else "outro"), seen_verse
    if b0 >= n - 5 and all(bars[k]["chords"] == ["N"] for k in range(b0, b1)):
        return "outro", seen_verse
    if b1 <= vocal_onset_bar:
        return "introdução", seen_verse
    if b0 >= mod_bar:
        return "ponte / modulação", seen_verse
    if block["kind"] in {"turnaround", "pickup"}:
        next_block = blocks[idx_b + 1] if idx_b + 1 < len(blocks) else None
        if block["kind"] == "turnaround" and all(f == "D" for f in functions):
            return ("refrão" if seen_verse else "verso"), seen_verse
        if block["kind"] == "pickup" and next_block and next_block.get("pattern_id") != primary_pattern:
            return ("pré-refrão" if seen_verse else "verso"), seen_verse
        if labeled_blocks:
            return labeled_blocks[-1]["name"], seen_verse
        if next_block:
            return "verso", seen_verse
        return "outro", seen_verse

    ai_vote = None
    if ai_bar_labels and len(ai_bar_labels) >= b1:
        votes = Counter(ai_bar_labels[b0:b1])
        if votes:
            ai_vote = votes.most_common(1)[0][0]

    has_dominant_hook = (
        romans.count("V7") >= 2
        or functions.count("D") >= 2
        or (idx_b > 0 and blocks[idx_b - 1]["kind"] == "turnaround")
    )
    is_modal_prechorus = len(romans) >= 2 and romans[:2] == ["bIII", "bVI"]

    if is_modal_prechorus:
        sec_name = "pré-refrão"
    elif pid == primary_pattern:
        prev_b = labeled_blocks[-1] if labeled_blocks else None
        next_b = blocks[idx_b + 1] if idx_b + 1 < len(blocks) else None
        if (
            prev_b
            and prev_b["name"] in {"pré-refrão", "refrão"}
            and next_b
            and next_b.get("pattern_id") != primary_pattern
        ):
            sec_name = "refrão"
        else:
            sec_name = "verso"
            seen_verse = True
    elif has_dominant_hook or "D2" in functions:
        sec_name = "refrão"
    elif ai_vote in {"refrão", "ponte", "verso", "outro"} and ai_vote != "introdução":
        sec_name = ai_vote
    elif block.get("repeat_count", 1) >= 2:
        sec_name = "refrão"
    elif seen_verse and b0 > int(n * BRIDGE_FRACTION):
        sec_name = "ponte"
    else:
        sec_name = "verso" if not seen_verse else "refrão"
    if sec_name == "verso":
        seen_verse = True
    return sec_name, seen_verse


def _smooth_short_islands(labeled_blocks: list[dict]) -> None:
    for idx_b in range(1, len(labeled_blocks) - 1):
        prev_s = labeled_blocks[idx_b - 1]["name"]
        curr_s = labeled_blocks[idx_b]["name"]
        next_s = labeled_blocks[idx_b + 1]["name"]
        span = labeled_blocks[idx_b]["end_bar"] - labeled_blocks[idx_b]["start_bar"]
        if prev_s == next_s and curr_s != prev_s and span <= SHORT_SECTION_SPAN and curr_s not in {"pré-refrão"}:
            labeled_blocks[idx_b]["name"] = prev_s


def _merge_sections(labeled_blocks: list[dict], bars: list[dict]) -> list[dict]:
    merged_sections: list[dict] = []
    for item in labeled_blocks:
        b0, b1 = item["start_bar"], item["end_bar"]
        sec_bars = bars[b0:b1]
        if not sec_bars:
            continue
        cycle_label = " | ".join(item["romans"]) if item["romans"] else ""
        if merged_sections and merged_sections[-1]["name"] == item["name"]:
            prev = merged_sections[-1]
            prev["end"] = sec_bars[-1]["end"]
            prev["bars"].extend(sec_bars)
            prev["repeats"] = len(prev["bars"])
            if item["pattern_id"] not in prev["patterns"]:
                prev["patterns"].append(item["pattern_id"])
            if cycle_label and cycle_label not in prev["roman_cycles"]:
                prev["roman_cycles"].append(cycle_label)
            if prev["cadence"] == "silêncio" and item["cadence"] != "silêncio":
                prev["cadence"] = item["cadence"]
            continue
        merged_sections.append(
            {
                "name": item["name"],
                "start": sec_bars[0]["start"],
                "end": sec_bars[-1]["end"],
                "repeats": len(sec_bars),
                "patterns": [item["pattern_id"]],
                "roman_cycles": [cycle_label] if cycle_label else [],
                "cadence": item["cadence"],
                "bars": list(sec_bars),
            }
        )
    return merged_sections


def _sections_out(merged_sections: list[dict]) -> list[dict]:
    sections_out: list[dict] = []
    for sec in merged_sections:
        pattern_tag = " + ".join(p for p in sec["patterns"] if p != "silêncio")
        non_silent_romans = [
            r for r in sec["roman_cycles"] if not all(part.strip() == "N" for part in r.split("|"))
        ]
        roman_summary = " ; ".join((non_silent_romans or sec["roman_cycles"])[:2])
        sections_out.append(
            {
                "name": sec["name"],
                "start": sec["start"],
                "end": sec["end"],
                "repeats": len(sec["bars"]),
                "padrao": pattern_tag or None,
                "graus": roman_summary or None,
                "cadencia": sec["cadence"],
                "bars": [
                    {
                        "bar": bar["bar"],
                        "chords": bar["chords"],
                        "chords_sharp": [respell_chord_symbol(c, prefer_sharp=True) for c in (bar.get("chords") or ["N"])],
                        "events": bar.get("events") or [],
                        "confidence": bar["confidence"],
                    }
                    for bar in sec["bars"]
                ],
            }
        )
    return sections_out


def map_sections_with_theory(
    bars: list[dict],
    key: dict,
    vocal_presence: np.ndarray | None = None,
    vocal_chest: np.ndarray | None = None,
    ai_bar_labels: list[str] | None = None,
    vocal_onset_bar: int | None = None,
    bar_chroma: np.ndarray | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """Mapeia as seções da música unindo ciclos repetidos, teoria musical, modulação e voz.

    Com o croma de cada compasso, um trecho que repete no áudio herda a cifra
    da volta que explica esse áudio, antes da estimativa de tom e dos ciclos.
    """
    n = len(bars)
    if n == 0:
        return [], {}, []

    if bar_chroma is not None:
        lock_repeated_phrases(bars, bar_chroma)
    tonic, mode, modulation = _refine_key(bars, key)
    blocks, catalog = discover_repeating_cycles(
        bars,
        tonic,
        mode,
        harmonize_outliers=True,
        modulation=modulation,
    )

    has_vocal_profile = vocal_presence is not None and vocal_presence.size == n
    if not has_vocal_profile:
        vocal_presence = np.ones(n, dtype=float)
    if vocal_chest is None or vocal_chest.size != n:
        vocal_chest = vocal_presence.copy()

    if vocal_onset_bar is None:
        if has_vocal_profile:
            vocal_onset_bar = _vocal_onset_from_profile(vocal_presence, vocal_chest, n)
        else:
            vocal_onset_bar = _vocal_onset_from_blocks(blocks)
    vocal_onset_bar = _snap_vocal_onset(blocks, vocal_onset_bar)

    primary_pattern = _primary_pattern(blocks, vocal_onset_bar)
    labeled_blocks: list[dict] = []
    seen_verse = False
    mod_bar = modulation["start_bar"] if modulation else n + 1

    for idx_b, block in enumerate(blocks):
        romans = list(block.get("romans") or [])
        functions = list(block.get("functions") or [])
        cadence = block.get("cadence") or classify_cadence(romans, functions)
        sec_name, seen_verse = _label_block(
            idx_b,
            block,
            blocks,
            bars,
            n,
            vocal_onset_bar,
            mod_bar,
            primary_pattern,
            seen_verse,
            labeled_blocks,
            ai_bar_labels,
        )
        labeled_blocks.append(
            {
                "name": sec_name,
                "start_bar": block["start_bar"],
                "end_bar": block["end_bar"],
                "pattern_id": block["pattern_id"],
                "romans": romans,
                "functions": functions,
                "cadence": cadence,
                "block": block,
            }
        )

    _smooth_short_islands(labeled_blocks)
    merged_sections = _merge_sections(labeled_blocks, bars)
    sections_out = _sections_out(merged_sections)

    verse_s = next((s["start"] for s in sections_out if s["name"] == "verso"), None)
    chorus_s = next((s["start"] for s in sections_out if s["name"] == "refrão"), None)
    voice_info = {
        "entrada_verso_s": None if verse_s is None else round(float(verse_s), 3),
        "entrada_refrao_s": None if chorus_s is None else round(float(chorus_s), 3),
        "refrao_pela_dinamica": True,
        "detector": "teoria-musical+ciclos",
    }
    if modulation is not None:
        voice_info["modulacao"] = modulation
    return sections_out, voice_info, catalog
