"""Tom, gabaritos NNLS e cifras por tempo e por compasso."""

from __future__ import annotations

import numpy as np
from constants import (
    BASS_MIN_NOTES,
    BASS_SPELL_CLOSE_RATIO,
    BASS_SPELL_CONFIDENCE,
    BASS_SPELL_PEAK_FRACTION,
    BASS_SPELL_SCORE,
    CHORD_KINDS,
    CHORD_QUALITY_SUFFIXES,
    CLOSE_SCORE_RATIO,
    EPS,
    HALF_BAR_SEVENTH_CHROMA_RATIO,
    HALF_BAR_SEVENTH_PEAK_FRACTION,
    HALF_BAR_SEVENTH_RATIO,
    KEY_BASS_SCORE_RATIO,
    KEY_FIFTH_SCORE_RATIO,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    NEAR_SCORE_RATIO,
    PRESENCE_PEAK_FRACTION,
    QUIET_BAR_GATE,
    QUIET_CHORD_GATE,
    RELATIVE_M7_PEAK_FRACTION,
    ROOTLESS_SUS4_FRACTION,
    SEVENTH_TONE_WEIGHT,
    SLASH_SEVENTH_CHROMA_FRACTION,
    SUS4_MARGIN,
    TEMPERLEY_MAJOR,
    TEMPERLEY_MINOR,
)
from pulse import bar_starts


def profile_scores(chroma: np.ndarray, major: np.ndarray, minor: np.ndarray) -> list[dict]:
    ranked = []
    for tonic in range(12):
        for mode, profile in (("maior", major), ("menor", minor)):
            rotated = np.roll(profile, tonic)
            rotated = rotated / np.linalg.norm(rotated)
            ranked.append({"tonic": tonic, "mode": mode, "score": float(np.dot(chroma, rotated))})
    return ranked


def bar_presence(
    beat_chroma: np.ndarray,
    energy: np.ndarray,
    beats_per_bar: int,
    phase: int,
    downbeats: np.ndarray | None = None,
) -> np.ndarray:
    """Cada compasso vota presente ou ausente. A corda forte não pesa pela duração."""
    steps = beat_chroma.shape[1]
    starts = bar_starts(steps, beats_per_bar, phase, downbeats)
    quiet = max(float(np.median(energy)) * QUIET_BAR_GATE, EPS)
    counts = np.zeros(12)
    for bar_index, start in enumerate(starts):
        stop = int(starts[bar_index + 1]) if bar_index + 1 < starts.size else steps
        if float(np.mean(energy[start:stop])) < quiet:
            continue
        column = np.mean(beat_chroma[:, start:stop], axis=1)
        peak = float(np.max(column))
        if peak < EPS:
            continue
        counts += (column >= peak * PRESENCE_PEAK_FRACTION).astype(float)
    return counts / (np.linalg.norm(counts) + EPS)


def estimate_key(
    beat_chroma: np.ndarray,
    bass_pc: np.ndarray,
    bass_conf: np.ndarray,
    energy: np.ndarray,
    beats_per_bar: int,
    phase: int,
    downbeats: np.ndarray | None = None,
) -> dict:
    from harmonic_structure import key_spelling

    mean = np.mean(beat_chroma, axis=1)
    mean = mean / (np.linalg.norm(mean) + EPS)
    ranked = profile_scores(mean, MAJOR_PROFILE, MINOR_PROFILE)
    best = max(ranked, key=lambda item: item["score"])
    strong = bass_pc[bass_conf >= BASS_SPELL_CONFIDENCE]
    strong = strong[strong >= 0]
    early = strong[: max(4, strong.size // 2)] if strong.size else strong
    if early.size:
        tonic_pc = int(np.bincount(early.astype(int), minlength=12).argmax())
        anchored = [item for item in ranked if item["tonic"] == tonic_pc]
        if anchored:
            chosen = max(anchored, key=lambda item: item["score"])
            if chosen["score"] >= best["score"] * KEY_BASS_SCORE_RATIO:
                best = chosen
    presence = bar_presence(beat_chroma, energy, beats_per_bar, phase, downbeats)
    temperley = max(
        profile_scores(presence, TEMPERLEY_MAJOR, TEMPERLEY_MINOR), key=lambda item: item["score"]
    )
    dominant = best
    fifth_below = (best["tonic"] - 7) % 12
    fifth_key = max(
        (item for item in ranked if item["tonic"] == fifth_below), key=lambda item: item["score"]
    )
    switched = False
    if (
        temperley["tonic"] == fifth_below
        and fifth_key["score"] >= best["score"] * KEY_FIFTH_SCORE_RATIO
    ):
        matched = [
            item
            for item in ranked
            if item["tonic"] == temperley["tonic"] and item["mode"] == temperley["mode"]
        ]
        best = matched[0] if matched else fifth_key
        switched = True
    names = key_spelling(best["tonic"], best["mode"])
    rival = None
    quinto = None
    if switched and dominant["tonic"] != best["tonic"]:
        rival_names = key_spelling(dominant["tonic"], dominant["mode"])
        rival = {
            "name": rival_names[dominant["tonic"]],
            "mode": dominant["mode"],
            "score": round(dominant["score"], 3),
        }
    elif dominant["score"] > 0 and fifth_key["score"] >= dominant["score"] * KEY_FIFTH_SCORE_RATIO:
        fifth_names = key_spelling(fifth_key["tonic"], fifth_key["mode"])
        quinto = {
            "name": fifth_names[fifth_key["tonic"]],
            "mode": fifth_key["mode"],
            "relacao": round(fifth_key["score"] / dominant["score"], 3),
        }
    return {
        "tonic": best["tonic"],
        "name": names[best["tonic"]],
        "mode": best["mode"],
        "names": names,
        "score": round(best["score"], 3),
        "rival": rival,
        "quinto": quinto,
    }


def chord_templates() -> tuple[np.ndarray, list[str]]:
    vectors = []
    suffixes = []
    for suffix, intervals in CHORD_KINDS:
        vector = np.zeros(12)
        for tone in intervals:
            vector[tone] = (
                SEVENTH_TONE_WEIGHT if tone == intervals[-1] and len(intervals) > 3 else 1.0
            )
        vector /= np.linalg.norm(vector)
        vectors.append(vector)
        suffixes.append(suffix)
    return np.stack(vectors), suffixes


def chord_emission(beat_chroma: np.ndarray, rolled: np.ndarray) -> np.ndarray:
    kinds = rolled.shape[0]
    steps = beat_chroma.shape[1]
    emission = np.zeros((12 * kinds, steps))
    for root in range(12):
        rotated = np.roll(beat_chroma, -root, axis=0)
        emission[root * kinds : (root + 1) * kinds, :] = rolled @ rotated
    return emission


def scale_steps(mode: str) -> tuple[int, ...]:
    if mode == "menor":
        return (0, 2, 3, 5, 7, 8, 10)
    return (0, 2, 4, 5, 7, 9, 11)


def bass_spells(chroma: np.ndarray, bass_pc: int, rolled: np.ndarray) -> int | None:
    """O grave é a fundamental quando as notas de cima cabem nesse acorde.

    A nona pode colorir. Uma nota fora do acorde impede a leitura.
    """
    peak = float(np.max(chroma))
    if peak < EPS or bass_pc < 0:
        return None
    ranked = [
        int(pitch)
        for pitch in np.argsort(chroma)[::-1]
        if int(pitch) != bass_pc and chroma[int(pitch)] >= peak * BASS_SPELL_PEAK_FRACTION
    ]
    if len(ranked) > 3:
        close = float(chroma[ranked[2]]) * BASS_SPELL_CLOSE_RATIO
        ranked = [pitch for pitch in ranked if chroma[pitch] >= close]
    observed = ranked
    if len(observed) < 2:
        return None
    color = {(bass_pc + 2) % 12}
    fitting = []
    for kind, (_suffix, intervals) in enumerate(CHORD_KINDS):
        allowed = {(bass_pc + tone) % 12 for tone in intervals}
        if all(pitch in allowed or pitch in color for pitch in observed):
            fitting.append(kind)
    if not fitting:
        return None
    rotated = np.roll(chroma, -bass_pc)
    scores = []
    for kind in fitting:
        template = rolled[kind].copy()
        template[0] = 0.0
        norm = float(np.linalg.norm(template))
        scores.append(0.0 if norm < EPS else float(np.dot(rotated, template / norm)))
    if max(scores) < BASS_SPELL_SCORE:
        return None
    return bass_pc * rolled.shape[0] + fitting[int(np.argmax(scores))]


def major_triad_root(pitches: list[int]) -> int | None:
    pool = set(pitches)
    for root in pool:
        if (root + 4) % 12 in pool and (root + 7) % 12 in pool:
            return root
    return None


def relative_minor_seventh(chroma: np.ndarray, bass_pc: int) -> bool:
    """Tríade maior com o baixo uma terça menor abaixo é o menor com sétima.

    As três notas mais fortes fecham a tríade. A quarta só entra se chegar
    perto da terceira, e mesmo assim não desfaz essa leitura.
    """
    peak = float(np.max(chroma))
    if peak < EPS or bass_pc < 0:
        return False
    order = [
        int(pitch)
        for pitch in np.argsort(chroma)[::-1]
        if chroma[int(pitch)] >= peak * RELATIVE_M7_PEAK_FRACTION
    ]
    if len(order) < 3:
        return False
    triad = major_triad_root(order[:3])
    if triad is None:
        return False
    return bass_pc == (triad - 3) % 12


def kind_index(suffix: str) -> int:
    return next(index for index, (name, _intervals) in enumerate(CHORD_KINDS) if name == suffix)


def demote_sus4(choice: int, bar_scores: np.ndarray, kinds: int) -> int:
    """O sus4 fica só quando a quarta ganha da tríade da mesma fundamental."""
    if choice % kinds != kind_index("sus4"):
        return choice
    root = choice // kinds
    major = float(bar_scores[root * kinds + kind_index("")])
    minor = float(bar_scores[root * kinds + kind_index("m")])
    if major >= minor:
        triad, triad_kind = major, kind_index("")
    else:
        triad, triad_kind = minor, kind_index("m")
    if float(bar_scores[choice]) < triad * SUS4_MARGIN:
        return root * kinds + triad_kind
    return choice


def rootless_sus4(
    choice: int, bar_chroma: np.ndarray, bar_scores: np.ndarray, kinds: int, bass_pc: int | None
) -> int:
    """Sus4 sem a fundamental no croma nem no baixo volta para o outro gabarito."""
    if choice % kinds != kind_index("sus4"):
        return choice
    root = choice // kinds
    peak = float(np.max(bar_chroma))
    root_heard = peak >= EPS and float(bar_chroma[root]) >= peak * ROOTLESS_SUS4_FRACTION
    if root_heard or bass_pc == root:
        return choice
    masked = np.array(bar_scores, copy=True)
    for candidate in range(12):
        masked[candidate * kinds + kind_index("sus4")] = -1.0
    return int(np.argmax(masked))


def half_bar_seventh(
    emission: np.ndarray,
    chroma: np.ndarray,
    start: int,
    stop: int,
    choice: int,
    kinds: int,
    bass_pc: int | None,
) -> tuple[list[int], list[int], int] | None:
    """Na segunda metade do compasso a mesma fundamental ganha a sétima, se ela só aparece aí.

    O baixo no sétimo grau não conta: isso já é a barra, não um acorde 7.
    """
    length = stop - start
    if length < 4:
        return None
    root = choice // kinds
    quality = choice % kinds
    if quality == kind_index(""):
        seventh = kind_index("7")
    elif quality == kind_index("m"):
        seventh = kind_index("m7")
    else:
        return None
    tone = (root + 10) % 12
    if bass_pc is not None and bass_pc == tone:
        return None
    beats = list(range(start, stop))
    first = [beat for beat in beats if beat - start < length // 2]
    second = [beat for beat in beats if beat - start >= length // 2]
    if len(first) < 2 or len(second) < 2:
        return None
    second_scores = np.mean(emission[:, second], axis=1)
    base = float(second_scores[root * kinds + quality])
    alt = float(second_scores[root * kinds + seventh])
    if alt < base * HALF_BAR_SEVENTH_RATIO:
        return None
    first_chroma = np.mean(chroma[:, first], axis=1)
    second_chroma = np.mean(chroma[:, second], axis=1)
    if float(second_chroma[tone]) < float(first_chroma[tone]) * HALF_BAR_SEVENTH_CHROMA_RATIO:
        return None
    peak = float(np.max(second_chroma))
    if peak < EPS or float(second_chroma[tone]) < peak * HALF_BAR_SEVENTH_PEAK_FRACTION:
        return None
    return first, second, root * kinds + seventh


def choose_beat_state(
    column: np.ndarray, previous_root: int | None, kinds: int, diatonic: set[int]
) -> int:
    order = np.argsort(column)
    best = int(order[-1])
    second = int(order[-2])
    choice = best
    if previous_root is not None and column[best] < column[second] * CLOSE_SCORE_RATIO:
        near = [
            int(state)
            for state in range(column.size)
            if column[state] >= column[best] * NEAR_SCORE_RATIO
        ]
        in_key = [state for state in near if state // kinds in diatonic]
        pool = in_key or near
        step_up = (previous_root + 2) % 12
        fifth = (previous_root + 7) % 12
        for wanted in (step_up, fifth, previous_root):
            rooted = [state for state in pool if state // kinds == wanted]
            if rooted:
                choice = max(rooted, key=lambda state: column[state])
                break
    return choice


def demote_slash_seventh(
    choice: int,
    suffix: str,
    slash: int | None,
    bar_chroma: np.ndarray,
    kinds: int,
    suffixes: list[str],
) -> tuple[int, str]:
    root = choice // kinds
    if slash is None or suffix not in {"7", "m7"} or (root + 10) % 12 != slash:
        return choice, suffix
    peak = float(np.max(bar_chroma))
    if peak < EPS or bar_chroma[slash] < peak * SLASH_SEVENTH_CHROMA_FRACTION:
        suffix = "m" if suffix == "m7" else ""
        choice = root * kinds + kind_index(suffix)
        suffix = suffixes[choice % kinds]
    return choice, suffix


def common_bass(bass_pc: np.ndarray, bass_conf: np.ndarray, start: int, stop: int) -> int | None:
    bass_notes = [
        int(bass_pc[beat])
        for beat in range(start, stop)
        if bass_conf[beat] >= BASS_SPELL_CONFIDENCE and bass_pc[beat] >= 0
    ]
    if len(bass_notes) < BASS_MIN_NOTES:
        return None
    common = int(np.bincount(bass_notes, minlength=12).argmax())
    if bass_notes.count(common) < BASS_MIN_NOTES:
        return None
    return common


def label_chords(
    beat_chroma: np.ndarray,
    energy: np.ndarray,
    bass_pc: np.ndarray,
    bass_conf: np.ndarray,
    key: dict,
    beats_per_bar: int,
    phase: int,
    downbeats: np.ndarray | None = None,
) -> list[dict]:
    rolled, suffixes = chord_templates()
    kinds = len(suffixes)
    steps = beat_chroma.shape[1]
    emission = chord_emission(beat_chroma, rolled)
    names = key["names"]
    diatonic = {(key["tonic"] + step) % 12 for step in scale_steps(key["mode"])}
    starts = bar_starts(steps, beats_per_bar, phase, downbeats)
    labels = ["N"] * steps
    confidences = [0.0] * steps
    # 0.24, não 0.35: a mediana é a mistura com voz. O violão sozinho fica
    # perto de 0.30 e o silêncio de verdade fica abaixo de 0.21.
    quiet = max(float(np.median(energy)) * QUIET_CHORD_GATE, EPS)
    previous_root = None
    previous_state = None
    for bar_index, start in enumerate(starts):
        stop = int(starts[bar_index + 1]) if bar_index + 1 < starts.size else steps
        if float(np.mean(energy[start:stop])) < quiet:
            previous_root = None
            previous_state = None
            continue
        choices = []
        spelled_states = []
        for beat in range(start, stop):
            spelled = None
            if bass_conf[beat] >= BASS_SPELL_CONFIDENCE and bass_pc[beat] >= 0:
                spelled = bass_spells(beat_chroma[:, beat], int(bass_pc[beat]), rolled)
            if spelled is not None:
                spelled_states.append(spelled)
                choices.append(spelled)
                continue
            choices.append(choose_beat_state(emission[:, beat], previous_root, kinds, diatonic))
        counts = {}
        for state in choices:
            counts[state] = counts.get(state, 0) + 1
        top = max(counts.values())
        tied = [state for state, count in counts.items() if count == top]
        choice = choices[0] if choices[0] in tied else tied[0]
        bar_scores = np.mean(emission[:, start:stop], axis=1)
        bass_forced = spelled_states.count(choice) >= 2
        if not bass_forced and choice % kinds == kind_index("sus4"):
            choice = int(np.argmax(bar_scores))
        choice = demote_sus4(choice, bar_scores, kinds)
        common = common_bass(bass_pc, bass_conf, start, stop)
        bar_chroma = np.mean(beat_chroma[:, start:stop], axis=1)
        if common is not None and relative_minor_seventh(bar_chroma, common):
            choice = common * kinds + kind_index("m7")
            bass_forced = True
        root = choice // kinds
        suffix = suffixes[choice % kinds]
        slash = common if common is not None and common != root else None
        choice, suffix = demote_slash_seventh(choice, suffix, slash, bar_chroma, kinds, suffixes)
        choice = demote_sus4(choice, bar_scores, kinds)
        choice = rootless_sus4(choice, bar_chroma, bar_scores, kinds, common)
        same_root = previous_state is not None and previous_state // kinds == choice // kinds
        if (
            same_root
            and not bass_forced
            and bar_scores[choice] < bar_scores[previous_state] * CLOSE_SCORE_RATIO
        ):
            choice = previous_state
        root = choice // kinds
        suffix = suffixes[choice % kinds]
        slash = common if common is not None and common != root else None
        choice, suffix = demote_slash_seventh(choice, suffix, slash, bar_chroma, kinds, suffixes)
        root = choice // kinds
        label = names[root] + suffix
        if slash is not None:
            label = f"{label}/{names[slash]}"
        score = float(bar_scores[choice])
        split = half_bar_seventh(
            emission,
            beat_chroma,
            start,
            stop,
            choice,
            kinds,
            common,
        )
        if split is None:
            for beat in range(start, stop):
                labels[beat] = label
                confidences[beat] = score
        else:
            first, second, tail = split
            tail_label = names[root] + suffixes[tail % kinds]
            if slash is not None:
                tail_label = f"{tail_label}/{names[slash]}"
            tail_score = float(np.mean(emission[tail, second]))
            for beat in first:
                labels[beat] = label
                confidences[beat] = score
            for beat in second:
                labels[beat] = tail_label
                confidences[beat] = tail_score
        previous_root = root
        previous_state = choice
    return [
        {"beat": beat, "label": label, "confidence": round(confidences[beat], 3)}
        for beat, label in enumerate(labels)
    ]


def chord_root_name(label: str) -> str:
    head = label.split("/")[0]
    if head == "N":
        return "N"
    for suffix in CHORD_QUALITY_SUFFIXES:
        if head.endswith(suffix) and len(head) > len(suffix):
            return head[: -len(suffix)]
    return head


def fold_bar(labels: list[str]) -> list[str]:
    cleaned = labels[:]
    for index in range(1, len(cleaned) - 1):
        if cleaned[index - 1] == cleaned[index + 1] and cleaned[index] != cleaned[index - 1]:
            cleaned[index] = cleaned[index - 1]
    folded = []
    for label in cleaned:
        if not folded or folded[-1] != label:
            folded.append(label)
    return folded or ["N"]


def bars_from(
    beat_times: np.ndarray,
    chords: list[dict],
    beats_per_bar: int,
    phase: int,
    downbeat_indexes: np.ndarray | None = None,
) -> list[dict]:
    downbeats = bar_starts(beat_times.size, beats_per_bar, phase, downbeat_indexes)
    interval = float(np.median(np.diff(beat_times)))
    bars = []
    held = ["N"]
    for bar_number, start in enumerate(downbeats, start=1):
        stop = int(downbeats[bar_number]) if bar_number < downbeats.size else beat_times.size
        chosen = [chords[beat] for beat in range(int(start), stop)]
        notes = fold_bar([item["label"] for item in chosen])
        if notes != ["N"]:
            held = notes
        elif not any(item["label"] != "N" for item in chosen):
            held = ["N"]
        end = (
            float(beat_times[stop]) if stop < beat_times.size else float(beat_times[-1] + interval)
        )
        confidences = [item["confidence"] for item in chosen]
        bars.append(
            {
                "bar": bar_number,
                "start": round(float(beat_times[start]), 3),
                "end": round(end, 3),
                "chords": notes if notes != ["N"] or held == ["N"] else held,
                "root": chord_root_name((notes if notes != ["N"] else held)[0]),
                "confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
                "origem": "gabarito-nnls",
            }
        )
    return bars
