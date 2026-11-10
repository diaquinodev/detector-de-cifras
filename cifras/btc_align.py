"""Alinha a linha do tempo do BTC à grade de compassos."""

from __future__ import annotations

import numpy as np

from chords import chord_root_name, profile_scores
from constants import (
    BASS_INTERVAL_PAD_S,
    BASS_MIN_NOTES,
    BTC_FLICKER_CONFIDENCE,
    BTC_FLICKER_MARGIN,
    BTC_INTERVALS,
    BTC_KEY_ABOVE_RATIO,
    BTC_KEY_FIFTH_RATIO,
    BTC_KEY_WINDOW_S,
    BTC_MIN_OCCUPANCY,
    BTC_QUALITIES,
    BTC_ROOT_WEIGHT,
    BTC_VERY_WEAK_CONFIDENCE,
    BTC_VERY_WEAK_MARGIN,
    BTC_VERY_WEAK_ROOT_SUPPORT,
    BTC_WEAK_CONFIDENCE,
    BTC_WEAK_CONFIDENCE_CAP,
    BTC_WEAK_MARGIN_CAP,
    BTC_WEAK_OCCUPANCY,
    BTC_WEAK_ROOT_SUPPORT,
    CHORD_ONSET_SLACK,
    EPS,
    EVENT_MERGE_SLACK,
    MAJOR_PROFILE,
    MINOR_PROFILE,
    NATURAL_SEVEN_RATIO,
    PITCHES_SHARP,
    SLASH_BASS_CONFIDENCE,
    SLASH_BASS_SHARE,
)


def btc_parts(raw: str, names: list[str]) -> tuple[str, int | None, str]:
    if raw in {"N", "X"}:
        return raw, None, ""
    root_text, separator, quality = raw.partition(":")
    try:
        root = PITCHES_SHARP.index(root_text)
    except ValueError:
        return "X", None, ""
    suffix = BTC_QUALITIES.get(quality if separator else "maj")
    if suffix is None:
        return "X", None, ""
    return names[root] + suffix, root, suffix


def timeline_vote(
    timeline: list[dict], start: float, end: float
) -> tuple[str, float, float, float]:
    duration = max(end - start, EPS)
    votes: dict[str, float] = {}
    confidence_sum: dict[str, float] = {}
    margin_sum: dict[str, float] = {}
    for segment in timeline:
        overlap = max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))
        if overlap > 0.0:
            chord = str(segment["chord"])
            votes[chord] = votes.get(chord, 0.0) + overlap
            confidence_sum[chord] = confidence_sum.get(chord, 0.0) + overlap * float(
                segment.get("confidence", 1.0)
            )
            margin_sum[chord] = margin_sum.get(chord, 0.0) + overlap * float(
                segment.get("margin", 1.0)
            )
    if not votes:
        return "X", 0.0, 0.0, 0.0
    chord, won = max(votes.items(), key=lambda item: item[1])
    return (
        chord,
        float(won / duration),
        float(confidence_sum[chord] / max(won, EPS)),
        float(margin_sum[chord] / max(won, EPS)),
    )


def interval_bass(
    start: float,
    end: float,
    beat_times: np.ndarray,
    bass_pc: np.ndarray,
    bass_conf: np.ndarray,
) -> int | None:
    """Sem baixista, o detector ouve a corda grave da voicing; só um pedal sustentado vira barra."""
    inside = [
        index
        for index, beat in enumerate(beat_times)
        if start - BASS_INTERVAL_PAD_S <= float(beat) < end + BASS_INTERVAL_PAD_S
    ]
    notes = [
        int(bass_pc[index])
        for index in inside
        if bass_pc[index] >= 0 and bass_conf[index] >= SLASH_BASS_CONFIDENCE
    ]
    if len(notes) < BASS_MIN_NOTES:
        return None
    common = int(np.bincount(notes, minlength=12).argmax())
    return (
        common
        if notes.count(common) >= max(BASS_MIN_NOTES, SLASH_BASS_SHARE * len(inside))
        else None
    )


def btc_root_support(raw: str, chroma: np.ndarray | None) -> float:
    if chroma is None:
        return 1.0
    _label, root, _suffix = btc_parts(raw, PITCHES_SHARP)
    total = float(np.max(chroma))
    if root is None or total <= EPS:
        return 1.0
    return float(chroma[root] / total)


def spell_btc(raw: str, names: list[str], bass: int | None) -> str:
    label, root, suffix = btc_parts(raw, names)
    if root is None:
        return label
    if bass is not None and bass != root:
        # G:7 com o F sustentado no baixo é G/F: o F pertence à barra e não
        # prova que a sétima foi tocada pela harmonia.
        if suffix == "7" and bass == (root + 10) % 12:
            label = names[root]
            label = f"{label}/{names[bass]}"
        elif (bass - root) % 12 not in BTC_INTERVALS.get(suffix, {0, 4, 7}):
            label = f"{label}/{names[bass]}"
    return label


def model_change_time(
    timeline: list[dict], raw: str, left: float, right: float, fallback: float
) -> float:
    """Devolve o início que o modelo marcou, quando ele cai longe do tempo."""
    covering_start = None
    next_start = None
    for segment in timeline:
        if str(segment.get("chord")) != raw:
            continue
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_start <= fallback < seg_end and (
            covering_start is None or seg_start > covering_start
        ):
            covering_start = seg_start
        if fallback < seg_start < right and (next_start is None or seg_start < next_start):
            next_start = seg_start
    if covering_start is not None and left < covering_start < fallback - CHORD_ONSET_SLACK:
        return covering_start
    if next_start is not None and fallback + CHORD_ONSET_SLACK < next_start < right:
        return next_start
    return fallback


def bar_interior_points(start: float, end: float, beat_times: np.ndarray) -> list[float]:
    inside = [
        float(beat)
        for beat in beat_times
        if start + CHORD_ONSET_SLACK < float(beat) < end - CHORD_ONSET_SLACK
    ]
    return [start, *inside, end]


def chroma_at_beat(
    beat_chroma: np.ndarray | None, beat_times: np.ndarray, left: float
) -> tuple[np.ndarray | None, int]:
    beat_index = int(np.argmin(np.abs(beat_times - left))) if beat_times.size else -1
    chroma = None
    if (
        beat_chroma is not None
        and beat_chroma.ndim == 2
        and beat_chroma.shape[0] == 12
        and 0 <= beat_index < beat_chroma.shape[1]
    ):
        chroma = beat_chroma[:, beat_index]
    return chroma, beat_index


def local_root_at(local_chords: list[dict] | None, beat_index: int, names: list[str]) -> int | None:
    if local_chords is None or not (0 <= beat_index < len(local_chords)):
        return None
    local_name = chord_root_name(str(local_chords[beat_index].get("label", "N")))
    if local_name in names:
        return names.index(local_name)
    return None


def is_weak_change(
    previous_raw: str,
    raw: str,
    occupancy: float,
    confidence: float,
    margin: float,
    root_support: float,
    candidate_root: int | None,
    previous_root: int | None,
    local_root: int | None,
) -> bool:
    return (
        previous_raw not in {"X", "N"}
        and raw not in {"X", "N"}
        and candidate_root != previous_root
        and (
            occupancy < BTC_WEAK_OCCUPANCY
            or (
                confidence < BTC_WEAK_CONFIDENCE
                and local_root is not None
                and local_root != candidate_root
                and root_support < BTC_WEAK_ROOT_SUPPORT
            )
            or (
                confidence < BTC_VERY_WEAK_CONFIDENCE
                and margin < BTC_VERY_WEAK_MARGIN
                and root_support < BTC_VERY_WEAK_ROOT_SUPPORT
            )
        )
    )


def vote_interval(
    timeline: list[dict],
    left: float,
    right: float,
    whole: tuple[str, float, float, float],
    previous_raw: str,
    beat_times: np.ndarray,
    names: list[str],
    beat_chroma: np.ndarray | None,
    local_chords: list[dict] | None,
) -> tuple[dict, str]:
    raw, occupancy, confidence, margin = timeline_vote(timeline, left, right)
    if raw == "X" or occupancy < BTC_MIN_OCCUPANCY:
        raw, occupancy, confidence, margin = whole
    chroma, beat_index = chroma_at_beat(beat_chroma, beat_times, left)
    root_support = btc_root_support(raw, chroma)
    candidate_root = btc_parts(raw, names)[1]
    previous_root = btc_parts(previous_raw, names)[1]
    local_root = local_root_at(local_chords, beat_index, names)
    if is_weak_change(
        previous_raw,
        raw,
        occupancy,
        confidence,
        margin,
        root_support,
        candidate_root,
        previous_root,
        local_root,
    ):
        raw = previous_raw
        confidence = min(confidence, BTC_WEAK_CONFIDENCE_CAP)
        margin = min(margin, BTC_WEAK_MARGIN_CAP)
    candidate = {
        "raw": raw,
        "start": left,
        "end": right,
        "confidence": confidence,
        "margin": margin,
        "occupancy": occupancy,
    }
    if raw != "X":
        previous_raw = raw
    return candidate, previous_raw


def remove_one_beat_flicker(candidates: list[dict]) -> None:
    for index in range(1, len(candidates) - 1):
        before, item, after = candidates[index - 1], candidates[index], candidates[index + 1]
        if (
            before["raw"] == after["raw"]
            and item["raw"] != before["raw"]
            and item["confidence"] < BTC_FLICKER_CONFIDENCE
            and item["margin"] < BTC_FLICKER_MARGIN
        ):
            item["raw"] = before["raw"]


def events_from_candidates(
    candidates: list[dict],
    timeline: list[dict],
    beat_times: np.ndarray,
    bass_pc: np.ndarray,
    bass_conf: np.ndarray,
    names: list[str],
) -> list[dict]:
    events = []
    for item in candidates:
        bass = interval_bass(
            item["start"],
            item["end"],
            beat_times,
            bass_pc,
            bass_conf,
        )
        label = spell_btc(item["raw"], names, bass)
        start = float(item["start"])
        end = float(item["end"])
        if events and events[-1]["chord"] == label:
            events[-1]["end"] = round(end, 3)
            events[-1]["confidence_values"].append(float(item["confidence"]))
            continue
        if events:
            onset = model_change_time(timeline, item["raw"], float(events[-1]["start"]), end, start)
            if float(events[-1]["start"]) + EVENT_MERGE_SLACK < onset < end - EVENT_MERGE_SLACK:
                events[-1]["end"] = round(onset, 3)
                start = onset
        events.append(
            {
                "chord": label,
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence_values": [float(item["confidence"])],
                "origem": "modelo",
            }
        )
    for event in events:
        values = event.pop("confidence_values")
        event["confidence"] = round(float(np.mean(values)), 3)
    return events


def btc_bars(
    bars: list[dict],
    timeline: list[dict],
    beat_times: np.ndarray,
    bass_pc: np.ndarray,
    bass_conf: np.ndarray,
    names: list[str],
    beat_chroma: np.ndarray | None = None,
    local_chords: list[dict] | None = None,
) -> list[dict]:
    """O modelo escolhe o acorde por tempo; uma troca longe do tempo guarda o instante do modelo."""
    previous_raw = "X"
    for bar in bars:
        start, end = float(bar["start"]), float(bar["end"])
        points = bar_interior_points(start, end, beat_times)
        whole = timeline_vote(timeline, start, end)
        candidates = []
        for left, right in zip(points, points[1:]):
            candidate, previous_raw = vote_interval(
                timeline,
                left,
                right,
                whole,
                previous_raw,
                beat_times,
                names,
                beat_chroma,
                local_chords,
            )
            candidates.append(candidate)
        remove_one_beat_flicker(candidates)
        events = events_from_candidates(candidates, timeline, beat_times, bass_pc, bass_conf, names)
        bar["events"] = events
        bar["origem"] = "modelo"
        bar["chords"] = [event["chord"] for event in events] or ["N"]
        bar["root"] = chord_root_name(bar["chords"][0])
        bar["confidence"] = round(
            float(np.mean([event["confidence"] for event in events])) if events else 0.0,
            3,
        )
    return bars


def key_from_btc(timeline: list[dict], fallback: dict) -> dict:
    """Corrige o caso em que o baixo do dominante coroa a quinta.

    Usa só o primeiro minuto e só troca para a quinta abaixo. Isso resolve
    Mi→Lá menor sem transformar Fá maior em Si bemol.
    """
    from harmonic_structure import key_spelling

    pitches = np.zeros(12)
    for segment in timeline:
        start = float(segment["start"])
        if start >= BTC_KEY_WINDOW_S:
            break
        end = min(float(segment["end"]), BTC_KEY_WINDOW_S)
        duration = max(0.0, end - start)
        raw = str(segment["chord"])
        if duration <= 0.0 or raw in {"N", "X"}:
            continue
        root_text, separator, quality = raw.partition(":")
        try:
            root = PITCHES_SHARP.index(root_text)
        except ValueError:
            continue
        suffix = BTC_QUALITIES.get(quality if separator else "maj")
        if suffix is None:
            continue
        pitches[root] += duration * BTC_ROOT_WEIGHT
        for interval in BTC_INTERVALS[suffix]:
            pitches[(root + interval) % 12] += duration
    norm = float(np.linalg.norm(pitches))
    if norm < EPS:
        return fallback
    vector = pitches / norm
    ranked = profile_scores(vector, MAJOR_PROFILE, MINOR_PROFILE)
    best = max(ranked, key=lambda item: item["score"])
    current = max(
        (
            item
            for item in ranked
            if item["tonic"] == fallback["tonic"] and item["mode"] == fallback["mode"]
        ),
        key=lambda item: item["score"],
    )
    fifth_below = (fallback["tonic"] - 7) % 12
    fifth_above = (fallback["tonic"] + 7) % 12
    cand_above = max(
        (item for item in ranked if item["tonic"] == fifth_above and item["mode"] == "maior"),
        key=lambda item: item["score"],
    )
    seventh_of_fallback = (fallback["tonic"] + 10) % 12
    natural_seven_of_fallback = (fallback["tonic"] + 11) % 12
    if (
        fallback["mode"] == "maior"
        and pitches[natural_seven_of_fallback] > pitches[seventh_of_fallback] * NATURAL_SEVEN_RATIO
        and cand_above["score"] >= current["score"] * BTC_KEY_ABOVE_RATIO
    ):
        best = cand_above
    elif best["tonic"] != fifth_below or best["score"] < current["score"] * BTC_KEY_FIFTH_RATIO:
        return fallback
    previous = {
        "name": fallback["name"],
        "mode": fallback["mode"],
        "score": fallback["score"],
    }
    names = key_spelling(best["tonic"], best["mode"])
    fallback.update(
        {
            "tonic": best["tonic"],
            "name": names[best["tonic"]],
            "mode": best["mode"],
            "names": names,
            "score": round(float(best["score"]), 3),
            "rival": previous,
            "quinto": None,
        }
    )
    return fallback
