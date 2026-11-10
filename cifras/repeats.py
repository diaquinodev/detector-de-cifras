"""Repetição de trecho pela semelhança do áudio, não pela cifra já lida.

Dois trechos do mesmo tamanho entram como a mesma volta quando o croma de
cada compasso do miolo coincide e essa coincidência não é uma transposição.
A volta cujos acordes explicam o próprio áudio empresta a cifra; a confiança
só desempata. Um compasso muda quando esse acorde encaixa no croma dele
melhor do que a cifra já escrita. Maior e menor paralelos, com a terça fraca
no áudio, ficam cada um no seu lugar. O alinhamento também precisa ser um
pico: escorregar um compasso tem de piorar o encaixe. Os compassos finais ficam
de fora: a cadência da volta seguinte costuma mudar.
"""

from __future__ import annotations

import numpy as np

from constants import (
    BTC_INTERVALS,
    REPEAT_ALIGN_MARGIN,
    REPEAT_BAR_SIM,
    REPEAT_CHORD_FIT_FLOOR,
    REPEAT_CHORD_FIT_MARGIN,
    REPEAT_CORE_MEAN,
    REPEAT_LONG_PHRASE,
    REPEAT_LONG_TAIL,
    REPEAT_MAX_BARS,
    REPEAT_MIN_BARS,
    REPEAT_MIN_MATCH_BARS,
    REPEAT_MIN_MATCH_FRACTION,
    REPEAT_SHIFT_MARGIN,
    REPEAT_SHORT_TAIL,
)
from pulse import bar_starts
from spelling import override_bar, parse_chord_symbol


def chroma_by_bar(
    beat_chroma: np.ndarray,
    beats_per_bar: int,
    phase: int,
    downbeats: np.ndarray | None = None,
) -> np.ndarray:
    """Média do croma dos tempos de cada compasso. Forma (12, compassos)."""
    steps = int(beat_chroma.shape[1]) if beat_chroma.ndim == 2 else 0
    starts = bar_starts(steps, beats_per_bar, phase, downbeats)
    columns = []
    for index, start in enumerate(starts):
        stop = int(starts[index + 1]) if index + 1 < starts.size else steps
        block = beat_chroma[:, int(start) : stop]
        if block.size == 0:
            columns.append(np.zeros(12, dtype=float))
            continue
        columns.append(np.mean(block, axis=1))
    if not columns:
        return np.zeros((12, 0), dtype=float)
    return np.stack(columns, axis=1)


def _normalize_columns(chroma: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    normalized = np.zeros_like(chroma, dtype=float)
    heard = norms.ravel() > 1e-8
    normalized[:, heard] = chroma[:, heard] / norms[:, heard]
    return normalized


def _free_tail(length: int) -> int:
    if length >= REPEAT_LONG_PHRASE:
        return REPEAT_LONG_TAIL
    return REPEAT_SHORT_TAIL


def _mean_confidence(bars: list[dict], start: int, core: int) -> float:
    chunk = bars[start : start + core]
    if not chunk:
        return 0.0
    return float(np.mean([float(bar.get("confidence") or 0.0) for bar in chunk]))


def _untransposed(chroma: np.ndarray, start: int, other: int, core: int) -> bool:
    """A mesma altura. Um deslocamento melhor é modulação, e a cifra absoluta não serve."""
    left = chroma[:, start : start + core]
    right = chroma[:, other : other + core]
    base = float(np.mean(np.sum(left * right, axis=0)))
    best_other = -1.0
    for shift in range(1, 12):
        score = float(np.mean(np.sum(left * np.roll(right, shift, axis=0), axis=0)))
        if score > best_other:
            best_other = score
    return base >= best_other + REPEAT_SHIFT_MARGIN


def _core_matches(gram: np.ndarray, start: int, other: int, core: int) -> np.ndarray | None:
    sims = np.array([gram[start + slot, other + slot] for slot in range(core)], dtype=float)
    matched = int(np.sum(sims >= REPEAT_BAR_SIM))
    if float(np.mean(sims)) < REPEAT_CORE_MEAN:
        return None
    if matched < REPEAT_MIN_MATCH_BARS or matched / core < REPEAT_MIN_MATCH_FRACTION:
        return None
    return sims


def _mean_sim(gram: np.ndarray, start: int, other: int, core: int) -> float:
    return float(np.mean(gram[start : start + core, other : other + core].diagonal()))


def _alignment_peak(gram: np.ndarray, start: int, other: int, core: int, count: int) -> bool:
    """Um trecho de verdade piora quando a janela anda um compasso.

    Textura parecida ao longo da música marca quase o mesmo valor em todo
    deslocamento. Isso não é uma frase repetida.
    """
    base = _mean_sim(gram, start, other, core)
    for delta in (-1, 1):
        for left, right in ((start + delta, other), (start, other + delta)):
            if min(left, right) < 0 or left + core > count or right + core > count or left == right:
                continue
            if _mean_sim(gram, left, right, core) > base - REPEAT_ALIGN_MARGIN:
                return False
    return True


def _overlaps(start: int, other: int, length: int) -> bool:
    return not (start + length <= other or other + length <= start)


def _hits_claimed(claimed: np.ndarray, start: int, length: int) -> bool:
    return bool(np.any(claimed[start : start + length]))


def _intervals(suffix: str) -> set[int]:
    if suffix in BTC_INTERVALS:
        return set(BTC_INTERVALS[suffix])
    for known in sorted((key for key in BTC_INTERVALS if key), key=len, reverse=True):
        if suffix.startswith(known):
            return set(BTC_INTERVALS[known])
    return set(BTC_INTERVALS[""])


def _chord_profile(chords: list[str]) -> np.ndarray:
    profile = np.zeros(12, dtype=float)
    for chord in chords:
        info = parse_chord_symbol(chord)
        if info["root"] is None:
            continue
        tones = {(info["root"] + interval) % 12 for interval in _intervals(info["suffix"])}
        if info["bass"] is not None:
            tones.add(info["bass"])
        for tone in tones:
            profile[tone] += 1.0
    return profile


def _chord_fit(bar: dict, column: np.ndarray) -> float:
    """Quanto o áudio deste compasso se concentra nas notas da cifra.

    A conta tira a fração de notas que o acorde ocupa. Um símbolo com mais
    notas não ganha só porque cobre um croma espalhado.
    """
    profile = _chord_profile(list(bar.get("chords") or ["N"]))
    audio = np.maximum(np.asarray(column, dtype=float), 0.0)
    total = float(audio.sum())
    mask = profile > 0
    if total < 1e-8 or not bool(np.any(mask)):
        return 0.0
    return float(audio[mask].sum() / total - float(mask.mean()))


def _core_fit(bars: list[dict], chroma: np.ndarray, start: int, core: int) -> float:
    if core <= 0:
        return 0.0
    scores = [_chord_fit(bars[start + slot], chroma[:, start + slot]) for slot in range(core)]
    return float(np.mean(scores))


def _explains_better(source: dict, target: dict, column: np.ndarray) -> bool:
    """O acorde emprestado precisa estar no áudio, e ganhar da cifra local."""
    source_fit = _chord_fit(source, column)
    if source_fit < REPEAT_CHORD_FIT_FLOOR:
        return False
    return source_fit >= _chord_fit(target, column) + REPEAT_CHORD_FIT_MARGIN


def _choose_template(bars: list[dict], chroma: np.ndarray, members: list[int], core: int) -> int:
    """A volta que explica o próprio áudio. Confiança só desempata encaixes próximos."""
    fits = {member: _core_fit(bars, chroma, member, core) for member in members}
    best = max(fits.values())
    candidates = [member for member in members if fits[member] >= best - REPEAT_CHORD_FIT_MARGIN]
    return max(candidates, key=lambda member: _mean_confidence(bars, member, core))


def _copy_slot(source: dict, target: dict) -> bool:
    source_chords = list(source.get("chords") or ["N"])
    target_chords = list(target.get("chords") or ["N"])
    if source_chords == target_chords:
        return False
    if source_chords == ["N"] and target_chords != ["N"]:
        return False
    override_bar(target, source_chords, "repetição do trecho")
    return True


def _claim_core(claimed: np.ndarray, start: int, other: int, sims: np.ndarray) -> None:
    for slot, similarity in enumerate(sims):
        if similarity < REPEAT_BAR_SIM:
            continue
        claimed[start + slot] = True
        claimed[other + slot] = True


def lock_repeated_phrases(bars: list[dict], bar_chroma: np.ndarray) -> int:
    """Copia o miolo da volta que o áudio confirma para as voltas com o mesmo áudio.

    Devolve quantos compassos trocaram de cifra. Sem um croma por compasso,
    ou com a grade em outro tamanho, não mexe em nada.
    """
    chroma = np.asarray(bar_chroma, dtype=float)
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] != len(bars):
        return 0
    count = len(bars)
    if count < REPEAT_MIN_BARS * 2:
        return 0

    normalized = _normalize_columns(chroma)
    gram = normalized.T @ normalized
    claimed = np.zeros(count, dtype=bool)
    copied = 0
    longest = min(REPEAT_MAX_BARS, count // 2)

    for length in range(longest, REPEAT_MIN_BARS - 1, -1):
        tail = _free_tail(length)
        core = length - tail
        if core < REPEAT_MIN_MATCH_BARS:
            continue
        pairs = _pairs_at_length(normalized, gram, claimed, count, length, core)
        if not pairs:
            continue
        copied += _lock_groups(bars, normalized, gram, claimed, pairs, length, core)
    return copied


def _pairs_at_length(
    chroma: np.ndarray,
    gram: np.ndarray,
    claimed: np.ndarray,
    count: int,
    length: int,
    core: int,
) -> list[tuple[int, int]]:
    found = []
    for start in range(0, count - length + 1):
        if _hits_claimed(claimed, start, length):
            continue
        for other in range(start + length, count - length + 1):
            if _hits_claimed(claimed, other, length):
                continue
            if _core_matches(gram, start, other, core) is None:
                continue
            if not _untransposed(chroma, start, other, core):
                continue
            if not _alignment_peak(gram, start, other, core, count):
                continue
            found.append((start, other))
    return found


def _lock_groups(
    bars: list[dict],
    chroma: np.ndarray,
    gram: np.ndarray,
    claimed: np.ndarray,
    pairs: list[tuple[int, int]],
    length: int,
    core: int,
) -> int:
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for start, other in pairs:
        union(start, other)

    groups: dict[int, list[int]] = {}
    for start in parent:
        groups.setdefault(find(start), []).append(start)

    ranked_groups = sorted(
        groups.values(),
        key=lambda members: (
            _core_fit(bars, chroma, _choose_template(bars, chroma, members, core), core),
            max(_mean_confidence(bars, member, core) for member in members),
        ),
        reverse=True,
    )
    copied = 0
    for members in ranked_groups:
        template = _choose_template(bars, chroma, members, core)
        if _hits_claimed(claimed, template, length):
            continue
        accepted = [template]
        for other in sorted(
            members, key=lambda member: _mean_confidence(bars, member, core), reverse=True
        ):
            if other == template or any(_overlaps(other, taken, length) for taken in accepted):
                continue
            if _hits_claimed(claimed, other, length):
                continue
            sims = _core_matches(gram, template, other, core)
            if sims is None or not _untransposed(chroma, template, other, core):
                continue
            for slot, similarity in enumerate(sims):
                if similarity < REPEAT_BAR_SIM:
                    continue
                source = bars[template + slot]
                target = bars[other + slot]
                if not _explains_better(source, target, chroma[:, other + slot]):
                    continue
                if _copy_slot(source, target):
                    copied += 1
            _claim_core(claimed, template, other, sims)
            accepted.append(other)
    return copied
