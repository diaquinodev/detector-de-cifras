"""Letras: frases que soam iguais ganham a mesma letra (A A B A).

Só o áudio decide (croma por compasso), nunca a cifra já lida. Duas frases são a mesma
letra quando os compassos de mesma posição se parecem. Depois, letras que sempre andam
juntas viram um bloco só: um verso de 8 compassos são duas frases que nunca se separam.

Uma frase igual a outra, só que mais aguda ou mais grave, é letra nova marcada como
modulação da primeira.
"""

from __future__ import annotations

import string

import numpy as np

SAME_LETTER = 0.9  # semelhança média entre compassos de mesma posição
SILENT = 1e-8
MAX_BLOCK_PHRASES = 4  # um bloco junta no máximo 4 frases (16 compassos)


def _unit(bar_chroma: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(bar_chroma, axis=0, keepdims=True)
    unit = np.zeros_like(bar_chroma, dtype=float)
    heard = norms.ravel() > SILENT
    unit[:, heard] = bar_chroma[:, heard] / norms[:, heard]
    return unit


def phrase_similarity(first: np.ndarray, second: np.ndarray, shift: int = 0) -> float:
    """Semelhança entre duas frases (12 x compassos, já normalizadas), de 0 a 1.

    Frases de tamanhos diferentes não se comparam: uma frase com compasso a mais ou a
    menos é outra coisa. ``shift`` sobe a segunda frase tantos semitons antes de comparar.
    """
    if first.shape != second.shape or first.shape[1] == 0:
        return 0.0
    moved = np.roll(second, shift, axis=0)
    return float(np.mean(np.sum(first * moved, axis=0)))


def _name(index: int, letters: str = string.ascii_uppercase) -> str:
    return letters[index] if index < len(letters) else f"{letters[index % 26]}{index // 26 + 1}"


def _phrase_name(index: int) -> str:
    """Frases usam minúsculas (a, b, c); as maiúsculas ficam para os blocos."""
    return _name(index, string.ascii_lowercase)


def letter_phrases(bar_chroma: np.ndarray, phrases: list[dict]) -> list[dict]:
    """Uma letra para cada frase, na ordem em que aparecem.

    Devolve as frases com ``letra`` (``None`` no silêncio) e, quando a frase é outra já
    ouvida em outro tom, ``modulacao`` com a letra de origem e os semitons.
    """
    unit = _unit(bar_chroma)
    groups: list[list[np.ndarray]] = []
    origins: list[dict | None] = []
    lettered = []
    for phrase in phrases:
        entry = dict(phrase)
        if phrase.get("excecao") == "silêncio":
            entry["letra"] = None
            lettered.append(entry)
            continue
        begin = int(phrase["compasso"]) - 1
        block = unit[:, begin : begin + int(phrase["compassos"])]
        scores = [
            float(np.mean([phrase_similarity(member, block) for member in members]))
            for members in groups
        ]
        best = int(np.argmax(scores)) if scores else -1
        if best >= 0 and scores[best] >= SAME_LETTER:
            groups[best].append(block)
            index = best
        else:
            index = len(groups)
            groups.append([block])
            origins.append(_transposed_from(block, groups[:-1], origins))
        entry["letra"] = _phrase_name(index)
        if origins[index] is not None:
            entry["modulacao"] = origins[index]
        lettered.append(entry)
    return lettered


def _transposed_from(
    block: np.ndarray, groups: list[list[np.ndarray]], origins: list[dict | None]
) -> dict | None:
    """A letra de que esta frase é uma cópia em outro tom, se houver."""
    best: tuple[float, int, int] | None = None
    for index, members in enumerate(groups):
        if origins[index] is not None:
            continue
        for shift in range(1, 12):
            score = float(np.mean([phrase_similarity(block, member, shift) for member in members]))
            if score >= SAME_LETTER and (best is None or score > best[0]):
                best = (score, index, shift)
    if best is None:
        return None
    up = best[2] if best[2] <= 6 else best[2] - 12
    return {"de": _phrase_name(best[1]), "semitons": up}


def _always_together(
    sequence: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Um par de blocos vizinhos que nunca aparecem um sem o outro, e mais de uma vez."""
    for index in range(len(sequence) - 1):
        first, second = sequence[index], sequence[index + 1]
        if first == second or len(first) + len(second) > MAX_BLOCK_PHRASES:
            continue
        pairs = sum(
            sequence[spot] == first and sequence[spot + 1] == second
            for spot in range(len(sequence) - 1)
        )
        if pairs >= 2 and pairs == sequence.count(first) == sequence.count(second):
            return first, second
    return None


def group_blocks(lettered: list[dict]) -> list[dict]:
    """Junta as frases em blocos e dá a letra final de cada bloco.

    Cada bloco traz ``letra``, ``compasso`` (o primeiro, contando de 1), ``compassos``,
    ``frases`` (as letras minúsculas das frases que o formam) e ``modulacao`` quando o
    bloco inteiro é outro já ouvido, em outro tom.
    """
    runs: list[list[dict]] = []
    for phrase in lettered:
        if phrase["letra"] is None:
            runs.append([phrase])
        elif runs and runs[-1][0]["letra"] is not None:
            runs[-1].append(phrase)
        else:
            runs.append([phrase])
    blocks = []
    names: dict[tuple[str, ...], str] = {}
    for run in runs:
        if run[0]["letra"] is None:
            blocks.append(_block(None, run))
            continue
        sequence = [(phrase["letra"],) for phrase in run]
        members = [[phrase] for phrase in run]
        while (pair := _always_together(sequence)) is not None:
            first, second = pair
            merged_sequence, merged_members, spot = [], [], 0
            while spot < len(sequence):
                if spot + 1 < len(sequence) and (sequence[spot], sequence[spot + 1]) == pair:
                    merged_sequence.append(first + second)
                    merged_members.append(members[spot] + members[spot + 1])
                    spot += 2
                else:
                    merged_sequence.append(sequence[spot])
                    merged_members.append(members[spot])
                    spot += 1
            sequence, members = merged_sequence, merged_members
        for key, group in zip(sequence, members, strict=True):
            if key not in names:
                names[key] = _name(len(names))
            block = _block(names[key], group)
            origin = _block_origin(group, names)
            if origin is not None:
                block["modulacao"] = origin
            blocks.append(block)
    return blocks


def _block_origin(phrases: list[dict], names: dict[tuple[str, ...], str]) -> dict | None:
    """O bloco de que este é uma cópia em outro tom: todas as frases sobem o mesmo tanto."""
    origins = [phrase.get("modulacao") for phrase in phrases]
    if any(origin is None for origin in origins):
        return None
    steps = {origin["semitons"] for origin in origins}
    source = names.get(tuple(origin["de"] for origin in origins))
    if len(steps) != 1 or source is None:
        return None
    return {"de": source, "semitons": steps.pop()}


def _block(letter: str | None, phrases: list[dict]) -> dict:
    return {
        "letra": letter,
        "compasso": int(phrases[0]["compasso"]),
        "compassos": sum(int(phrase["compassos"]) for phrase in phrases),
        "frases": [phrase["letra"] for phrase in phrases],
    }


def find_letters(bar_chroma: np.ndarray, phrases: list[dict]) -> list[dict]:
    """Os blocos da música com as suas letras: a forma (por exemplo ``A A B A``)."""
    return group_blocks(letter_phrases(bar_chroma, phrases))


def form_string(blocks: list[dict]) -> str:
    """A forma numa linha, como ``A A B A``; silêncio aparece como ``-``."""
    return " ".join(block["letra"] or "-" for block in blocks)
