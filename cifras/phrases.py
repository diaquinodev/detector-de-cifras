"""Frases: os compassos agrupados de 4 em 4, com as exceções ditas pelo nome.

Música popular conta em frases de 4 compassos, e uma seção começa em início de frase.
Este módulo só olha o áudio (croma por compasso), nunca a cifra já lida.

A grade regular a partir do primeiro compasso com som é o ponto de partida. Uma exceção
(anacruse, compasso extra, frase curta) só entra quando as mudanças fortes da música
passam a cair fora da grade e ficam dentro dela com a exceção. Mudança forte é o ponto em
que os compassos de antes não se parecem com os de depois (novidade).

O tamanho é fixo em 4: frases de 2 e de 8 compassos cabem na mesma grade.
"""

from __future__ import annotations

import numpy as np

PHRASE_BARS = 4
# Uma exceção custa mais que uma mudança forte: ela precisa pôr pelo menos duas na grade.
EXCEPTION_COST = 1.5
MIN_NOVELTY = 0.05  # abaixo disso a música não muda o bastante para servir de evidência
SILENT = 1e-8


def _similarity(bar_chroma: np.ndarray) -> np.ndarray:
    """Semelhança (cosseno) entre cada par de compassos. Forma (compassos, compassos)."""
    norms = np.linalg.norm(bar_chroma, axis=0, keepdims=True)
    unit = np.zeros_like(bar_chroma, dtype=float)
    heard = norms.ravel() > SILENT
    unit[:, heard] = bar_chroma[:, heard] / norms[:, heard]
    return unit.T @ unit


def novelty(bar_chroma: np.ndarray, size: int = PHRASE_BARS) -> np.ndarray:
    """Quanto a música muda em cada fronteira de compasso, de 0 a 1.

    Compara os ``size`` compassos de antes com os de depois. Um ciclo que se repete não
    gera novidade em ponto nenhum; a troca de um ciclo por outro gera um pico. Só os picos
    ficam: a subida e a descida em volta deles viram zero.
    """
    count = int(bar_chroma.shape[1])
    raw = np.zeros(count + 1, dtype=float)
    if count < 2:
        return raw
    gram = _similarity(bar_chroma)
    # Perto das pontas não há ``size`` compassos de cada lado para comparar: fica zero.
    for edge in range(size, count - size + 1):
        before = slice(edge - size, edge)
        after = slice(edge, edge + size)
        inside = (gram[before, before].mean() + gram[after, after].mean()) / 2.0
        raw[edge] = max(0.0, float(inside - gram[before, after].mean()))
    top = float(raw.max())
    if top < MIN_NOVELTY:
        return np.zeros(count + 1, dtype=float)
    peaks = np.zeros(count + 1, dtype=float)
    for edge in range(1, count):
        if raw[edge] > raw[edge - 1] and raw[edge] >= raw[edge + 1]:
            peaks[edge] = raw[edge] / top
    return peaks


def _heard_span(bar_chroma: np.ndarray) -> tuple[int, int]:
    """Do primeiro compasso com som até logo depois do último. Sem som nenhum: vazio."""
    heard = np.flatnonzero(np.linalg.norm(bar_chroma, axis=0) > SILENT)
    count = int(bar_chroma.shape[1])
    return (int(heard[0]), int(heard[-1]) + 1) if heard.size else (count, count)


def _reason(length: int, size: int, first: bool, last: bool) -> str | None:
    if length == size:
        return None
    if length > size:
        return "compasso extra"
    if last:
        return "final incompleto"
    return "anacruse" if first else "frase curta"


def _join_turnarounds(
    cuts: list[tuple[int, int]], size: int, gain: np.ndarray
) -> list[tuple[int, int]]:
    """Um ou dois compassos soltos no meio da música ficam com a frase de antes.

    É o caso comum: a frase ganha um compasso de virada antes de a música seguir. Se a
    música muda de verdade onde o trecho solto começa, ele fica como frase curta.
    """
    joined: list[tuple[int, int]] = []
    for index, (begin, length) in enumerate(cuts):
        loose = 0 < index < len(cuts) - 1 and length <= size // 2 and gain[begin] == 0.0
        if loose and joined and joined[-1][1] == size:
            joined[-1] = (joined[-1][0], size + length)
        else:
            joined.append((begin, length))
    return joined


def find_phrases(bar_chroma: np.ndarray, size: int = PHRASE_BARS) -> list[dict]:
    """As frases da música, em ordem, cobrindo todos os compassos.

    Cada frase traz ``compasso`` (o primeiro, contando de 1), ``compassos`` (quantos) e
    ``excecao`` (``None`` na frase regular; senão o motivo). Silêncio antes e depois da
    música fica à parte; compasso sem som no meio conta como pausa dentro da frase.
    """
    count = int(bar_chroma.shape[1]) if bar_chroma.ndim == 2 else 0
    if count == 0:
        return []
    start, stop = _heard_span(bar_chroma)
    gain = novelty(bar_chroma, size)
    lengths = [size, *range(1, size), size + 1, size + 2]
    best = np.full(stop + 1, -np.inf)
    back = np.zeros(stop + 1, dtype=int)
    best[start] = 0.0
    for end in range(start + 1, stop + 1):
        for length in lengths:
            begin = end - length
            if begin < start or not np.isfinite(best[begin]):
                continue
            free = length == size or (end == stop and length < size)
            score = best[begin] + gain[end] - (0.0 if free else EXCEPTION_COST)
            if score > best[end]:
                best[end] = score
                back[end] = length
    cuts = []
    end = stop
    while end > start:
        cuts.append((end - back[end], int(back[end])))
        end -= back[end]
    cuts.reverse()
    cuts = _join_turnarounds(cuts, size, gain)
    phrases = []
    if start:
        phrases.append({"compasso": 1, "compassos": start, "excecao": "silêncio"})
    for index, (begin, length) in enumerate(cuts):
        phrases.append(
            {
                "compasso": int(begin) + 1,
                "compassos": length,
                "excecao": _reason(length, size, index == 0, index == len(cuts) - 1),
            }
        )
    if start < stop < count:
        phrases.append({"compasso": stop + 1, "compassos": count - stop, "excecao": "silêncio"})
    return phrases


def phrase_starts(phrases: list[dict]) -> list[int]:
    """Índices (de 0) dos compassos em que uma frase começa."""
    return [int(phrase["compasso"]) - 1 for phrase in phrases]
