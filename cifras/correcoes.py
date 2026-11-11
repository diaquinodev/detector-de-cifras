"""Correções feitas na página, guardadas à parte da cifra do detector.

A cifra medida fica em ``saida/<faixa>.json`` e é reescrita a cada medição. O que a
pessoa corrige fica em ``correcoes/<faixa>.json`` e sobrevive a uma medição nova.

Por enquanto só há barras de repetição: ``{"inicio": 5, "fim": 8, "vezes": 4}`` diz
que os compassos 5 a 8 são tocados 4 vezes seguidas. A folha mostra a primeira volta
entre barras e esconde os compassos das outras voltas (9 a 20, no exemplo).
"""

from __future__ import annotations

import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "correcoes"
MAX_TIMES = 64


def _path(stem: str, folder: Path | None = None) -> Path:
    return (folder or FOLDER) / f"{stem}.json"


def read(stem: str, folder: Path | None = None) -> dict:
    """As correções de uma faixa; vazio quando não há arquivo ou ele não se lê."""
    path = _path(stem, folder)
    if not path.is_file():
        return {"repeticoes": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"repeticoes": []}
    repeats = data.get("repeticoes") if isinstance(data, dict) else None
    return {"repeticoes": [item for item in repeats or [] if _well_formed(item)]}


def write(stem: str, data: dict, folder: Path | None = None) -> None:
    path = _path(stem, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _well_formed(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    values = [item.get("inicio"), item.get("fim"), item.get("vezes")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    first, last, times = values
    return 1 <= first <= last and 2 <= times <= MAX_TIMES


def covered(repeat: dict) -> tuple[int, int]:
    """Primeiro e último compasso que a repetição ocupa, contando todas as voltas."""
    length = repeat["fim"] - repeat["inicio"] + 1
    return repeat["inicio"], repeat["inicio"] + length * repeat["vezes"] - 1


def valid_repeats(data: dict, bar_count: int) -> list[dict]:
    """As repetições que cabem na cifra atual, em ordem e sem se cruzar.

    Uma medição nova pode mudar o número de compassos; a repetição que não cabe mais
    fica guardada no arquivo, mas não é mostrada.
    """
    shown: list[dict] = []
    for repeat in sorted(data.get("repeticoes") or [], key=lambda item: item["inicio"]):
        first, last = covered(repeat)
        if last > bar_count:
            continue
        if shown and first <= covered(shown[-1])[1]:
            continue
        shown.append(repeat)
    return shown


def add_repeat(data: dict, bar_count: int, first: int, last: int, times: int) -> None:
    """Põe uma barra de repetição nos compassos ``first`` a ``last``, tocados ``times`` vezes."""
    repeat = {"inicio": first, "fim": last, "vezes": times}
    if not _well_formed(repeat):
        raise ValueError("Marque do primeiro ao último compasso e repita de 2 a 64 vezes.")
    start, end = covered(repeat)
    if end > bar_count:
        length = last - first + 1
        fits = (bar_count - first + 1) // length
        raise ValueError(
            f"A música acaba antes: esse trecho cabe no máximo {fits} vez(es) a partir dali."
        )
    for other in data.get("repeticoes") or []:
        other_start, other_end = covered(other)
        if start <= other_end and other_start <= end:
            raise ValueError(
                f"Já existe uma repetição nos compassos {other_start} a {other_end}. Tire-a antes."
            )
    data.setdefault("repeticoes", []).append(repeat)
    data["repeticoes"].sort(key=lambda item: item["inicio"])


def remove_repeat(data: dict, first: int) -> None:
    """Tira a repetição que começa no compasso ``first``."""
    kept = [item for item in data.get("repeticoes") or [] if item["inicio"] != first]
    if len(kept) == len(data.get("repeticoes") or []):
        raise ValueError("Não há repetição começando nesse compasso.")
    data["repeticoes"] = kept
