"""Régua de erros: onde a cifra erra nos acordes e onde o mapa é incoerente.

Duas medidas, sem dependência além do numpy:

- ``error_classes``: divide o tempo de um trecho com gabarito em classes de erro
  (sem acorde onde há acorde, quinta acima, relativa, qualidade trocada...). Uma
  média só diz quanto errou; as classes dizem que conserto vale a pena.
- ``form_invariants``: conta, numa cifra salva, o que um mapa coerente não teria:
  seção fora de frase de 4 compassos, seção curta, volta do mesmo ciclo com cifra
  diferente, o mesmo ciclo com dois nomes.

    python cifras/diagnostico.py cifras/saida

As classes de erro do GuitarSet saem pelo ``benchmark_guitarset.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from avaliar import ROOTS, to_harte  # noqa: E402

CLASSES = (
    "certo",
    "sem_acorde_indevido",
    "acorde_onde_nao_ha",
    "relativa",
    "quinta_acima",
    "quinta_abaixo",
    "outra_fundamental",
    "qualidade",
    "fora_da_medida",
)
PHRASE_BARS = 4
MIN_SECTION_BARS = 2
LAP_MATCH = 0.75  # fração das fundamentais que precisa bater para contar como a mesma volta
MINOR_PREFIXES = ("min", "dim", "hdim")
MAJOR_PREFIXES = ("maj", "aug", "7", "9", "11", "13")
EDGE_NAMES = {"introdução", "outro", "silêncio"}


def parse_harte(label: str) -> tuple[int | None, str]:
    """Fundamental (0 a 11) e terça ("maj", "min" ou "?") de um rótulo Harte.

    "N" (sem acorde) e "X" (fora do vocabulário) voltam sem fundamental.
    """
    text = (label or "N").strip()
    if text in {"N", "X"}:
        return None, text
    head = text.split("/")[0]
    name, _, quality = head.partition(":")
    if name not in ROOTS:
        return None, "X"
    shorthand, _, listed = quality.partition("(")
    degrees = {degree.strip() for degree in listed.rstrip(")").split(",")}
    if not quality:
        third = "maj"
    elif not shorthand and "b3" in degrees:
        third = "min"
    elif not shorthand and "3" in degrees:
        third = "maj"
    elif shorthand.startswith(MINOR_PREFIXES):
        third = "min"
    elif shorthand.startswith(MAJOR_PREFIXES):
        third = "maj"
    else:
        third = "?"  # sus, quinta sozinha ou lista de graus sem terça
    return ROOTS[name], third


def classify(reference: str, estimate: str) -> str:
    """A classe de um instante, dado o rótulo do gabarito e o da cifra."""
    ref_root, ref_third = parse_harte(reference)
    est_root, est_third = parse_harte(estimate)
    if ref_third == "X" or est_third == "X":
        return "fora_da_medida"
    if ref_root is None:
        return "certo" if est_root is None else "acorde_onde_nao_ha"
    if est_root is None:
        return "sem_acorde_indevido"
    if est_root == ref_root:
        different = {ref_third, est_third} == {"maj", "min"}
        return "qualidade" if different else "certo"
    step = (est_root - ref_root) % 12
    if (ref_third, est_third, step) in {("maj", "min", 9), ("min", "maj", 3)}:
        return "relativa"
    if step == 7:
        return "quinta_acima"
    if step == 5:
        return "quinta_abaixo"
    return "outra_fundamental"


def _without_overlap(
    intervals: list[tuple[float, float]], labels: list[str]
) -> tuple[list[tuple[float, float]], list[str]]:
    """Põe os trechos em ordem e corta cada um onde o seguinte começa.

    Sem isso, dois trechos sobrepostos dariam resultado diferente conforme a ordem do arquivo.
    """
    ordered = sorted(zip(intervals, labels, strict=True))
    flat, names = [], []
    for index, ((start, end), label) in enumerate(ordered):
        if index + 1 < len(ordered):
            end = min(end, ordered[index + 1][0][0])
        if end > start:
            flat.append((start, end))
            names.append(label)
    return flat, names


def _label_at(intervals: list[tuple[float, float]], labels: list[str], moment: float) -> str:
    for (start, end), label in zip(intervals, labels, strict=True):
        if start <= moment < end:
            return label
    return "N"


def error_classes(
    ref_intervals: list[tuple[float, float]],
    ref_labels: list[str],
    est_intervals: list[tuple[float, float]],
    est_labels: list[str],
) -> dict[str, float]:
    """Segundos de cada classe, dentro da janela do gabarito.

    Onde a cifra não diz nada, vale "N". A soma das classes é a duração do gabarito.
    """
    seconds = dict.fromkeys(CLASSES, 0.0)
    ref_intervals, ref_labels = _without_overlap(ref_intervals, ref_labels)
    est_intervals, est_labels = _without_overlap(est_intervals, est_labels)
    if not ref_intervals:
        return seconds
    first = min(start for start, _ in ref_intervals)
    last = max(end for _, end in ref_intervals)
    edges = {first, last}
    for start, end in (*ref_intervals, *est_intervals):
        edges.update(moment for moment in (start, end) if first < moment < last)
    cuts = sorted(edges)
    for start, end in zip(cuts, cuts[1:], strict=False):
        middle = (start + end) / 2.0
        reference = _label_at(ref_intervals, ref_labels, middle)
        estimate = _label_at(est_intervals, est_labels, middle)
        seconds[classify(reference, estimate)] += end - start
    return {name: round(value, 4) for name, value in seconds.items()}


def read_lab(path: Path) -> tuple[list[tuple[float, float]], list[str]]:
    """Lê ``início fim acorde`` por linha (o formato das anotações do avaliador)."""
    intervals, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            intervals.append((float(parts[0]), float(parts[1])))
            labels.append(parts[2])
    return intervals, labels


def class_shares(entries: list[dict[str, float]]) -> dict[str, float]:
    """Fração do tempo total em cada classe, somando vários trechos."""
    total = sum(sum(entry.values()) for entry in entries)
    if total <= 0:
        return dict.fromkeys(CLASSES, 0.0)
    return {
        name: round(sum(entry.get(name, 0.0) for entry in entries) / total, 4) for name in CLASSES
    }


def _root(symbol: str) -> int | None:
    return parse_harte(to_harte(symbol))[0]


def _bar_text(bar: dict) -> str:
    return " ".join(bar.get("chords") or ["N"])


def _is_silent(bar: dict) -> bool:
    """Compasso sem acorde nenhum: "N", "N.C." ou vazio, uma ou mais vezes."""
    return all(to_harte(str(chord)) == "N" for chord in bar.get("chords") or ["N"])


def _laps(bars: list[dict], patterns: list[dict]) -> tuple[int, list[dict]]:
    """Voltas de cada ciclo do catálogo e as que não saíram com a cifra do ciclo.

    Uma janela é uma volta quando a fundamental do primeiro acorde de cada compasso
    bate com a do ciclo em pelo menos ``LAP_MATCH`` das posições.
    """
    cycles = []
    for pattern in patterns:
        chords = [str(chord) for chord in pattern.get("acordes") or []]
        roots = [_root(chord.split()[0]) if chord.split() else None for chord in chords]
        known = {root for root in roots if root is not None}
        if int(pattern.get("repeticoes") or 0) >= 2 and len(known) >= 2:
            cycles.append((str(pattern.get("id")), chords, roots))
    cycles.sort(key=lambda cycle: -len(cycle[1]))
    bar_roots = [_root((bar.get("chords") or ["N"])[0]) for bar in bars]
    laps, different, index = 0, [], 0
    while index < len(bars):
        for name, chords, roots in cycles:
            window = bar_roots[index : index + len(roots)]
            if len(window) < len(roots):
                continue
            hits = sum(a == b and a is not None for a, b in zip(window, roots, strict=True))
            if hits < math.ceil(LAP_MATCH * len(roots)):
                continue
            laps += 1
            written = [_bar_text(bar) for bar in bars[index : index + len(roots)]]
            if written != chords:
                different.append(
                    {
                        "ciclo": name,
                        "compasso": bars[index].get("bar", index + 1),
                        "esperado": chords,
                        "escrito": written,
                    }
                )
            index += len(roots)
            break
        else:
            index += 1
    return laps, different


def form_invariants(document: dict) -> dict:
    """O que um mapa coerente não teria, contado numa cifra salva."""
    bars = document.get("compassos") or []
    sections = document.get("secoes") or []
    sizes = [(str(section.get("name")), len(section.get("bars") or [])) for section in sections]
    # Só introdução, outro e silêncio nas pontas ficam de fora: anacruse e final cortado
    # são comuns ali. Um verso de 1 compasso abrindo a música continua sendo defeito.
    last = len(sizes) - 1
    inner = [
        (name, size)
        for index, (name, size) in enumerate(sizes)
        if not (index in {0, last} and name in EDGE_NAMES)
    ]
    sounding = [index for index, bar in enumerate(bars) if not _is_silent(bar)]
    silent_inside = (
        sum(_is_silent(bar) for bar in bars[sounding[0] : sounding[-1] + 1]) if sounding else 0
    )
    names_by_pattern: dict[str, set[str]] = defaultdict(set)
    for section in sections:
        pattern, name = str(section.get("padrao") or ""), str(section.get("name"))
        if pattern and " + " not in pattern and name not in EDGE_NAMES:
            names_by_pattern[pattern].add(name)
    laps, different = _laps(bars, document.get("padroes_harmonicos") or [])
    return {
        "compassos": len(bars),
        "secoes": len(sections),
        "secoes_fora_de_frase": [
            {"nome": name, "compassos": size} for name, size in inner if size % PHRASE_BARS
        ],
        "secoes_curtas": [
            {"nome": name, "compassos": size} for name, size in inner if size < MIN_SECTION_BARS
        ],
        "compassos_sem_acorde_no_meio": int(silent_inside),
        "voltas": laps,
        "voltas_com_cifra_diferente": different,
        "ciclos_com_dois_nomes": {
            pattern: sorted(names)
            for pattern, names in sorted(names_by_pattern.items())
            if len(names) > 1
        },
    }


def form_summary(reports: dict[str, dict]) -> dict:
    """Totais das invariantes de várias cifras."""
    sections = sum(report["secoes"] for report in reports.values())
    laps = sum(report["voltas"] for report in reports.values())
    return {
        "musicas": len(reports),
        "secoes": sections,
        "secoes_fora_de_frase": sum(len(r["secoes_fora_de_frase"]) for r in reports.values()),
        "secoes_curtas": sum(len(r["secoes_curtas"]) for r in reports.values()),
        "compassos_sem_acorde_no_meio": sum(
            r["compassos_sem_acorde_no_meio"] for r in reports.values()
        ),
        "voltas": laps,
        "voltas_com_cifra_diferente": sum(
            len(r["voltas_com_cifra_diferente"]) for r in reports.values()
        ),
        "ciclos_com_dois_nomes": sum(len(r["ciclos_com_dois_nomes"]) for r in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Invariantes de forma das cifras salvas.")
    parser.add_argument("pasta", type=Path, help="pasta com os .json gerados pelo analyze.py")
    parser.add_argument("--detalhe", action="store_true", help="mostra cada caso, não só a conta")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    reports = {}
    for path in sorted(args.pasta.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "compassos" in document and "secoes" in document:
            reports[path.stem] = form_invariants(document)
    if not reports:
        raise SystemExit(f"Nenhuma cifra em {args.pasta}.")
    output = {"total": form_summary(reports)}
    if args.detalhe:
        output["musicas"] = reports
    else:
        output["musicas"] = {name: form_summary({name: report}) for name, report in reports.items()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
