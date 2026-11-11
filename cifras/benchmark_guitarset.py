"""Mede acordes e pulsos no GuitarSet (Xi et al., ISMIR 2018; CC BY 4.0).

Usa os 180 trechos de acompanhamento ("_comp") gravados no microfone. A
referência de acordes é a cifra da partitura (primeira anotação "chord" do
JAMS); a referência de pulso é o metrônomo da gravação.

O BTC não treinou no GuitarSet, então a medida de acordes é limpa. O Beat This!
treinou nos trechos "_comp" do GuitarSet: a medida de pulso só confere o
pós-processamento, não a rede.

    python cifras/benchmark_guitarset.py dados/guitarset
    python cifras/benchmark_guitarset.py dados/guitarset --resumo

Cada trecho vai para resultados.jsonl assim que termina; rodar de novo continua
de onde parou. Para medir de novo depois de mudar o código, use outro arquivo:

    python cifras/benchmark_guitarset.py dados/guitarset --resultados fase0.jsonl

O resumo traz também as classes de erro (ver diagnostico.py): que parte do tempo
saiu sem acorde, com a relativa, uma quinta acima...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHORD_KEYS = ("root", "majmin", "sevenths", "mirex", "sobressegmentacao", "subsegmentacao")
BEAT_KEYS = ("f_measure", "cmlt", "amlt", "downbeat_f_measure")


def references(jams_path: Path, folder: Path) -> tuple[Path, Path, float]:
    jam = json.loads(jams_path.read_text(encoding="utf-8"))
    chords = next(ann for ann in jam["annotations"] if ann["namespace"] == "chord")
    beats = next(ann for ann in jam["annotations"] if ann["namespace"] == "beat_position")
    stem = jams_path.stem
    lab = folder / f"{stem}.lab"
    lab.write_text(
        "".join(
            f"{o['time']:.4f} {o['time'] + o['duration']:.4f} {o['value']}\n"
            for o in chords["data"]
        ),
        encoding="utf-8",
    )
    beat_file = folder / f"{stem}.beats"
    beat_file.write_text(
        "".join(f"{o['time']:.4f} {o['value']['position']}\n" for o in beats["data"]),
        encoding="utf-8",
    )
    return lab, beat_file, float(jam["file_metadata"]["duration"])


def run(root: Path, results: Path) -> None:
    import mir_eval
    from analyze import analyze
    from avaliar import chord_intervals, evaluate_beats, evaluate_chords
    from diagnostico import error_classes, read_lab

    refs = root / "referencias"
    refs.mkdir(exist_ok=True)
    # Resultado antigo sem as classes de erro é medido de novo: a régua não vale pela metade.
    done = {
        entry["trecho"]
        for entry in latest_entries(results)
        if "erro" not in entry and "classes_s" in entry
    }
    pending = [path for path in sorted(root.glob("*_comp.jams")) if path.stem not in done]
    print(f"{len(done)} feitos, {len(pending)} pendentes", flush=True)
    for number, jams_path in enumerate(pending, start=1):
        audio = root / f"{jams_path.stem}_mic.wav"
        entry = {
            "trecho": jams_path.stem,
            "estilo": jams_path.stem.split("_")[1].split("-")[0].rstrip("0123456789"),
        }
        started = time.perf_counter()
        try:
            lab, beat_file, duration = references(jams_path, refs)
            data = analyze(audio)
            entry["duracao_s"] = duration
            bars = data.get("compassos") or []
            entry["acordes"] = evaluate_chords(mir_eval, lab, bars)
            estimated, labels = chord_intervals(bars, model_only=False)
            entry["classes_s"] = error_classes(
                *read_lab(lab), [tuple(pair) for pair in estimated.tolist()], labels
            )
            entry["ritmo"] = evaluate_beats(mir_eval, beat_file, data)
            entry["bpm"] = data["pulsacao"]["bpm"]
            entry["compasso"] = data["compasso"]["tempos_por_compasso"]
            entry["ajustados_por_regra"] = data.get("compassos_ajustados_por_regra")
        except Exception as error:
            entry["erro"] = f"{type(error).__name__}: {error}"
            entry["traceback"] = traceback.format_exc(limit=4)
        entry["segundos"] = round(time.perf_counter() - started, 1)
        with results.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        status = entry.get("erro") or f"majmin {entry['acordes']['final']['majmin']:.3f}"
        print(
            f"[{number}/{len(pending)}] {jams_path.stem}: {status} ({entry['segundos']} s)",
            flush=True,
        )


def latest_entries(results: Path) -> list[dict]:
    if not results.is_file():
        return []
    latest = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            latest[entry["trecho"]] = entry
    return list(latest.values())


def summary(results: Path) -> dict:
    from diagnostico import class_shares

    entries = latest_entries(results)
    good = [entry for entry in entries if "erro" not in entry]

    def chords(group: list[dict], variant: str) -> dict:
        total = sum(entry["duracao_s"] for entry in group)
        return {
            key: round(
                sum(entry["acordes"][variant][key] * entry["duracao_s"] for entry in group) / total,
                4,
            )
            for key in CHORD_KEYS
        }

    def beats(group: list[dict]) -> dict:
        out = {}
        for key in BEAT_KEYS:
            values = [entry["ritmo"][key] for entry in group if key in entry["ritmo"]]
            out[key] = round(sum(values) / len(values), 4) if values else None
        out["tempo_acc1"] = round(
            sum(bool(entry["ritmo"].get("tempo_acc1")) for entry in group) / len(group), 4
        )
        out["tempo_acc2"] = round(
            sum(bool(entry["ritmo"].get("tempo_acc2")) for entry in group) / len(group), 4
        )
        out["compasso_4"] = round(sum(entry["compasso"] == 4 for entry in group) / len(group), 4)
        return out

    def classes(group: list[dict]) -> dict | None:
        measured = [entry["classes_s"] for entry in group if "classes_s" in entry]
        return class_shares(measured) if measured else None

    styles = defaultdict(list)
    for entry in good:
        styles[entry["estilo"]].append(entry)
    return {
        "metodo": {
            "conjunto": "GuitarSet 1.1.0, trechos _comp, áudio mono-mic",
            "referencia_acordes": "primeira anotação chord do JAMS (cifra da partitura)",
            "acordes": "média ponderada pela duração; mir_eval com janela do trecho inteiro",
            "ritmo": "média por trecho; F 70 ms; Beat This! treinou nesses trechos",
        },
        "trechos": len(entries),
        "erros": [entry["trecho"] for entry in entries if "erro" in entry],
        "acordes_final": chords(good, "final") if good else None,
        "acordes_so_modelo": chords(good, "so_modelo") if good else None,
        "classes_de_erro": classes(good),
        "trechos_com_classes": sum("classes_s" in entry for entry in good),
        "ritmo": beats(good) if good else None,
        "por_estilo": {
            style: {
                "n": len(group),
                "majmin_final": chords(group, "final")["majmin"],
                "majmin_so_modelo": chords(group, "so_modelo")["majmin"],
                "classes_de_erro": classes(group),
            }
            for style, group in sorted(styles.items())
        },
        "piores_majmin": sorted(
            (entry["acordes"]["final"]["majmin"], entry["trecho"]) for entry in good
        )[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o pipeline no GuitarSet.")
    parser.add_argument("pasta", type=Path)
    parser.add_argument("--resumo", action="store_true")
    parser.add_argument(
        "--resultados",
        default="resultados.jsonl",
        help="arquivo de resultados dentro da pasta (padrão: resultados.jsonl)",
    )
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.pasta.resolve()
    results = root / Path(args.resultados).name
    if not args.resumo:
        run(root, results)
    report = summary(results)
    resumo = "resumo.json" if results.name == "resultados.jsonl" else f"{results.stem}.resumo.json"
    (root / resumo).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
