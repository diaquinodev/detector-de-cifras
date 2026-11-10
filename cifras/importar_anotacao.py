"""Converte marcações exportadas do Sonic Visualiser no gabarito do avaliar.py.

Entradas (camadas "Time Instants" exportadas em CSV):

- pulsos: um instante por tempo; rótulo opcional com a posição no compasso
  ("1", "2"... ou "compasso.tempo", como "3.1");
- acordes: um instante em cada troca, com o acorde no rótulo ("Am7", "G/B", "N").

Saída em cifras/anotacoes/<faixa>.beats e <faixa>.lab.

    python cifras/importar_anotacao.py "Get You" --pulsos pulsos.csv --acordes acordes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from avaliar import to_harte

HERE = Path(__file__).resolve().parent
OUT = HERE / "saida"
NOTES = HERE / "anotacoes"
GAP_TOO_LONG = 1.6
GAP_TOO_SHORT = 0.6
ONSET_RADIUS = 0.05


def plain(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def find_track(name: str) -> Path:
    exact = OUT / f"{name}.json"
    if exact.is_file():
        return exact
    matches = [path for path in OUT.glob("*.json") if plain(name) in plain(path.stem)]
    if len(matches) != 1:
        options = "\n  ".join(path.stem for path in sorted(OUT.glob("*.json")))
        found = "Nenhuma faixa" if not matches else f"{len(matches)} faixas"
        raise SystemExit(f'{found} em saida/ com "{name}" no nome. Faixas analisadas:\n  {options}')
    return matches[0]


def read_instants(path: Path) -> list[tuple[int, float, str]]:
    text = path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if "\t" in text and "," not in text else ","
    rows = []
    for number, row in enumerate(csv.reader(text.splitlines(), delimiter=delimiter), start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if not cells:
            continue
        try:
            moment = float(cells[0])
        except ValueError:
            if number == 1:
                continue
            raise SystemExit(f'{path.name}, linha {number}: o tempo "{cells[0]}" não é um número.')
        rows.append((number, moment, cells[-1] if len(cells) > 1 else ""))
    if not rows:
        raise SystemExit(f"{path.name} não tem nenhum instante.")
    for (line_a, a, _), (line_b, b, _) in zip(rows, rows[1:]):
        if b <= a:
            raise SystemExit(
                f"{path.name}, linhas {line_a} e {line_b}: os tempos {a:.3f} e {b:.3f} não estão em ordem crescente."
            )
    return rows


def snap_to_onsets(audio: Path, beats: np.ndarray) -> tuple[np.ndarray, int]:
    import librosa

    samples, rate = librosa.load(str(audio), sr=22050, mono=True)
    onsets = librosa.onset.onset_detect(y=samples, sr=rate, units="time", backtrack=False)
    if not onsets.size:
        return beats, 0
    nearest = onsets[np.abs(onsets[None, :] - beats[:, None]).argmin(axis=1)]
    close = np.abs(nearest - beats) <= ONSET_RADIUS
    return np.where(close, nearest, beats), int(close.sum())


def position_of(label: str) -> int | None:
    head = label.split(".")[-1] if "." in label else label
    return int(head) if head.isdigit() else None


def convert_beats(
    rows, duration: float, beats_per_bar: int | None, audio: Path | None
) -> tuple[str, list[str]]:
    notes = []
    beats = np.asarray([moment for _, moment, _ in rows])
    late = [line for line, moment, _ in rows if moment > duration + 0.5]
    if late:
        raise SystemExit(f"Pulsos depois do fim da faixa ({duration:.1f} s), linhas {late[:5]}.")
    if audio is not None:
        beats, moved = snap_to_onsets(audio, beats)
        notes.append(
            f"{moved} de {beats.size} pulsos foram encaixados no ataque mais próximo (até {ONSET_RADIUS * 1000:.0f} ms)."
        )
    gaps = np.diff(beats)
    median = float(np.median(gaps))
    notes.append(f"{beats.size} pulsos, {60 / median:.1f} BPM pela mediana dos intervalos.")
    for index, gap in enumerate(gaps):
        if gap > GAP_TOO_LONG * median:
            notes.append(
                f"AVISO: falta pulso entre {beats[index]:.2f} s e {beats[index + 1]:.2f} s (linha {rows[index][0]})."
            )
        elif gap < GAP_TOO_SHORT * median:
            notes.append(
                f"AVISO: pulso duplo em {beats[index + 1]:.2f} s (linha {rows[index + 1][0]})."
            )
    positions = [position_of(label) for _, _, label in rows]
    if all(position is not None for position in positions):
        source = "pelos rótulos"
    elif beats_per_bar:
        positions = [index % beats_per_bar + 1 for index in range(beats.size)]
        source = f"contando {beats_per_bar} a partir do primeiro pulso"
    else:
        positions = [None] * beats.size
        source = None
    if source:
        notes.append(f"Tempo forte {source}: {positions.count(1)} compassos.")
    else:
        notes.append(
            "Sem posição no compasso: o avaliar.py não vai medir o tempo forte. Use --tempos-por-compasso."
        )
    lines = [
        f"{moment:.3f}" + (f" {position}" if position else "")
        for moment, position in zip(beats, positions)
    ]
    return "\n".join(lines) + "\n", notes


def convert_chords(rows, duration: float) -> tuple[str, list[str]]:
    errors, entries = [], []
    for line, moment, label in rows:
        if not label:
            errors.append(f"linha {line} ({moment:.2f} s): instante sem acorde no rótulo")
            continue
        harte = label if ":" in label else to_harte(label)
        if harte == "X":
            errors.append(f'linha {line} ({moment:.2f} s): não reconheço o acorde "{label}"')
            continue
        entries.append((moment, harte))
    if errors:
        raise SystemExit(
            "Corrija o arquivo de acordes:\n  "
            + "\n  ".join(errors)
            + "\nFormas aceitas: C, Cm, C7, Cmaj7, Cm7, Cdim, Cdim7, Cm7b5, Caug, C6, Cm6, Csus2, Csus4, C7sus4,"
            " C9, Cmaj9, Cm9, Cadd9, C5, C/E, N (sem acorde) ou Harte (C:min7)."
        )
    try:
        import mir_eval

        for moment, harte in entries:
            mir_eval.chord.encode(harte)
    except ImportError:
        pass
    except Exception as error:
        raise SystemExit(f"Acorde inválido em {moment:.2f} s: {error}")
    if entries[0][0] > 0.05:
        entries.insert(0, (0.0, "N"))
    merged = [entries[0]]
    for moment, harte in entries[1:]:
        if harte != merged[-1][1]:
            merged.append((moment, harte))
    bounds = [moment for moment, _ in merged[1:]] + [duration]
    lines = [
        f"{start:.3f} {end:.3f} {harte}"
        for (start, harte), end in zip(merged, bounds)
        if end > start
    ]
    notes = [
        f"{len(lines)} trechos de acorde, {len(set(h for _, h in merged))} acordes diferentes."
    ]
    return "\n".join(lines) + "\n", notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o gabarito do avaliar.py a partir do Sonic Visualiser."
    )
    parser.add_argument("faixa", help="nome (ou parte do nome) da faixa já analisada em saida/")
    parser.add_argument("--pulsos", type=Path)
    parser.add_argument("--acordes", type=Path)
    parser.add_argument(
        "--tempos-por-compasso",
        type=int,
        help="quando os pulsos não têm rótulo; o primeiro pulso é o tempo 1",
    )
    parser.add_argument(
        "--audio", type=Path, help="encaixa cada pulso no ataque mais próximo, até 50 ms"
    )
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if not args.pulsos and not args.acordes:
        raise SystemExit("Passe --pulsos, --acordes ou os dois.")
    track = find_track(args.faixa)
    duration = float(json.loads(track.read_text(encoding="utf-8"))["duracao_s"])
    NOTES.mkdir(exist_ok=True)
    print(f"Faixa: {track.stem} ({duration:.1f} s)")
    if args.pulsos:
        text, notes = convert_beats(
            read_instants(args.pulsos), duration, args.tempos_por_compasso, args.audio
        )
        (NOTES / f"{track.stem}.beats").write_text(text, encoding="utf-8")
        print("\n".join(notes))
    if args.acordes:
        text, notes = convert_chords(read_instants(args.acordes), duration)
        (NOTES / f"{track.stem}.lab").write_text(text, encoding="utf-8")
        print("\n".join(notes))
    print(f"Gravado em {NOTES}")


if __name__ == "__main__":
    main()
