"""Testes da régua de erros, com gabarito e cifra escritos à mão (nenhuma música real)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from diagnostico import (  # noqa: E402
    CLASSES,
    class_shares,
    classify,
    error_classes,
    form_invariants,
    form_summary,
    parse_harte,
    read_lab,
)


def bar(number: int, *chords: str) -> dict:
    return {"bar": number, "chords": list(chords)}


def section(name: str, pattern: str, bars: list[dict]) -> dict:
    return {"name": name, "padrao": pattern, "bars": bars}


class ParseHarteTest(unittest.TestCase):
    def test_reads_root_and_third(self):
        cases = {
            "C:maj": (0, "maj"),
            "A:min7": (9, "min"),
            "G:7": (7, "maj"),
            "B:hdim7": (11, "min"),
            "F": (5, "maj"),
            "Bb:maj/3": (10, "maj"),
            "D:sus4(b7)": (2, "?"),
            "C:(1,5,b7)/1": (0, "?"),
            "C:(1,3,5)": (0, "maj"),
            "C:(1,b3,5)": (0, "min"),
            "N": (None, "N"),
            "X": (None, "X"),
            "H:maj": (None, "X"),
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(parse_harte(label), expected)


class ClassifyTest(unittest.TestCase):
    def test_each_class_has_one_plain_example(self):
        cases = [
            ("C:maj", "C:maj", "certo"),
            ("C:maj", "C:7", "certo"),
            ("N", "N", "certo"),
            ("C:maj", "N", "sem_acorde_indevido"),
            ("N", "C:maj", "acorde_onde_nao_ha"),
            ("C:maj", "C:min", "qualidade"),
            ("D:maj", "B:min", "relativa"),
            ("A:min", "C:maj", "relativa"),
            ("A:min", "E:min", "quinta_acima"),
            ("C:maj", "F:maj", "quinta_abaixo"),
            ("C:maj", "D:maj", "outra_fundamental"),
            ("C:maj", "X", "fora_da_medida"),
            ("X", "C:maj", "fora_da_medida"),
        ]
        for reference, estimate, expected in cases:
            with self.subTest(reference=reference, estimate=estimate):
                self.assertEqual(classify(reference, estimate), expected)
                self.assertIn(expected, CLASSES)

    def test_a_sus_chord_on_the_right_root_is_not_a_quality_error(self):
        self.assertEqual(classify("C:maj", "C:sus4"), "certo")

    def test_a_degree_list_with_a_minor_third_is_a_quality_error(self):
        self.assertEqual(classify("C:(1,b3,5)", "C:maj"), "qualidade")

    def test_the_major_a_sixth_above_is_not_the_relative(self):
        # Lá maior no lugar de Dó maior não é a relativa (que seria Lá menor).
        self.assertEqual(classify("C:maj", "A:maj"), "outra_fundamental")


class ErrorClassesTest(unittest.TestCase):
    def test_splits_the_reference_time_by_class(self):
        reference = [(0.0, 4.0), (4.0, 8.0)]
        labels = ["C:maj", "G:maj"]
        estimated = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        estimated_labels = ["C:maj", "A:min", "N"]
        seconds = error_classes(reference, labels, estimated, estimated_labels)
        self.assertEqual(seconds["certo"], 2.0)
        self.assertEqual(seconds["relativa"], 2.0)
        # De 4 a 6 a cifra diz "N"; de 6 a 8 não diz nada, e isso também vale "N".
        self.assertEqual(seconds["sem_acorde_indevido"], 4.0)
        self.assertAlmostEqual(sum(seconds.values()), 8.0)

    def test_ignores_what_the_chart_says_outside_the_reference_window(self):
        seconds = error_classes([(2.0, 4.0)], ["C:maj"], [(0.0, 10.0)], ["C:maj"])
        self.assertEqual(seconds["certo"], 2.0)
        self.assertAlmostEqual(sum(seconds.values()), 2.0)

    def test_overlapping_intervals_do_not_depend_on_the_file_order(self):
        intervals = [(0.0, 3.0), (2.0, 4.0)]
        labels = ["C:maj", "G:maj"]
        estimate = ([(0.0, 4.0)], ["C:maj"])
        straight = error_classes(intervals, labels, *estimate)
        reversed_order = error_classes(intervals[::-1], labels[::-1], *estimate)
        self.assertEqual(straight, reversed_order)
        # O trecho que começa depois vale a partir do início dele: 2 s de Dó, 2 s de Sol.
        self.assertEqual(straight["certo"], 2.0)
        self.assertEqual(straight["quinta_abaixo"], 2.0)
        self.assertAlmostEqual(sum(straight.values()), 4.0)

    def test_an_empty_reference_measures_nothing(self):
        seconds = error_classes([], [], [(0.0, 1.0)], ["C:maj"])
        self.assertEqual(sum(seconds.values()), 0.0)

    def test_shares_add_several_clips_weighted_by_time(self):
        short = error_classes([(0.0, 1.0)], ["C:maj"], [(0.0, 1.0)], ["C:maj"])
        long = error_classes([(0.0, 3.0)], ["C:maj"], [], [])
        shares = class_shares([short, long])
        self.assertEqual(shares["certo"], 0.25)
        self.assertEqual(shares["sem_acorde_indevido"], 0.75)
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertEqual(sum(class_shares([]).values()), 0.0)

    def test_reads_a_lab_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "faixa.lab"
            path.write_text("0.0 2.5 C:maj\n\n2.5 5.0 N\n", encoding="utf-8")
            intervals, labels = read_lab(path)
        self.assertEqual(intervals, [(0.0, 2.5), (2.5, 5.0)])
        self.assertEqual(labels, ["C:maj", "N"])


class FormInvariantsTest(unittest.TestCase):
    def setUp(self):
        chords = ["Am", "G", "F", "C"]
        clean = [bar(index + 1, chords[index % 4]) for index in range(16)]
        self.pattern = {"id": "Ciclo A", "repeticoes": 4, "acordes": chords}
        self.coherent = {
            "compassos": clean,
            "padroes_harmonicos": [self.pattern],
            "secoes": [
                section("introdução", "Ciclo A", clean[0:4]),
                section("verso", "Ciclo A", clean[4:12]),
                section("outro", "Ciclo A", clean[12:16]),
            ],
        }

    def test_a_coherent_chart_has_nothing_to_report(self):
        report = form_invariants(self.coherent)
        self.assertEqual(report["voltas"], 4)
        self.assertEqual(report["secoes_fora_de_frase"], [])
        self.assertEqual(report["secoes_curtas"], [])
        self.assertEqual(report["compassos_sem_acorde_no_meio"], 0)
        self.assertEqual(report["voltas_com_cifra_diferente"], [])
        self.assertEqual(report["ciclos_com_dois_nomes"], {})

    def test_a_lap_written_differently_is_reported_with_both_versions(self):
        bars = self.coherent["compassos"]
        bars[5] = bar(6, "G", "G7", "G")
        bars[6] = bar(7, "G", "F7")
        report = form_invariants(self.coherent)
        self.assertEqual(report["voltas"], 4)
        self.assertEqual(len(report["voltas_com_cifra_diferente"]), 1)
        case = report["voltas_com_cifra_diferente"][0]
        self.assertEqual(case["compasso"], 5)
        self.assertEqual(case["esperado"], ["Am", "G", "F", "C"])
        self.assertEqual(case["escrito"], ["Am", "G G7 G", "G F7", "C"])

    def test_a_window_with_half_the_roots_is_not_a_lap(self):
        bars = self.coherent["compassos"]
        bars[4:8] = [bar(5, "Am"), bar(6, "D"), bar(7, "E"), bar(8, "C")]
        report = form_invariants(self.coherent)
        self.assertEqual(report["voltas"], 3)
        self.assertEqual(report["voltas_com_cifra_diferente"], [])

    def test_sections_off_the_phrase_grid_and_short_sections_are_listed(self):
        bars = self.coherent["compassos"]
        self.coherent["secoes"] = [
            section("introdução", "Ciclo A", bars[0:3]),
            section("verso", "Ciclo A", bars[3:8]),
            section("pré-refrão", "Ciclo B", bars[8:9]),
            section("refrão", "Ciclo A", bars[9:13]),
            section("outro", "Ciclo A", bars[13:16]),
        ]
        report = form_invariants(self.coherent)
        # Introdução e outro nas pontas ficam de fora: anacruse e final cortado são comuns.
        self.assertEqual(
            report["secoes_fora_de_frase"],
            [{"nome": "verso", "compassos": 5}, {"nome": "pré-refrão", "compassos": 1}],
        )
        self.assertEqual(report["secoes_curtas"], [{"nome": "pré-refrão", "compassos": 1}])
        self.assertEqual(report["ciclos_com_dois_nomes"], {"Ciclo A": ["refrão", "verso"]})

    def test_counts_no_chord_bars_only_between_the_first_and_last_chord(self):
        bars = self.coherent["compassos"]
        bars[0] = bar(1, "N")
        bars[9] = bar(10, "N")
        bars[15] = bar(16, "N")
        self.assertEqual(form_invariants(self.coherent)["compassos_sem_acorde_no_meio"], 1)

    def test_silence_written_in_other_ways_is_still_silence(self):
        bars = self.coherent["compassos"]
        bars[0] = bar(1, "N", "N")
        bars[9] = bar(10, "N", "N")
        bars[10] = bar(11, "N.C.")
        self.assertEqual(form_invariants(self.coherent)["compassos_sem_acorde_no_meio"], 2)

    def test_a_short_verse_at_the_edge_is_still_reported(self):
        bars = self.coherent["compassos"]
        self.coherent["secoes"] = [
            section("verso", "Ciclo A", bars[0:1]),
            section("verso", "Ciclo A", bars[1:9]),
            section("refrão", "Ciclo A", bars[9:16]),
        ]
        report = form_invariants(self.coherent)
        self.assertEqual(
            report["secoes_fora_de_frase"],
            [{"nome": "verso", "compassos": 1}, {"nome": "refrão", "compassos": 7}],
        )
        self.assertEqual(report["secoes_curtas"], [{"nome": "verso", "compassos": 1}])

    def test_a_one_chord_vamp_is_not_used_as_a_cycle(self):
        self.coherent["padroes_harmonicos"] = [
            {"id": "Ciclo B", "repeticoes": 5, "acordes": ["A", "A"]}
        ]
        self.assertEqual(form_invariants(self.coherent)["voltas"], 0)

    def test_an_empty_document_does_not_break(self):
        report = form_invariants({})
        self.assertEqual(report["compassos"], 0)
        self.assertEqual(report["voltas"], 0)

    def test_summary_adds_the_songs(self):
        bars = self.coherent["compassos"]
        bars[5] = bar(6, "G", "G7")
        one = form_invariants(self.coherent)
        total = form_summary({"a": one, "b": one})
        self.assertEqual(total["musicas"], 2)
        self.assertEqual(total["voltas"], 8)
        self.assertEqual(total["voltas_com_cifra_diferente"], 2)


if __name__ == "__main__":
    unittest.main()
