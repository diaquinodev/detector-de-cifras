"""Testes das correções guardadas à parte (nenhuma música real)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

import correcoes  # noqa: E402


class RepeatBarsTest(unittest.TestCase):
    def test_a_repeat_covers_every_lap(self):
        self.assertEqual(correcoes.covered({"inicio": 5, "fim": 8, "vezes": 4}), (5, 20))

    def test_adding_and_removing(self):
        data = {"repeticoes": []}
        correcoes.add_repeat(data, 40, 5, 8, 4)
        correcoes.add_repeat(data, 40, 1, 2, 2)
        self.assertEqual([item["inicio"] for item in data["repeticoes"]], [1, 5])
        correcoes.remove_repeat(data, 5)
        self.assertEqual(data["repeticoes"], [{"inicio": 1, "fim": 2, "vezes": 2}])
        with self.assertRaises(ValueError):
            correcoes.remove_repeat(data, 5)

    def test_a_repeat_must_fit_in_the_song(self):
        data = {"repeticoes": []}
        with self.assertRaises(ValueError) as caught:
            correcoes.add_repeat(data, 18, 5, 8, 4)
        self.assertIn("no máximo 3", str(caught.exception))
        self.assertEqual(data["repeticoes"], [])

    def test_repeats_cannot_overlap_counting_the_hidden_laps(self):
        data = {"repeticoes": []}
        correcoes.add_repeat(data, 40, 5, 8, 4)
        with self.assertRaises(ValueError):
            correcoes.add_repeat(data, 40, 17, 18, 2)
        correcoes.add_repeat(data, 40, 21, 22, 2)

    def test_bad_values_are_refused(self):
        for first, last, times in ((0, 4, 2), (5, 4, 2), (1, 4, 1), (1, 4, 65), (1, 4, True)):
            with self.assertRaises(ValueError):
                correcoes.add_repeat({"repeticoes": []}, 40, first, last, times)

    def test_a_repeat_that_no_longer_fits_is_kept_but_not_shown(self):
        data = {
            "repeticoes": [{"inicio": 5, "fim": 8, "vezes": 4}, {"inicio": 1, "fim": 2, "vezes": 2}]
        }
        self.assertEqual(correcoes.valid_repeats(data, 12), [{"inicio": 1, "fim": 2, "vezes": 2}])
        self.assertEqual(len(data["repeticoes"]), 2)

    def test_reading_and_writing_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            where = Path(folder)
            self.assertEqual(correcoes.read("faixa", where), {"repeticoes": []})
            correcoes.write("faixa", {"repeticoes": [{"inicio": 1, "fim": 4, "vezes": 2}]}, where)
            self.assertEqual(correcoes.read("faixa", where)["repeticoes"][0]["vezes"], 2)
            (where / "quebrado.json").write_text("{", encoding="utf-8")
            self.assertEqual(correcoes.read("quebrado", where), {"repeticoes": []})
            (where / "torto.json").write_text(
                '{"repeticoes": [{"inicio": "a"}, {"inicio": 1, "fim": 2, "vezes": 3}]}',
                encoding="utf-8",
            )
            self.assertEqual(len(correcoes.read("torto", where)["repeticoes"]), 1)


if __name__ == "__main__":
    unittest.main()
