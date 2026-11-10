"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from importar_anotacao import plain, position_of, read_instants  # noqa: E402


class ImportarTest(unittest.TestCase):
    def test_plain_ignores_punctuation_and_case(self):
        self.assertEqual(plain("Leave the Door Open!"), "leave the door open")

    def test_beat_position_reads_the_number_after_the_dot(self):
        self.assertEqual(position_of("1.1"), 1)
        self.assertIsNone(position_of("verso"))

    def test_instants_must_move_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anotacao.csv"
            path.write_text("0.0,C\n1.0,G\n", encoding="utf-8")
            rows = read_instants(path)
            self.assertEqual([row[1] for row in rows], [0.0, 1.0])
            path.write_text("1.0,C\n0.5,G\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                read_instants(path)
