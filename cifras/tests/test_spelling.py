"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from spelling import (  # noqa: E402
    bar_similarity,
    bar_signature,
    parse_chord_symbol,
    respell_chord_symbol,
    roman_and_function,
)


class SpellingComponentTest(unittest.TestCase):
    def test_parse_keeps_root_quality_and_bass(self):
        parsed = parse_chord_symbol("F#m7/C#")
        self.assertEqual(parsed["root_name"], "F#")
        self.assertEqual(parsed["suffix"], "m7")
        self.assertEqual(parsed["family"], "min")
        self.assertEqual(parsed["bass_name"], "C#")

    def test_rest_and_flat_respelling(self):
        self.assertEqual(parse_chord_symbol("N")["root"], None)
        self.assertEqual(respell_chord_symbol("Bbmaj7", prefer_sharp=True), "A#maj7")
        self.assertEqual(respell_chord_symbol("A#maj7", prefer_sharp=False), "Bbmaj7")

    def test_c_major_tonic_is_degree_one(self):
        info = parse_chord_symbol("C")
        roman, function = roman_and_function(info, tonic=0, mode="maior")
        self.assertEqual(roman, "I")
        self.assertEqual(function, "T")

    def test_the_same_bar_matches_itself(self):
        signature = bar_signature({"chords": ["Am7", "D7"]}, tonic=0, mode="maior")
        self.assertEqual(bar_similarity(signature, signature), 1.0)
