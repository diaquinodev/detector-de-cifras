"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from cycles import canonical_chord_choice, classify_cadence  # noqa: E402


class CycleComponentTest(unittest.TestCase):
    def test_silence_and_authentic_cadence(self):
        self.assertEqual(classify_cadence(["N"], ["N"]), "silêncio")
        self.assertEqual(
            classify_cadence(["IV", "V", "I"], ["SD", "D", "T"]), "cadência autêntica (SD–D–T)"
        )

    def test_a_repeated_slot_keeps_the_chord_the_bars_agree_on(self):
        bars = [
            {"chords": ["C"], "confidence": 0.4},
            {"chords": ["C"], "confidence": 0.9},
            {"chords": ["G"], "confidence": 0.2},
        ]
        sigs = [{"root": 0}] * 3
        self.assertEqual(canonical_chord_choice(bars, sigs), ["C"])
