"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from avaliar import chord_intervals, tempo_accuracy, to_harte  # noqa: E402


class AvaliarTest(unittest.TestCase):
    def test_symbols_become_harte(self):
        self.assertEqual(to_harte("N"), "N")
        self.assertEqual(to_harte("Am7"), "A:min7")
        self.assertEqual(to_harte("C/E"), "C:maj/3")
        self.assertEqual(to_harte("nota"), "X")

    def test_tempo_accepts_the_same_pulse_and_its_double(self):
        exact, flexible = tempo_accuracy(120.0, 120.0)
        self.assertTrue(exact)
        self.assertTrue(flexible)
        exact, flexible = tempo_accuracy(60.0, 120.0)
        self.assertFalse(exact)
        self.assertTrue(flexible)

    def test_events_become_labeled_intervals(self):
        bars = [{"start": 0.0, "end": 2.0, "chords": ["C"], "events": [{"chord": "C", "start": 0.0, "end": 1.0}, {"chord": "G", "start": 1.0, "end": 2.0}]}]
        intervals, labels = chord_intervals(bars, model_only=False)
        self.assertEqual(labels, ["C:maj", "G:maj"])
        self.assertEqual(intervals.shape, (2, 2))
