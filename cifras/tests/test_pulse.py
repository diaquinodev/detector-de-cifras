"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from pulse import bar_starts, neural_meter, normalize, prefer_musical_pulse, signature_for  # noqa: E402


class PulseTest(unittest.TestCase):
    def test_flat_values_normalize_to_zero(self):
        self.assertTrue(np.all(normalize(np.array([3.0, 3.0, 3.0])) == 0))

    def test_compound_meters_use_an_eighth_denominator(self):
        self.assertEqual(signature_for(4, False), (4, 4))
        self.assertEqual(signature_for(4, True), (12, 8))
        self.assertEqual(signature_for(2, True), (6, 8))

    def test_bar_starts_include_the_first_beat(self):
        self.assertEqual(bar_starts(8, 4, 0).tolist(), [0, 4])
        self.assertEqual(bar_starts(8, 4, 0, np.array([4])).tolist(), [0, 4])

    def test_a_fast_pulse_with_a_strong_half_collapses_to_the_beat(self):
        times = np.arange(16, dtype=float) * 0.25
        accent = np.array([2.0, 1.0] * 8)
        kept, tempo, collapsed = prefer_musical_pulse(times, accent, 160.0)
        self.assertTrue(collapsed)
        self.assertEqual(tempo, 80.0)
        self.assertEqual(kept.tolist(), times[0::2].tolist())

    def test_neural_meter_names_the_bar_without_calling_it_compound(self):
        meter = neural_meter({"beats_per_bar": 4, "phase": 0, "downbeat_indexes": [0, 4]})
        self.assertEqual(meter["numerator"], 4)
        self.assertEqual(meter["denominator"], 4)
        self.assertFalse(meter["compound_subdivision"])
        self.assertEqual(meter["source"], "beat_this")
