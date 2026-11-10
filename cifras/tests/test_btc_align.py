"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from btc_align import bar_interior_points, btc_parts, interval_bass, spell_btc, timeline_vote  # noqa: E402
from constants import PITCHES_SHARP  # noqa: E402


class BtcAlignTest(unittest.TestCase):
    def test_btc_symbols_become_chart_symbols(self):
        self.assertEqual(btc_parts("N", PITCHES_SHARP), ("N", None, ""))
        self.assertEqual(btc_parts("C:maj", PITCHES_SHARP), ("C", 0, ""))
        self.assertEqual(btc_parts("A:min7", PITCHES_SHARP)[0], "Am7")

    def test_a_sustained_flat_seventh_in_the_bass_is_a_slash_not_a_seventh(self):
        self.assertEqual(spell_btc("G:7", PITCHES_SHARP, bass=5), "G/F")

    def test_the_longer_overlap_wins_the_bar(self):
        timeline = [
            {"chord": "C:maj", "start": 0.0, "end": 0.8, "confidence": 0.9, "margin": 0.5},
            {"chord": "G:maj", "start": 0.8, "end": 1.0, "confidence": 0.4, "margin": 0.1},
        ]
        chord, share, confidence, _margin = timeline_vote(timeline, 0.0, 1.0)
        self.assertEqual(chord, "C:maj")
        self.assertAlmostEqual(share, 0.8)
        self.assertAlmostEqual(confidence, 0.9)

    def test_interior_beats_keep_the_bar_edges(self):
        points = bar_interior_points(0.0, 4.0, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(points[0], 0.0)
        self.assertEqual(points[-1], 4.0)
        self.assertIn(2.0, points)

    def test_a_pedal_bass_is_kept_only_when_it_repeats(self):
        beats = np.array([0.0, 1.0, 2.0, 3.0])
        bass = np.array([5, 5, 5, 5])
        confidence = np.array([0.95, 0.95, 0.95, 0.95])
        self.assertEqual(interval_bass(0.0, 4.0, beats, bass, confidence), 5)
        self.assertIsNone(interval_bass(0.0, 4.0, beats, np.array([-1, -1, -1, -1]), confidence))
