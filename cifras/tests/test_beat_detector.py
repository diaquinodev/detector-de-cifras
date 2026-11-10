"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from beat_detector import NEUTRAL_PROBABILITY, downbeat_probability  # noqa: E402


class BeatDetectorComponentTest(unittest.TestCase):
    def test_a_strong_logit_raises_the_downbeat_probability(self):
        logits = np.zeros(20)
        logits[10] = 5.0
        probabilities = downbeat_probability(logits, np.array([10 / 50]))
        self.assertGreater(float(probabilities[0]), 0.9)

    def test_a_beat_outside_the_logits_stays_neutral(self):
        probabilities = downbeat_probability(np.zeros(4), np.array([10.0]))
        self.assertEqual(float(probabilities[0]), NEUTRAL_PROBABILITY)
