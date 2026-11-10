"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from constants import SAMPLE_RATE  # noqa: E402
from form import bar_level, clock, final_rest, first_sustained, voice_reentries  # noqa: E402


class FormComponentTest(unittest.TestCase):
    def test_clock_rounds_to_the_minute(self):
        self.assertEqual(clock(65.4), "1:05")
        self.assertEqual(clock(-3), "0:00")

    def test_the_voice_onset_is_the_first_sustained_rise(self):
        presence = np.array([0.1, 0.1, 0.8, 0.9, 0.2])
        self.assertEqual(first_sustained(presence, 0.5), 2)

    def test_a_long_quiet_tail_is_the_final_rest(self):
        presence = np.array([0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
        self.assertLess(final_rest(presence, 0.5), presence.size)

    def test_the_voice_can_come_back_after_a_quiet_stretch(self):
        presence = np.array([0.9] + [0.1] * 8 + [0.9])
        self.assertEqual(voice_reentries(presence, floor=0.5, start_at=0), [9])

    def test_bar_level_reads_the_samples_inside_the_bar(self):
        envelope = np.zeros(SAMPLE_RATE)
        envelope[: SAMPLE_RATE // 2] = 1.0
        self.assertGreater(bar_level(envelope, 0.0, 0.4), bar_level(envelope, 0.6, 1.0))
