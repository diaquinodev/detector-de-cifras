"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

import constants  # noqa: E402


class ConstantsTest(unittest.TestCase):
    def test_pitch_names_cover_the_octave(self):
        self.assertEqual(len(constants.PITCHES_SHARP), 12)
        self.assertEqual(len(constants.PITCHES_FLAT), 12)
        self.assertEqual(constants.PITCH_TO_PC["Bb"], 10)
        self.assertEqual(constants.PITCH_TO_PC["A#"], 10)

    def test_major_triad_is_the_first_chord_kind(self):
        name, intervals = constants.CHORD_KINDS[0]
        self.assertEqual(name, "")
        self.assertEqual(intervals, (0, 4, 7))

    def test_audio_limits_stay_explicit(self):
        self.assertEqual(constants.SAMPLE_RATE, 22050)
        self.assertEqual(constants.MAX_AUDIO_SECONDS, 20 * 60)
        self.assertIn(".wav", constants.AUDIO_SUFFIXES)
