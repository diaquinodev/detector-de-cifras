"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from chroma import note_name, pitch_class, tuning_from_bass  # noqa: E402
from constants import A4_HZ, PITCHES_SHARP  # noqa: E402


class ChromaTest(unittest.TestCase):
    def test_a4_is_the_pitch_class_a(self):
        self.assertEqual(pitch_class(A4_HZ, A4_HZ), 9)
        self.assertEqual(note_name(A4_HZ, PITCHES_SHARP), "A")

    def test_exact_bass_notes_keep_a440(self):
        reference, cents = tuning_from_bass(np.array([110.0, 220.0, 440.0]), np.array([0.9, 0.9, 0.9]))
        self.assertEqual(cents, 0.0)
        self.assertAlmostEqual(reference, A4_HZ, places=1)

    def test_too_few_confident_notes_do_not_move_the_tuning(self):
        reference, cents = tuning_from_bass(np.array([430.0]), np.array([0.99]))
        self.assertEqual(cents, 0.0)
        self.assertEqual(reference, A4_HZ)
