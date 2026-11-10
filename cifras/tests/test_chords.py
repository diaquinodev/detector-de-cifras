"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from chords import (  # noqa: E402
    choose_beat_state,
    chord_root_name,
    fold_bar,
    kind_index,
    major_triad_root,
    profile_scores,
    relative_minor_seventh,
    scale_steps,
)
from constants import MAJOR_PROFILE, MINOR_PROFILE  # noqa: E402


class ChordComponentTest(unittest.TestCase):
    def test_c_major_profile_ranks_c_major_first(self):
        chroma = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
        best = max(
            profile_scores(chroma, MAJOR_PROFILE, MINOR_PROFILE), key=lambda item: item["score"]
        )
        self.assertEqual(best["tonic"], 0)
        self.assertEqual(best["mode"], "maior")

    def test_scale_and_triad_helpers(self):
        self.assertEqual(scale_steps("menor"), (0, 2, 3, 5, 7, 8, 10))
        self.assertEqual(major_triad_root([0, 4, 7]), 0)
        self.assertIsNone(major_triad_root([0, 3, 7]))
        self.assertEqual(chord_root_name("F#m7"), "F#")
        self.assertEqual(chord_root_name("N"), "N")

    def test_a_one_beat_blip_between_the_same_chord_is_folded_away(self):
        self.assertEqual(fold_bar(["C", "G", "C", "C"]), ["C"])

    def test_bass_a_minor_third_under_a_major_triad_is_a_minor_seventh(self):
        chroma = np.zeros(12)
        chroma[[0, 4, 7]] = 1.0
        self.assertTrue(relative_minor_seventh(chroma, bass_pc=9))
        self.assertFalse(relative_minor_seventh(chroma, bass_pc=0))

    def test_a_close_call_keeps_the_previous_root_when_it_is_in_key(self):
        kinds = 3
        column = np.zeros(12 * kinds)
        column[0] = 1.0
        column[kinds * 7] = 0.99
        choice = choose_beat_state(column, previous_root=7, kinds=kinds, diatonic={0, 7})
        self.assertEqual(choice // kinds, 7)
        self.assertEqual(kind_index("m"), 1)
