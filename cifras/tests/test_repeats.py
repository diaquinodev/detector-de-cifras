"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from repeats import _explains_better, chroma_by_bar, lock_repeated_phrases  # noqa: E402
from section_map import map_sections_with_theory  # noqa: E402


ROOTS = {
    "C": 0,
    "Cm": 0,
    "G": 7,
    "A": 9,
    "Am": 9,
    "F": 5,
    "Fm": 5,
    "Em": 4,
    "D": 2,
    "Dm": 2,
    "B": 11,
    "Bm": 11,
    "Bb": 10,
    "Bbm": 10,
}
MAJOR = ["F", "Bb", "F", "C", "F", "Bb", "C", "F"]
PARALLEL = ["Fm", "Bbm", "Fm", "Cm", "Fm", "Bbm", "Cm", "Fm"]
PHRASE = ["C", "G", "Am", "F", "C", "G", "F", "G"]


def root_fifth(name: str) -> np.ndarray:
    vector = np.zeros(12, dtype=float)
    root = ROOTS[name]
    vector[[root, (root + 7) % 12]] = 1.0
    return vector


def triad(name: str) -> np.ndarray:
    vector = np.zeros(12, dtype=float)
    if name == "N":
        return vector
    root = ROOTS[name]
    third = 3 if "m" in name else 4
    vector[[root, (root + third) % 12, (root + 7) % 12]] = 1.0
    return vector


def chart(chords: list[str], confidence: float | list[float]) -> tuple[list[dict], np.ndarray]:
    bars = []
    columns = []
    for index, chord in enumerate(chords):
        start = float(index)
        end = start + 1.0
        level = confidence[index] if isinstance(confidence, list) else confidence
        bars.append(
            {
                "bar": index + 1,
                "start": start,
                "end": end,
                "chords": [chord],
                "root": chord,
                "confidence": level,
                "events": [{"chord": chord, "start": start, "end": end, "confidence": level}],
            }
        )
        columns.append(triad(chord))
    return bars, np.stack(columns, axis=1)


def repeat_chart(second: list[str], source: float, target: float) -> tuple[list[dict], np.ndarray]:
    bars, chroma = chart(PHRASE + second, [source] * len(PHRASE) + [target] * len(second))
    chroma[:, len(PHRASE) :] = chroma[:, : len(second)]
    return bars, chroma


class RepeatLockTest(unittest.TestCase):
    def test_bar_chroma_is_the_mean_of_the_beats_in_the_bar(self):
        beats = np.zeros((12, 8), dtype=float)
        beats[0, :4] = 1.0
        beats[7, 4:] = 1.0
        profile = chroma_by_bar(beats, beats_per_bar=4, phase=0, downbeats=None)
        self.assertEqual(profile.shape, (12, 2))
        self.assertEqual(profile[0, 0], 1.0)
        self.assertEqual(profile[7, 1], 1.0)

    def test_the_core_copies_the_more_confident_pass_and_the_cadence_stays(self):
        second = ["C", "Em", "Am", "F", "C", "D", "Em", "Em"]
        bars, chroma = repeat_chart(second, source=0.9, target=0.4)
        copied = lock_repeated_phrases(bars, chroma)
        self.assertGreaterEqual(copied, 2)
        self.assertEqual(
            [bar["chords"] for bar in bars[8:14]], [["C"], ["G"], ["Am"], ["F"], ["C"], ["G"]]
        )
        self.assertEqual(bars[9]["regra"], "repetição do trecho")
        self.assertEqual(bars[9]["origem"], "regra")
        self.assertEqual([bar["chords"] for bar in bars[14:16]], [["Em"], ["Em"]])
        self.assertNotIn("regra", bars[14])

    def test_a_later_clearer_pass_becomes_the_model(self):
        first = ["C", "Em", "Am", "F", "C", "D", "F", "G"]
        bars, chroma = chart(first + PHRASE + first, [0.4] * 8 + [0.95] * 8 + [0.3] * 8)
        chroma[:, 8:16] = np.stack([triad(chord) for chord in PHRASE], axis=1)
        chroma[:, :8] = chroma[:, 8:16]
        chroma[:, 16:] = chroma[:, 8:16]
        lock_repeated_phrases(bars, chroma)
        self.assertEqual(bars[1]["chords"], ["G"])
        self.assertEqual(bars[17]["chords"], ["G"])
        self.assertEqual(bars[9]["chords"], ["G"])
        self.assertNotIn("regra", bars[9])

    def test_a_confident_miss_follows_the_model_and_a_blank_does_not_erase(self):
        second = ["C", "Em", "Am", "F", "C", "D", "F", "G"]
        bars, chroma = repeat_chart(second, source=0.7, target=0.4)
        bars[0]["chords"] = ["N"]
        bars[0]["confidence"] = 0.7
        bars[9]["confidence"] = 0.95
        lock_repeated_phrases(bars, chroma)
        self.assertEqual(bars[8]["chords"], ["C"])
        self.assertEqual(bars[9]["chords"], ["G"])
        self.assertEqual(bars[9]["regra"], "repetição do trecho")
        self.assertEqual(bars[13]["chords"], ["G"])

    def test_a_weak_fit_does_not_replace_the_local_chord(self):
        diffuse = np.ones(12)
        diffuse[0] += 0.2
        self.assertFalse(_explains_better({"chords": ["C"]}, {"chords": ["G"]}, diffuse))
        self.assertTrue(_explains_better({"chords": ["C"]}, {"chords": ["G"]}, triad("C")))

    def test_a_flat_texture_is_not_a_phrase_repeat(self):
        bars, chroma = chart(["G"] * 8 + ["C"] * 8, [0.4] * 8 + [0.95] * 8)
        chroma[:, :] = triad("C")[:, None]
        before = [bar["chords"][:] for bar in bars]
        self.assertEqual(lock_repeated_phrases(bars, chroma), 0)
        self.assertEqual([bar["chords"] for bar in bars], before)

    def test_a_parallel_minor_does_not_overwrite_major_when_the_third_is_missing(self):
        bars, chroma = chart(MAJOR + PARALLEL, [0.6] * 8 + [0.95] * 8)
        open_fifths = np.stack([root_fifth(chord) for chord in MAJOR], axis=1)
        chroma[:, :8] = open_fifths
        chroma[:, 8:] = open_fifths
        before = [bar["chords"][:] for bar in bars]
        self.assertEqual(lock_repeated_phrases(bars, chroma), 0)
        self.assertEqual([bar["chords"] for bar in bars], before)

    def test_a_noisier_label_does_not_replace_the_chord_the_audio_supports(self):
        bars, chroma = chart(MAJOR + MAJOR, [0.6] * 8 + [0.95] * 8)
        heard = np.stack([triad(chord) for chord in MAJOR], axis=1)
        chroma[:, :8] = heard
        chroma[:, 8:] = heard
        messy = [["F", "C"], ["Bb", "F"], ["Fm"], ["C", "Fm"], ["F"], ["Bb"], ["Cm", "C"], ["F"]]
        for index, chords in enumerate(messy):
            bars[8 + index]["chords"] = chords
        lock_repeated_phrases(bars, chroma)
        self.assertEqual([bar["chords"] for bar in bars[:8]], [[chord] for chord in MAJOR])
        self.assertEqual(bars[10]["chords"], ["F"])
        self.assertEqual([bar["chords"] for bar in bars[14:16]], [["Cm", "C"], ["F"]])

    def test_the_pass_that_matches_the_audio_becomes_the_model(self):
        bars, chroma = chart(PARALLEL + MAJOR, [0.95] * 8 + [0.6] * 8)
        heard = np.stack([triad(chord) for chord in MAJOR], axis=1)
        chroma[:, :8] = heard
        chroma[:, 8:] = heard
        lock_repeated_phrases(bars, chroma)
        self.assertEqual(
            [bar["chords"] for bar in bars[:6]], [["F"], ["Bb"], ["F"], ["C"], ["F"], ["Bb"]]
        )
        self.assertEqual(bars[0]["regra"], "repetição do trecho")
        self.assertEqual([bar["chords"] for bar in bars[6:8]], [["Cm"], ["Fm"]])
        self.assertNotIn("regra", bars[8])

    def test_a_transposition_is_not_copied(self):
        transposed = ["D", "A", "Bm", "G", "D", "A", "G", "A"]
        bars, _chroma = chart(PHRASE + transposed, [0.9] * 8 + [0.4] * 8)
        before = [bar["chords"][:] for bar in bars]
        self.assertEqual(lock_repeated_phrases(bars, _chroma), 0)
        self.assertEqual([bar["chords"] for bar in bars], before)

    def test_a_different_progression_is_not_copied(self):
        bars, chroma = chart(
            PHRASE + ["B", "B", "B", "B", "B", "B", "B", "B"], [0.9] * 8 + [0.4] * 8
        )
        before = [bar["chords"][:] for bar in bars]
        self.assertEqual(lock_repeated_phrases(bars, chroma), 0)
        self.assertEqual([bar["chords"] for bar in bars], before)

    def test_a_chroma_of_another_length_leaves_the_chart_alone(self):
        bars, _chroma = chart(PHRASE + PHRASE, 0.8)
        before = [bar["chords"][:] for bar in bars]
        self.assertEqual(lock_repeated_phrases(bars, np.zeros((12, 3))), 0)
        self.assertEqual([bar["chords"] for bar in bars], before)

    def test_the_section_map_uses_the_repeated_audio(self):
        second = ["C", "Em", "Am", "F", "C", "D", "B", "B"]
        confidence = [0.9] * len(PHRASE) + [0.85] * len(second)
        bars, chroma = chart(PHRASE + second, confidence)
        chroma[:, len(PHRASE) :] = chroma[:, : len(PHRASE)]
        map_sections_with_theory(
            bars,
            {"name": "C", "mode": "maior", "score": 1.0},
            vocal_presence=np.ones(len(bars)),
            vocal_chest=np.ones(len(bars)),
            bar_chroma=chroma,
        )
        self.assertEqual(bars[9]["chords"], ["G"])
        self.assertEqual(bars[9]["regra"], "repetição do trecho")
        self.assertEqual(bars[13]["chords"], ["G"])
        self.assertEqual([bar["chords"] for bar in bars[14:16]], [["B"], ["B"]])

        fresh, _chroma = chart(PHRASE + second, confidence)
        map_sections_with_theory(
            fresh,
            {"name": "C", "mode": "maior", "score": 1.0},
            vocal_presence=np.ones(len(fresh)),
            vocal_chest=np.ones(len(fresh)),
        )
        self.assertEqual(fresh[9]["chords"], ["Em"])
