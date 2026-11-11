"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from cycles import (  # noqa: E402
    canonical_chord_choice,
    classify_cadence,
    discover_repeating_cycles,
)

F_MAJOR = 5
CYCLE = ["Dm", "F", "Bb", "F"]


def song(*laps: list, confidence: float = 0.8) -> list[dict]:
    """Compassos de várias voltas. Cada posição é uma cifra ou (cifra, confiança)."""
    rows = []
    for lap in laps:
        for slot in lap:
            text, trust = slot if isinstance(slot, tuple) else (slot, confidence)
            start = float(len(rows))
            chords = text.split()
            step = 1.0 / len(chords)
            rows.append(
                {
                    "bar": len(rows) + 1,
                    "start": start,
                    "end": start + 1.0,
                    "chords": chords,
                    "confidence": trust,
                    "events": [
                        {
                            "chord": chord,
                            "start": start + step * spot,
                            "end": start + step * (spot + 1),
                        }
                        for spot, chord in enumerate(chords)
                    ],
                }
            )
    return rows


def written(bars: list[dict]) -> list[str]:
    return [" ".join(bar["chords"]) for bar in bars]


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


class LapVoteTest(unittest.TestCase):
    """As voltas de um ciclo votam a cifra de cada posição."""

    def test_a_passing_chord_read_in_few_laps_gives_way_to_the_cycle(self):
        bars = song(CYCLE, ["Dm", "F", "Bb", "F C"], ["D Gm", "F", "Bb F", "F"], CYCLE, CYCLE)
        discover_repeating_cycles(bars, F_MAJOR, "maior")
        self.assertEqual(written(bars), CYCLE * 5)
        self.assertEqual(bars[7]["regra"], "consenso do ciclo")
        self.assertEqual(bars[7]["chords_modelo"], ["F", "C"])
        self.assertEqual(bars[8]["chords_modelo"], ["D", "Gm"])

    def test_a_confident_different_ending_stays_and_is_marked(self):
        bars = song(CYCLE, CYCLE, CYCLE, ["Dm", "F", "Bb", ("C", 0.9)])
        discover_repeating_cycles(bars, F_MAJOR, "maior")
        self.assertEqual(written(bars)[12:], ["Dm", "F", "Bb", "C"])
        self.assertEqual(bars[15]["variacao_do_ciclo"], "F")
        self.assertNotIn("regra", bars[15])
        self.assertNotIn("variacao_do_ciclo", bars[14])

    def test_an_unsure_different_chord_takes_the_cycle(self):
        bars = song(CYCLE, CYCLE, CYCLE, ["Dm", "F", "Bb", ("C", 0.5)])
        discover_repeating_cycles(bars, F_MAJOR, "maior")
        self.assertEqual(written(bars), CYCLE * 4)

    def test_a_passing_chord_read_in_most_laps_is_part_of_the_cycle(self):
        with_passing = ["Dm", "F", "Bb", "F C"]
        bars = song(with_passing, with_passing, with_passing, CYCLE)
        _, catalog = discover_repeating_cycles(bars, F_MAJOR, "maior")
        self.assertEqual(catalog[0]["acordes"], with_passing)
        self.assertEqual(written(bars)[:12], with_passing * 3)

    def test_the_lap_before_the_voice_weighs_more(self):
        dirty = ["Dm", "F", ("Bb F", 0.6), "F"]
        clean_intro = ["Dm", "F", ("Bb", 0.9), "F"]
        voiced = song(clean_intro, dirty, dirty)
        discover_repeating_cycles(voiced, F_MAJOR, "maior")
        self.assertEqual(written(voiced)[4:], ["Dm", "F", "Bb F", "F"] * 2)

        with_intro = song(clean_intro, dirty, dirty)
        discover_repeating_cycles(with_intro, F_MAJOR, "maior", quiet_until=4)
        self.assertEqual(written(with_intro), CYCLE * 3)

    def test_a_lap_that_differs_in_two_positions_is_left_alone(self):
        other = ["Dm", ("C", 0.5), ("Gm", 0.5), "F"]
        bars = song(CYCLE, CYCLE, other, CYCLE)
        discover_repeating_cycles(bars, F_MAJOR, "maior")
        self.assertEqual(written(bars)[8:12], ["Dm", "C", "Gm", "F"])


if __name__ == "__main__":
    unittest.main()
