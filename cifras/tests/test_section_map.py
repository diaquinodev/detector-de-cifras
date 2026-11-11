"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from section_map import _vocal_onset_from_profile, map_sections_with_theory  # noqa: E402


def bars_of(chord: str, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        start = float(index)
        end = start + 1.0
        rows.append(
            {
                "bar": index + 1,
                "start": start,
                "end": end,
                "chords": [chord],
                "events": [{"chord": chord, "start": start, "end": end, "confidence": 0.8}],
                "confidence": 0.8,
            }
        )
    return rows


class SectionMapTest(unittest.TestCase):
    def test_an_empty_chart_has_no_sections(self):
        sections, voice, patterns = map_sections_with_theory([], {"name": "C", "mode": "maior"})
        self.assertEqual(sections, [])
        self.assertEqual(voice, {})
        self.assertEqual(patterns, [])

    def test_a_repeated_loop_becomes_at_least_one_section(self):
        bars = bars_of("C", 8)
        sections, _voice, _patterns = map_sections_with_theory(
            bars,
            {"name": "C", "mode": "maior", "score": 1.0},
            vocal_presence=np.ones(8),
            vocal_chest=np.ones(8),
        )
        self.assertGreaterEqual(len(sections), 1)
        self.assertIn("name", sections[0])
        self.assertEqual(sections[0]["start"], 0.0)


def lap_bars(laps: list[list]) -> list[dict]:
    rows = []
    for lap in laps:
        for slot in lap:
            text, trust = slot if isinstance(slot, tuple) else (slot, 0.8)
            start = float(len(rows))
            rows.append(
                {
                    "bar": len(rows) + 1,
                    "start": start,
                    "end": start + 1.0,
                    "chords": text.split(),
                    "confidence": trust,
                    "events": [],
                }
            )
    return rows


class QuietLapVoteTest(unittest.TestCase):
    """A volta antes de a voz entrar pesa mais, pelo fluxo inteiro do mapa."""

    LAPS = [
        ["Dm", "F", ("Bb", 0.9), "F"],
        ["Dm", "F", ("Bb F", 0.7), "F"],
        ["Dm", "F", ("Bb F", 0.7), "F"],
    ]
    # Votos na terceira posição: limpa 0,9 e suja 0,7 + 0,7. A limpa só ganha quando
    # a primeira volta, e só ela, vale em dobro (1,8 contra 1,4). Com duas voltas em
    # dobro a suja ganharia (2,1 contra 1,8).
    KEY = {"name": "F", "mode": "maior"}

    def third_slots(self, bars: list[dict]) -> list[str]:
        return [" ".join(bars[index]["chords"]) for index in (2, 6, 10)]

    def test_without_any_voice_measure_every_lap_weighs_the_same(self):
        bars = lap_bars(self.LAPS)
        map_sections_with_theory(bars, self.KEY)
        self.assertEqual(self.third_slots(bars), ["Bb", "Bb F", "Bb F"])

    def test_the_voice_onset_from_the_detector_counts(self):
        bars = lap_bars(self.LAPS)
        map_sections_with_theory(bars, self.KEY, vocal_onset_bar=4)
        self.assertEqual(self.third_slots(bars), ["Bb", "Bb", "Bb"])

    def test_two_quiet_laps_would_give_another_answer(self):
        bars = lap_bars(self.LAPS)
        map_sections_with_theory(bars, self.KEY, vocal_onset_bar=8)
        self.assertEqual(self.third_slots(bars), ["Bb", "Bb F", "Bb F"])

    def test_the_voice_onset_from_the_profile_counts_too(self):
        bars = lap_bars(self.LAPS)
        presence = np.array([0.05] * 4 + [1.0] * 8)
        self.assertEqual(_vocal_onset_from_profile(presence, presence.copy(), 12), 4)
        map_sections_with_theory(bars, self.KEY, vocal_presence=presence)
        self.assertEqual(self.third_slots(bars), ["Bb", "Bb", "Bb"])


if __name__ == "__main__":
    unittest.main()
