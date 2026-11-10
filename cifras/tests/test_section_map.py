"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from section_map import map_sections_with_theory  # noqa: E402


def bars_of(chord: str, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        start = float(index)
        end = start + 1.0
        rows.append({
            "bar": index + 1,
            "start": start,
            "end": end,
            "chords": [chord],
            "events": [{"chord": chord, "start": start, "end": end, "confidence": 0.8}],
            "confidence": 0.8,
        })
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
