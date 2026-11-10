"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

import harmonic_structure  # noqa: E402
import section_map  # noqa: E402
import spelling  # noqa: E402


class HarmonicFacadeTest(unittest.TestCase):
    def test_the_facade_exposes_the_component_functions(self):
        self.assertIs(harmonic_structure.key_spelling, spelling.key_spelling)
        self.assertIs(harmonic_structure.spell_with, spelling.spell_with)
        self.assertIs(harmonic_structure.map_sections_with_theory, section_map.map_sections_with_theory)
        self.assertTrue(callable(harmonic_structure.discover_repeating_cycles))
