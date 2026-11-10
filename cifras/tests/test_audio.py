"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from audio import magnitude  # noqa: E402
from constants import SAMPLE_RATE  # noqa: E402


class MagnitudeTest(unittest.TestCase):
    def test_a_tone_puts_energy_in_the_spectrum(self):
        seconds = 0.5
        times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
        samples = 0.2 * np.sin(2 * np.pi * 440 * times)
        freqs, spectrum = magnitude(samples)
        self.assertGreater(spectrum.size, 0)
        self.assertEqual(freqs.shape[0], spectrum.shape[0])
        self.assertGreater(float(np.max(spectrum)), 0.0)
