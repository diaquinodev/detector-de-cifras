"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

import analyze  # noqa: E402
from analyze import chart_document  # noqa: E402


class AnalyzeInterfaceTest(unittest.TestCase):
    def test_keeps_the_names_other_code_imported_from_analyze(self):
        # servir.py e os testes pegavam estes nomes por dentro do analyze antes da fase 1.
        for name in ("MAX_AUDIO_SECONDS", "PITCHES_SHARP", "interval_bass", "ROOT", "analyze"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(analyze, name))


class AnalyzeDocumentTest(unittest.TestCase):
    def test_the_chart_document_counts_ruled_bars_and_keeps_the_key(self):
        path = Path("faixa.wav")
        pulse = {
            "bpm": 120.0,
            "beat_times": np.array([0.0, 0.5, 1.0, 1.5]),
            "compasso_ms": 0.0,
            "collapsed_eighths": False,
        }
        meter = {
            "numerator": 4,
            "denominator": 4,
            "beats_per_bar": 4,
            "formula_confirmed": True,
            "compound_subdivision": False,
            "margin": 0.2,
            "downbeat_indexes": [0],
        }
        key = {"name": "C", "mode": "maior", "score": 0.8, "reference": 440.0, "cents": 0.0}
        bars = [
            {
                "bar": 1,
                "start": 0.0,
                "end": 2.0,
                "chords": ["C"],
                "origem": "regra",
                "confidence": 0.5,
            }
        ]
        sections = [{"name": "verso", "start": 0.0, "end": 2.0, "bars": bars}]
        document = chart_document(
            path,
            2.0,
            pulse,
            meter,
            key,
            bars,
            "btc",
            sections,
            {},
            [],
            "ciclos",
            {},
            1.0,
        )
        self.assertEqual(document["arquivo"], "faixa.wav")
        self.assertEqual(document["tom"]["nota"], "C")
        self.assertEqual(document["compasso"]["formula"], "4/4")
        self.assertEqual(document["compassos_ajustados_por_regra"], 1)
        self.assertIn("verso", document["cifra"])
