"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

import servir  # noqa: E402


class ServirComponentTest(unittest.TestCase):
    def test_upload_names_stay_inside_the_audio_folder(self):
        self.assertTrue(servir.safe_upload_name("Meu Som.mp3").endswith(".mp3"))
        with self.assertRaises(ValueError):
            servir.safe_upload_name("../segredo.txt")
        with self.assertRaises(ValueError):
            servir.safe_upload_name("faixa.exe")

    def test_only_youtube_hosts_pass(self):
        self.assertTrue(servir.youtube_url("https://youtu.be/abc").startswith("https://"))
        with self.assertRaises(ValueError):
            servir.youtube_url("https://example.com/watch?v=1")
        with self.assertRaises(ValueError):
            servir.youtube_url("ftp://youtu.be/abc")

    def test_a_path_outside_the_project_is_rejected(self):
        with self.assertRaises(ValueError):
            servir.resolve_track("../fora.mp3")
        with self.assertRaises(ValueError):
            servir.resolve_track("C:/Windows/media.wav")
