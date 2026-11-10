"""Rode com: python -m unittest discover -s cifras/tests"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from beat_detector import decode_bar_positions  # noqa: E402
from harmonic_structure import (  # noqa: E402
    detect_key_modulation,
    key_fitting_the_chords,
    key_spelling,
    override_bar,
    spell_with,
)
import servir  # noqa: E402


def probabilities_for(
    size: int, strong: list[int], high: float = 0.9, low: float = 0.05
) -> np.ndarray:
    values = np.full(size, low)
    values[strong] = high
    return values


class BarPositionTest(unittest.TestCase):
    def test_regular_downbeats(self):
        result = decode_bar_positions(probabilities_for(16, [0, 4, 8, 12]), 4)
        self.assertEqual(result.tolist(), [0, 4, 8, 12])

    def test_one_missing_downbeat_is_filled(self):
        result = decode_bar_positions(probabilities_for(16, [0, 4, 12]), 4)
        self.assertEqual(result.tolist(), [0, 4, 8, 12])

    def test_mid_bar_marks_do_not_halve_the_bar(self):
        result = decode_bar_positions(probabilities_for(24, [0, 4, 6, 8, 10, 12, 14, 16, 20]), 4)
        self.assertEqual(result.tolist(), [0, 4, 8, 12, 16, 20])

    def test_sustained_shift_moves_the_phase(self):
        strong = [0, 4, 8, 10, 14, 18, 22, 26, 30]
        result = decode_bar_positions(probabilities_for(32, strong), 4)
        self.assertEqual(result.tolist(), strong)

    def test_no_one_beat_bar(self):
        result = decode_bar_positions(
            probabilities_for(16, [0, 4, 5, 6, 7, 8, 12], high=0.98, low=0.02), 4
        )
        self.assertTrue(np.all(np.diff(result) >= 2))

    def test_neutral_edges_follow_the_grid(self):
        values = probabilities_for(20, [4, 8, 12])
        values[:3] = 0.5
        values[16:] = 0.5
        self.assertEqual(decode_bar_positions(values, 4).tolist(), [0, 4, 8, 12, 16])


class OverrideBarTest(unittest.TestCase):
    def test_events_follow_the_rule_and_keep_the_model(self):
        bar = {
            "start": 10.0,
            "end": 12.0,
            "chords": ["Bm", "A"],
            "events": [
                {"chord": "Bm", "start": 10.0, "end": 11.5, "confidence": 0.3},
                {"chord": "A", "start": 11.5, "end": 12.0, "confidence": 0.3},
            ],
            "confidence": 0.3,
        }
        override_bar(bar, ["Gbm", "Gbm7"], "consenso do ciclo")
        self.assertEqual([event["chord"] for event in bar["events"]], bar["chords"])
        self.assertEqual(bar["chords_modelo"], ["Bm", "A"])
        self.assertEqual(bar["events"][0]["start"], 10.0)
        self.assertEqual(bar["events"][-1]["end"], 12.0)
        self.assertEqual(bar["origem"], "regra")

    def test_same_chords_change_nothing(self):
        bar = {"start": 0.0, "end": 1.0, "chords": ["C"], "events": []}
        override_bar(bar, ["C"], "consenso do ciclo")
        self.assertNotIn("origem", bar)


class ServerGuardTest(unittest.TestCase):
    def test_hosts(self):
        self.assertTrue(servir.host_allowed("127.0.0.1:8765"))
        self.assertTrue(servir.host_allowed("localhost:8765"))
        self.assertFalse(servir.host_allowed("evil.example:8765"))
        self.assertFalse(servir.host_allowed(None))

    def test_origins(self):
        self.assertTrue(servir.origin_allowed("http://127.0.0.1:8765"))
        self.assertFalse(servir.origin_allowed("https://evil.example"))
        self.assertFalse(servir.origin_allowed("null"))
        self.assertFalse(servir.origin_allowed(None))

    def test_content_type(self):
        self.assertTrue(
            servir.content_type_is("application/json; charset=utf-8", "application/json")
        )
        self.assertFalse(servir.content_type_is("text/plain", "application/json"))

    def test_youtube_is_off_by_default(self):
        self.assertFalse(servir.YOUTUBE_ENABLED)


class PageContractTest(unittest.TestCase):
    """O visual pode mudar; estes IDs são o contrato entre a página e o servidor."""

    def test_controls_keep_their_ids(self):
        html = (CIFRAS / "web" / "index.html").read_text(encoding="utf-8")
        for needle in (
            'id="audio-file"',
            'id="upload"',
            'id="youtube-box"',
            'id="youtube-url"',
            'id="youtube"',
            'id="import-status"',
            'id="songs"',
            'id="sheet"',
            'id="song-filter"',
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, html)
        self.assertIn("/api/upload", html)
        self.assertIn("/api/analisar", html)
        self.assertIn("/api/cifra", html)
        self.assertIn("/api/youtube", html)
        self.assertIn("/api/musicas", html)
        self.assertIn('id="player"', html)
        self.assertIn('id="again"', html)
        self.assertIn('id="toggle-notation"', html)
        self.assertIn('id="toggle-model"', html)

    def test_transport_controls_keep_their_ids(self):
        html = (CIFRAS / "web" / "index.html").read_text(encoding="utf-8")
        for needle in (
            'id="play-toggle"',
            'id="count-in"',
            'id="bpm-readout"',
            'id="transpose-readout"',
            'id="loop-toggle"',
            'id="edit-toggle"',
            "/api/cifra",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, html)

    def test_design_module_is_separate(self):
        self.assertTrue((CIFRAS / "web" / "styles.css").is_file())
        html = (CIFRAS / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="/styles.css"', html)
        self.assertTrue(hasattr(servir, "STYLES"))


class ChordEditTest(unittest.TestCase):
    def test_replaces_one_chord_and_keeps_the_rest(self):
        data = {
            "compassos": [
                {
                    "bar": 2,
                    "start": 1,
                    "end": 2,
                    "chords": ["C", "G"],
                    "events": [
                        {"chord": "C", "start": 1, "end": 1.5},
                        {"chord": "G", "start": 1.5, "end": 2},
                    ],
                }
            ]
        }
        servir.apply_chord_edit(data, 2, 1, "D7")
        self.assertEqual(data["compassos"][0]["chords"], ["C", "D7"])
        self.assertEqual(data["compassos"][0]["events"][0]["chord"], "C")
        self.assertEqual(data["compassos"][0]["origem"], "edicao")

    def test_rejects_a_chord_that_is_not_a_symbol(self):
        with self.assertRaises(ValueError):
            servir.apply_chord_edit(
                {"compassos": [{"bar": 1, "chords": ["C"], "events": [{"chord": "C"}]}]},
                1,
                0,
                "alert(1)",
            )


class SpellingTest(unittest.TestCase):
    def test_degrees_keep_their_letter(self):
        c_major = key_spelling(0, "maior")
        self.assertEqual([c_major[pc] for pc in (1, 3, 6, 8, 10)], ["Db", "Eb", "F#", "Ab", "Bb"])
        a_minor = key_spelling(9, "menor")
        self.assertEqual([a_minor[pc] for pc in (10, 1, 6, 8)], ["Bb", "C#", "F#", "G#"])
        self.assertEqual(key_spelling(2, "maior")[1], "C#")
        self.assertEqual(key_spelling(8, "maior")[1], "Db")

    def test_no_odd_names(self):
        for tonic in range(12):
            for mode in ("maior", "menor"):
                self.assertFalse(set(key_spelling(tonic, mode)) & {"E#", "B#", "Cb", "Fb"})

    def test_slash_chords(self):
        self.assertEqual(spell_with("A#7/D#", key_spelling(0, "maior")), "Bb7/Eb")


class KeyFitTest(unittest.TestCase):
    def test_switches_only_with_a_wide_margin(self):
        bars = [{"chords": [chord]} for chord in ("Bm", "Em7", "F#7", "Bm", "G", "D", "A", "F#7")]
        fixed = key_fitting_the_chords(bars, {"tonic": 9, "mode": "menor"})
        self.assertEqual((fixed["tonic"], fixed["mode"]), (11, "menor"))
        self.assertIsNone(key_fitting_the_chords(bars, {"tonic": 2, "mode": "maior"}))


class ModulationTest(unittest.TestCase):
    def test_minor_key_iv_is_not_a_modulation(self):
        loop = ["Bm", "Em7", "F#7", "Bm", "G", "Em7", "A", "F#7"]
        bars = [
            {"chords": [chord], "start": float(i), "end": float(i + 1)}
            for i, chord in enumerate(loop * 5)
        ]
        self.assertIsNone(detect_key_modulation(bars, 11, "menor"))

    def test_half_step_up_in_major_is_found(self):
        home = ["Fmaj7", "G", "Em7", "Am7"] * 7
        up = ["F#maj7", "G#", "Fm7", "A#m7"] * 4
        bars = [
            {"chords": [chord], "start": float(i), "end": float(i + 1)}
            for i, chord in enumerate(home + up)
        ]
        found = detect_key_modulation(bars, 0, "maior")
        self.assertEqual((found["new_tonic"], found["semitones"]), (1, 1))


class ChordOnsetTest(unittest.TestCase):
    def setUp(self):
        from analyze import PITCHES_SHARP, btc_bars

        self.names = PITCHES_SHARP
        self.btc_bars = btc_bars
        self.beats = np.arange(0.0, 4.01, 0.5)
        self.bass_pc = np.full(self.beats.size, -1)
        self.bass_conf = np.zeros(self.beats.size)
        self.bar = [{"bar": 1, "start": 0.0, "end": 4.0, "chords": ["N"], "confidence": 0.0}]

    def events_for(self, timeline):
        bars = [dict(self.bar[0])]
        self.btc_bars(bars, timeline, self.beats, self.bass_pc, self.bass_conf, self.names)
        return bars[0]["events"]

    def test_syncopated_change_keeps_the_model_time(self):
        events = self.events_for(
            [
                {"start": 0.0, "end": 2.35, "chord": "C", "confidence": 0.9, "margin": 0.4},
                {"start": 2.35, "end": 4.0, "chord": "G", "confidence": 0.9, "margin": 0.4},
            ]
        )
        self.assertEqual([event["chord"] for event in events], ["C", "G"])
        self.assertAlmostEqual(events[1]["start"], 2.35, places=2)
        self.assertAlmostEqual(events[0]["end"], 2.35, places=2)
        self.assertAlmostEqual(events[0]["start"], 0.0, places=2)
        self.assertAlmostEqual(events[-1]["end"], 4.0, places=2)

    def test_change_within_a_frame_stays_on_the_beat(self):
        events = self.events_for(
            [
                {"start": 0.0, "end": 2.04, "chord": "C", "confidence": 0.9, "margin": 0.4},
                {"start": 2.04, "end": 4.0, "chord": "G", "confidence": 0.9, "margin": 0.4},
            ]
        )
        self.assertAlmostEqual(events[1]["start"], 2.0, places=2)

    def test_one_beat_flicker_stays_removed(self):
        events = self.events_for(
            [
                {"start": 0.0, "end": 2.0, "chord": "C", "confidence": 0.9, "margin": 0.4},
                {"start": 2.0, "end": 2.5, "chord": "G", "confidence": 0.2, "margin": 0.05},
                {"start": 2.5, "end": 4.0, "chord": "C", "confidence": 0.9, "margin": 0.4},
            ]
        )
        self.assertEqual([event["chord"] for event in events], ["C"])


class RootSupportTest(unittest.TestCase):
    """O croma é (12, tempos). Do tempo 12 em diante a coluna tem de valer."""

    def bars_for(self, chroma):
        from analyze import PITCHES_SHARP, btc_bars

        beats = np.arange(0.0, 8.01, 0.5)
        local = [{"label": "C", "confidence": 0.8} for _ in range(beats.size)]
        timeline = [
            {"start": 0.0, "end": 6.5, "chord": "C", "confidence": 0.9, "margin": 0.5},
            {"start": 6.5, "end": 7.0, "chord": "G", "confidence": 0.40, "margin": 0.20},
            {"start": 7.0, "end": 8.0, "chord": "C", "confidence": 0.9, "margin": 0.5},
        ]
        bars = [{"bar": 1, "start": 0.0, "end": 8.0, "chords": ["N"], "confidence": 0.0}]
        btc_bars(
            bars,
            timeline,
            beats,
            np.full(beats.size, -1),
            np.zeros(beats.size),
            PITCHES_SHARP,
            chroma,
            local,
        )
        return [event["chord"] for event in bars[0]["events"]]

    def test_weak_change_without_the_root_is_rejected_after_beat_twelve(self):
        chroma = np.zeros((12, 17))
        chroma[0, :] = 1.0
        self.assertEqual(self.bars_for(chroma), ["C"])

    def test_weak_change_keeps_the_root_the_chroma_supports(self):
        chroma = np.zeros((12, 17))
        chroma[0, :] = 1.0
        chroma[0, 13] = 0.1
        chroma[7, 13] = 1.0
        self.assertEqual(self.bars_for(chroma), ["C", "G", "C"])

    def test_confident_change_survives_a_quiet_root(self):
        from analyze import PITCHES_SHARP, btc_bars

        beats = np.arange(0.0, 8.01, 0.5)
        chroma = np.zeros((12, beats.size))
        chroma[0, :] = 1.0
        local = [{"label": "C", "confidence": 0.8} for _ in range(beats.size)]
        timeline = [
            {"start": 0.0, "end": 6.5, "chord": "C", "confidence": 0.9, "margin": 0.5},
            {"start": 6.5, "end": 7.0, "chord": "G", "confidence": 0.9, "margin": 0.4},
            {"start": 7.0, "end": 8.0, "chord": "C", "confidence": 0.9, "margin": 0.5},
        ]
        bars = [{"bar": 1, "start": 0.0, "end": 8.0, "chords": ["N"], "confidence": 0.0}]
        btc_bars(
            bars,
            timeline,
            beats,
            np.full(beats.size, -1),
            np.zeros(beats.size),
            PITCHES_SHARP,
            chroma,
            local,
        )
        self.assertEqual([event["chord"] for event in bars[0]["events"]], ["C", "G", "C"])


class SlashBassTest(unittest.TestCase):
    def bass(self, notes, confidence=0.95):
        from analyze import interval_bass

        beats = np.arange(0.0, 2.0, 0.5)
        return interval_bass(0.0, 2.0, beats, np.asarray(notes), np.full(len(notes), confidence))

    def test_sustained_pedal_is_kept(self):
        self.assertEqual(self.bass([5, 5, 5, 7]), 5)

    def test_passing_note_is_ignored(self):
        self.assertIsNone(self.bass([5, 7, 9, 11]))

    def test_unsure_bass_is_ignored(self):
        self.assertIsNone(self.bass([5, 5, 5, 5], confidence=0.85))


class HarteTest(unittest.TestCase):
    def test_conversions(self):
        from avaliar import to_harte

        self.assertEqual(to_harte("Gbm7"), "F#:min7")
        self.assertEqual(to_harte("G/F"), "G:maj/b7")
        self.assertEqual(to_harte("Fmaj7"), "F:maj7")
        self.assertEqual(to_harte("Gb7/B"), "F#:7/4")
        self.assertEqual(to_harte("N"), "N")
        self.assertEqual(to_harte("Q7"), "X")


class SavedChartsTest(unittest.TestCase):
    """A grade da página desenha ``events``; o texto escreve ``chords``. Os dois têm de bater."""

    def test_events_match_chords(self):
        # A fixture fictícia garante que o teste confere algo no CI, onde saida/ não existe.
        fixtures = sorted((CIFRAS / "tests" / "fixtures").glob("*.json"))
        self.assertTrue(fixtures, "sem fixture em tests/fixtures/")
        for path in fixtures + sorted((CIFRAS / "saida").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for bar in data.get("compassos") or []:
                events = bar.get("events") or []
                if not events:
                    continue
                with self.subTest(arquivo=path.name, compasso=bar.get("bar")):
                    self.assertEqual([event["chord"] for event in events], bar["chords"])
                    self.assertAlmostEqual(events[0]["start"], bar["start"], places=2)
                    self.assertAlmostEqual(events[-1]["end"], bar["end"], places=2)
                    for left, right in zip(events, events[1:]):
                        self.assertAlmostEqual(left["end"], right["start"], places=2)


if __name__ == "__main__":
    unittest.main()
