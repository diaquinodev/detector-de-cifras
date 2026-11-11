"""Testes das frases, com croma construído à mão (nenhuma música real)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from phrases import find_phrases, novelty, phrase_starts  # noqa: E402

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11, "Bb": 10}
VERSE = ["Am", "G", "F", "C"]
CHORUS = ["D", "Bb", "E", "B"]
BRIDGE = ["Dm", "Em", "Bm", "Gm"]


def triad(name: str) -> np.ndarray:
    minor = name.endswith("m")
    root = ROOTS[name[:-1] if minor else name]
    vector = np.zeros(12, dtype=float)
    vector[root] = 1.0
    vector[(root + (3 if minor else 4)) % 12] = 0.8
    vector[(root + 7) % 12] = 0.9
    return vector


def chroma(chords: list[str | None]) -> np.ndarray:
    """Um compasso por acorde; ``None`` é compasso sem som."""
    columns = [np.zeros(12) if chord is None else triad(chord) for chord in chords]
    return np.stack(columns, axis=1)


def shape(phrases: list[dict]) -> list[tuple[int, int, str | None]]:
    return [(p["compasso"], p["compassos"], p["excecao"]) for p in phrases]


class NoveltyTest(unittest.TestCase):
    def test_a_repeating_cycle_has_no_novelty_anywhere(self):
        self.assertEqual(float(novelty(chroma(VERSE * 6)).max()), 0.0)

    def test_the_peak_is_where_one_cycle_gives_way_to_another(self):
        peaks = novelty(chroma(VERSE * 2 + CHORUS * 2 + VERSE * 2))
        self.assertEqual(list(np.flatnonzero(peaks)), [8, 16])


class FindPhrasesTest(unittest.TestCase):
    def test_a_regular_song_is_cut_every_four_bars(self):
        phrases = find_phrases(chroma(VERSE * 2 + CHORUS * 2 + VERSE * 2 + CHORUS * 2))
        self.assertEqual(phrase_starts(phrases), [0, 4, 8, 12, 16, 20, 24, 28])
        self.assertTrue(all(phrase["excecao"] is None for phrase in phrases))

    def test_a_song_with_no_change_at_all_still_gets_the_regular_grid(self):
        phrases = find_phrases(chroma(VERSE * 4))
        self.assertEqual(shape(phrases), [(1, 4, None), (5, 4, None), (9, 4, None), (13, 4, None)])

    def test_a_pickup_bar_is_an_explicit_exception(self):
        song = ["C", *VERSE * 2, *CHORUS * 2, *VERSE * 2, *CHORUS * 2]
        phrases = find_phrases(chroma(song))
        self.assertEqual(shape(phrases)[0], (1, 1, "anacruse"))
        self.assertEqual(phrase_starts(phrases)[1:], [1, 5, 9, 13, 17, 21, 25, 29])
        self.assertEqual(sum(phrase["excecao"] is not None for phrase in phrases), 1)

    def test_an_extra_bar_in_the_middle_is_an_explicit_exception(self):
        song = [*VERSE * 2, *CHORUS * 2, "G", *VERSE * 2, *CHORUS * 2, *BRIDGE * 2]
        phrases = find_phrases(chroma(song))
        exceptions = [item for item in shape(phrases) if item[2] is not None]
        self.assertEqual(exceptions, [(13, 5, "compasso extra")])
        self.assertEqual(phrase_starts(phrases), [0, 4, 8, 12, 17, 21, 25, 29, 33, 37])

    def test_one_change_off_the_grid_is_not_enough_for_an_exception(self):
        # Uma só mudança fora da grade pode ser ruído: a grade regular fica.
        song = [*VERSE * 2, "Am", *CHORUS * 2]
        phrases = find_phrases(chroma(song))
        self.assertEqual(
            shape(phrases),
            [(1, 4, None), (5, 4, None), (9, 4, None), (13, 4, None), (17, 1, "final incompleto")],
        )

    def test_a_cut_ending_costs_nothing(self):
        phrases = find_phrases(chroma(VERSE * 2 + CHORUS[:2]))
        self.assertEqual(shape(phrases), [(1, 4, None), (5, 4, None), (9, 2, "final incompleto")])

    def test_leading_silence_is_set_apart_and_the_grid_starts_at_the_music(self):
        phrases = find_phrases(chroma([None, None, *VERSE * 2, *CHORUS * 2]))
        self.assertEqual(shape(phrases)[0], (1, 2, "silêncio"))
        self.assertEqual(phrase_starts(phrases)[1:], [2, 6, 10, 14])

    def test_a_short_phrase_between_two_real_changes_stays_a_short_phrase(self):
        # Dois compassos entre duas mudanças de verdade não são virada da frase de antes.
        song = [*VERSE * 2, *CHORUS[:2], *BRIDGE * 2, *VERSE * 2, *CHORUS * 2]
        phrases = find_phrases(chroma(song))
        exceptions = [item for item in shape(phrases) if item[2] is not None]
        self.assertEqual(exceptions, [(9, 2, "frase curta")])
        self.assertEqual(phrase_starts(phrases)[:4], [0, 4, 8, 10])

    def test_trailing_silence_is_set_apart_like_leading_silence(self):
        phrases = find_phrases(chroma([*VERSE * 2, None, None]))
        self.assertEqual(shape(phrases), [(1, 4, None), (5, 4, None), (9, 2, "silêncio")])

    def test_silent_bars_in_the_middle_are_a_rest_inside_the_phrase(self):
        song = [*VERSE, None, None, None, None, *VERSE * 2, *CHORUS * 2]
        phrases = find_phrases(chroma(song))
        self.assertEqual(phrase_starts(phrases), [0, 4, 8, 12, 16, 20])
        self.assertTrue(all(phrase["excecao"] is None for phrase in phrases))

    def test_the_phrases_cover_every_bar_once(self):
        song = ["C", *VERSE * 3, "G", *CHORUS * 2, *BRIDGE, "D", "E"]
        phrases = find_phrases(chroma(song))
        self.assertEqual(sum(phrase["compassos"] for phrase in phrases), len(song))
        position = 1
        for phrase in phrases:
            self.assertEqual(phrase["compasso"], position)
            position += phrase["compassos"]

    def test_empty_and_tiny_inputs_do_not_break(self):
        self.assertEqual(find_phrases(np.zeros((12, 0))), [])
        self.assertEqual(shape(find_phrases(chroma(["C", "G"]))), [(1, 2, "final incompleto")])
        self.assertEqual(shape(find_phrases(chroma([None, None]))), [(1, 2, "silêncio")])


if __name__ == "__main__":
    unittest.main()
