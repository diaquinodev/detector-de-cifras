"""Testes das letras, com croma construído à mão e forma conhecida (nenhuma música real)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CIFRAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIFRAS))

from letters import find_letters, form_string, letter_phrases  # noqa: E402
from phrases import find_phrases  # noqa: E402

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11, "Bb": 10, "Eb": 3, "Ab": 8}
VERSE_1 = ["Am", "G", "F", "C"]
VERSE_2 = ["Am", "G", "F", "E"]
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


def chroma(chords: list[str | None], noise: float = 0.0) -> np.ndarray:
    columns = [np.zeros(12) if chord is None else triad(chord) for chord in chords]
    matrix = np.stack(columns, axis=1)
    if noise:
        matrix = matrix + noise * np.random.default_rng(7).random(matrix.shape)
    return matrix


def form_of(chords: list[str | None], noise: float = 0.0) -> str:
    matrix = chroma(chords, noise)
    return form_string(find_letters(matrix, find_phrases(matrix)))


class LetterPhrasesTest(unittest.TestCase):
    def test_equal_phrases_share_a_letter_and_different_ones_do_not(self):
        matrix = chroma(VERSE_1 * 2 + CHORUS + VERSE_1)
        letters = [phrase["letra"] for phrase in letter_phrases(matrix, find_phrases(matrix))]
        self.assertEqual(letters, ["a", "a", "b", "a"])

    def test_one_different_bar_in_four_is_another_letter(self):
        matrix = chroma(VERSE_1 + VERSE_2 + VERSE_1 + VERSE_2)
        letters = [phrase["letra"] for phrase in letter_phrases(matrix, find_phrases(matrix))]
        self.assertEqual(letters, ["a", "b", "a", "b"])

    def test_a_little_noise_does_not_split_a_letter(self):
        self.assertEqual(form_of(VERSE_1 * 2 + CHORUS * 2 + VERSE_1 * 2, noise=0.15), "A A B B A A")

    def test_silence_gets_no_letter(self):
        matrix = chroma([None, None, *VERSE_1 * 2])
        phrases = letter_phrases(matrix, find_phrases(matrix))
        self.assertEqual([phrase["letra"] for phrase in phrases], [None, "a", "a"])


class KnownFormsTest(unittest.TestCase):
    def test_verse_and_chorus(self):
        verse = VERSE_1 + VERSE_2
        song = verse + verse + CHORUS * 2 + verse + CHORUS * 2
        blocks = find_letters(chroma(song), find_phrases(chroma(song)))
        # O verso são duas frases que nunca se separam: um bloco de 8 compassos.
        self.assertEqual(form_string(blocks), "A A B B A B B")
        self.assertEqual(blocks[0]["compassos"], 8)
        self.assertEqual(blocks[0]["frases"], ["a", "b"])
        self.assertEqual(blocks[2]["compassos"], 4)

    def test_aaba_of_32_bars(self):
        a_part = VERSE_1 + VERSE_2
        b_part = BRIDGE + CHORUS
        song = a_part + a_part + b_part + a_part
        blocks = find_letters(chroma(song), find_phrases(chroma(song)))
        self.assertEqual(form_string(blocks), "A A B C A")
        self.assertEqual([block["compassos"] for block in blocks], [8, 8, 4, 4, 8])

    def test_twelve_bar_blues(self):
        blues = ["C", "C", "C", "C", "F", "F", "C", "C", "G", "F", "C", "G"]
        blocks = find_letters(chroma(blues * 3), find_phrases(chroma(blues * 3)))
        self.assertEqual(form_string(blocks), "A A A")
        self.assertEqual([block["compassos"] for block in blocks], [12, 12, 12])

    def test_a_song_with_no_repetition_gets_one_letter_per_phrase(self):
        self.assertEqual(form_of(VERSE_1 + CHORUS + BRIDGE + ["C", "Eb", "Ab", "D"]), "A B C D")

    def test_the_same_part_a_tone_up_is_a_new_letter_marked_as_modulation(self):
        up = ["Bm", "A", "G", "D"]  # VERSE_1 um tom acima
        song = VERSE_1 * 2 + CHORUS * 2 + up * 2
        blocks = find_letters(chroma(song), find_phrases(chroma(song)))
        self.assertEqual(form_string(blocks), "A A B B C C")
        self.assertEqual(blocks[4]["modulacao"], {"de": "A", "semitons": 2})
        self.assertNotIn("modulacao", blocks[0])
        self.assertNotIn("modulacao", blocks[2])

    def test_blocks_cover_every_bar_once(self):
        song = [None, *VERSE_1 * 2, *CHORUS, "G", *VERSE_1, *BRIDGE[:2]]
        matrix = chroma(song)
        blocks = find_letters(matrix, find_phrases(matrix))
        position = 1
        for block in blocks:
            self.assertEqual(block["compasso"], position)
            position += block["compassos"]
        self.assertEqual(position - 1, len(song))

    def test_empty_input_does_not_break(self):
        self.assertEqual(find_letters(np.zeros((12, 0)), []), [])


if __name__ == "__main__":
    unittest.main()
