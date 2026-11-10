"""Limiares e tabelas do pipeline. Os valores são os mesmos de antes; só têm nome."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 22050
HOP = 512
N_FFT = 4096
MAX_AUDIO_SECONDS = 20 * 60
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac"}
SKIPPED_AUDIO_DIRS = {"moises-app-x64", "moises-setup-unpacked", "rea", "node_modules", "cifras"}

PITCHES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCHES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PC_NAMES_SHARP = PITCHES_SHARP
PC_NAMES_FLAT = PITCHES_FLAT
PITCH_TO_PC = {name: index for index, name in enumerate(PITCHES_SHARP)}
PITCH_TO_PC.update({name: index for index, name in enumerate(PITCHES_FLAT)})
LETTERS = "CDEFGAB"
LETTER_PCS = (0, 2, 4, 5, 7, 9, 11)
DEGREE_OF_INTERVAL = (0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6)
MAJOR_TONIC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_TONIC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
UNUSUAL_NAMES = {"E#", "B#", "Cb", "Fb"}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
METER_CANDIDATES = (2, 3, 4, 6)
CHORD_KINDS = (
    ("", (0, 4, 7)),
    ("m", (0, 3, 7)),
    ("7", (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)),
    ("m7", (0, 3, 7, 10)),
    ("sus4", (0, 5, 7)),
)
SPEC_LOW = 36
SPEC_HIGH = 96
NOTE_LOW = 36
NOTE_HIGH = 84
HARMONY_MIDI_MIN = 48
HARMONICS = tuple(
    (int(round(12 * np.log2(harmonic))), (0.35 if harmonic == 7 else 1.0) / harmonic)
    for harmonic in range(1, 9)
)
A4_HZ = 440.0
A4_MIDI = 69.0
EPS = 1e-8

ONSET_LOG_GAIN = 20
ONSET_WEIGHT_HIGH = 1.4
ONSET_WEIGHT_LOW = 0.6
ONSET_SMOOTH_FRAMES = 5
TEMPO_MAX_BPM = 180
TEMPO_MIN_BPM = 55
DOUBLE_LAG_RATIO = 0.92
DOUBLE_LAG_BPM = 110
BEAT_TIGHTNESS = 80.0
BEAT_LAG_WIDE = 1.6
BEAT_LAG_NEAR = 0.6
MIN_BEATS_FOR_GRID = 8
MAX_BEAT_FRAMES = 10000
COLLAPSE_TEMPO_BPM = 110
COLLAPSE_MIN_BEATS = 16
COLLAPSE_ACCENT_RATIO = 1.25

SEMITONE_FREQ_FLOOR = 30.0
METER_BASS_WEIGHT = 0.65
METER_CHANGE_WEIGHT = 0.35
METER_MIN_HITS = 4
COMPOUND_DUPLE_RATIO = 1.15
COMPOUND_WHOLE_RATIO = 0.35
QUIET_BAR_GATE = 0.35
HARMONY_PHASE_RATIO = 1.15
HARMONY_PHASE_EXTRA = 4
BASS_FIR_TAPS = 401
BASS_FIR_HZ = 150
BASS_ENERGY_WINDOW_S = 0.05
BASS_BAND_LOW = 40
BASS_BAND_HIGH = 180
BAR_OFFSET_SMOOTH_S = 0.02
BAR_OFFSET_RADIUS_S = 0.07
BAR_OFFSET_SKIP = 4
BAR_OFFSET_MIN_ERRORS = 8
BAR_OFFSET_LEAD_S = -0.08
BAR_OFFSET_LAG_S = 0.04
BAR_OFFSET_PRINT_MS = 15

BASS_SPELL_CONFIDENCE = 0.85
KEY_BASS_SCORE_RATIO = 0.92
KEY_FIFTH_SCORE_RATIO = 0.85
PRESENCE_PEAK_FRACTION = 0.55
TUNING_CONFIDENCE = 0.9
TUNING_MIN_SAMPLES = 6
TUNING_CENTS_CLIP = 35

SEVENTH_TONE_WEIGHT = 0.55
BASS_SPELL_PEAK_FRACTION = 0.65
BASS_SPELL_CLOSE_RATIO = 0.9
BASS_SPELL_SCORE = 0.72
RELATIVE_M7_PEAK_FRACTION = 0.5
SUS4_MARGIN = 1.12
ROOTLESS_SUS4_FRACTION = 0.45
HALF_BAR_SEVENTH_RATIO = 1.04
HALF_BAR_SEVENTH_CHROMA_RATIO = 1.12
HALF_BAR_SEVENTH_PEAK_FRACTION = 0.4
QUIET_CHORD_GATE = 0.24
CLOSE_SCORE_RATIO = 1.12
NEAR_SCORE_RATIO = 0.9
SLASH_SEVENTH_CHROMA_FRACTION = 0.65
BASS_MIN_NOTES = 2
CHORD_QUALITY_SUFFIXES = ("maj7", "m7", "sus4", "m", "7")

BASS_SPAN_MIN_S = 0.2
BASS_WINDOW_START = 0.14
BASS_WINDOW_END = 0.9
FUNDAMENTAL_MIN_SAMPLES = 256
FUNDAMENTAL_MIN_AMP = 1e-4
FUNDAMENTAL_MAX_HZ = 180
FUNDAMENTAL_MIN_HZ = 40
FUNDAMENTAL_PEAK = 0.45
FUNDAMENTAL_WEAK = 0.4
OCTAVE_LAG_RATIO = 0.72
OCTAVE_STRENGTH_RATIO = 0.82

BTC_QUALITIES = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "6",
    "min7": "m7",
    "minmaj7": "mMaj7",
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}
BTC_INTERVALS = {
    "": {0, 4, 7},
    "m": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "m6": {0, 3, 7, 9},
    "6": {0, 4, 7, 9},
    "m7": {0, 3, 7, 10},
    "mMaj7": {0, 3, 7, 11},
    "maj7": {0, 4, 7, 11},
    "7": {0, 4, 7, 10},
    "dim7": {0, 3, 6, 9},
    "m7b5": {0, 3, 6, 10},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
}
SLASH_BASS_CONFIDENCE = 0.90
SLASH_BASS_SHARE = 0.5
BASS_INTERVAL_PAD_S = 0.08
CHORD_ONSET_SLACK = 0.12
BTC_MIN_OCCUPANCY = 0.52
BTC_WEAK_OCCUPANCY = 0.68
BTC_WEAK_CONFIDENCE = 0.50
BTC_WEAK_ROOT_SUPPORT = 0.62
BTC_VERY_WEAK_CONFIDENCE = 0.22
BTC_VERY_WEAK_MARGIN = 0.075
BTC_VERY_WEAK_ROOT_SUPPORT = 0.38
BTC_WEAK_CONFIDENCE_CAP = 0.35
BTC_WEAK_MARGIN_CAP = 0.08
BTC_FLICKER_CONFIDENCE = 0.34
BTC_FLICKER_MARGIN = 0.12
EVENT_MERGE_SLACK = 0.02
BTC_KEY_WINDOW_S = 60.0
BTC_ROOT_WEIGHT = 1.5
NATURAL_SEVEN_RATIO = 1.35
BTC_KEY_ABOVE_RATIO = 0.92
BTC_KEY_FIFTH_RATIO = 0.90

PRESENCE_BAND = (1800, 4500)
CHEST_BAND = (250, 700)
ENVELOPE_WINDOW_S = 0.08
SINGING_FLOOR_FRACTION = 0.08
SINGING_PERCENTILE = 25
SINGING_PERCENTILE_GAIN = 1.5
FINAL_REST_BARS = 3
INSTRUMENTAL_PERCENTILE = 20
VOICE_REENTRY_QUIET = 8
CHEST_LOOKBACK = 8
CHEST_MIN_WINDOW = 4
CHEST_PUSH_RATIO = 1.55
CHEST_HOLD_RATIO = 1.35
PUSH_LAND_BARS = 7
PUSH_LAND_MIN = 3
PUSH_LAND_HITS = 4
FALLBACK_VOCAB_BARS = 12
FALLBACK_CHORUS_MIN = 6
CHORUS_LOOKAHEAD = 6

MAJOR_FUNCTIONS = {
    (0, "maj"): ("I", "T"),
    (2, "min"): ("ii", "SD"),
    (4, "min"): ("iii", "T"),
    (5, "maj"): ("IV", "SD"),
    (7, "maj"): ("V", "D"),
    (7, "dom"): ("V7", "D"),
    (9, "min"): ("vi", "T"),
    (11, "dim"): ("vii°", "D"),
    (3, "maj"): ("bIII", "T"),
    (8, "maj"): ("bVI", "SD"),
    (8, "dom"): ("bVI7", "SD"),
    (10, "maj"): ("bVII", "SD"),
    (10, "dom"): ("bVII7", "SD"),
    (5, "min"): ("iv", "SD"),
    (7, "min"): ("v", "SD"),
    (2, "dom"): ("V7/V", "D2"),
    (2, "maj"): ("V/V", "D2"),
    (4, "dom"): ("V7/vi", "D2"),
    (4, "maj"): ("V/vi", "D2"),
    (9, "dom"): ("V7/ii", "D2"),
    (9, "maj"): ("V/ii", "D2"),
    (0, "dom"): ("V7/IV", "D2"),
}
DIATONIC_MAJOR = {
    (0, "maj"),
    (2, "min"),
    (4, "min"),
    (5, "maj"),
    (7, "maj"),
    (7, "dom"),
    (9, "min"),
    (11, "dim"),
}
DIATONIC_MINOR = {
    (0, "min"),
    (2, "dim"),
    (3, "maj"),
    (5, "min"),
    (7, "min"),
    (7, "maj"),
    (7, "dom"),
    (8, "maj"),
    (10, "maj"),
}
KEY_FIT_MARGIN = 0.25
INFER_KEY_RATIO = 1.12
CYCLE_REPEAT_SIM = 0.72
PICKUP_SIM_DELTA = 0.06
CONSECUTIVE_SIM = 0.68
FAMILY_SIM = 0.70
STRONG_SLOT_SIM = 0.85
OUTLIER_SIM = 0.55
OUTLIER_CONFIDENCE = 0.82
REPEAT_MIN_BARS = 4
REPEAT_MAX_BARS = 32
REPEAT_LONG_PHRASE = 8
REPEAT_SHORT_TAIL = 1
REPEAT_LONG_TAIL = 2
REPEAT_CORE_MEAN = 0.90
REPEAT_BAR_SIM = 0.88
REPEAT_MIN_MATCH_FRACTION = 0.75
REPEAT_MIN_MATCH_BARS = 3
REPEAT_SHIFT_MARGIN = 0.03
REPEAT_ALIGN_MARGIN = 0.03
REPEAT_CHORD_FIT_MARGIN = 0.08
REPEAT_CHORD_FIT_FLOOR = 0.25
VOCAL_PEAK_FRACTION = 0.12
VOCAL_PERCENTILE = 28
VOCAL_PERCENTILE_GAIN = 1.30
VOCAL_NEXT_RATIO = 0.85
CHEST_ONSET_CANDIDATES = (4, 6, 8)
CHEST_ONSET_RATIO = 1.25
ONSET_SNAP_BARS = 2
MODULATION_MIN_BARS = 24
MODULATION_SEARCH_FRACTION = 0.45
MODULATION_TAIL_BARS = 6
MODULATION_SHIFT_CANDIDATES = (1, 2, -1, 5, 7)
MODULATION_NEW_FIT_RATIO = 1.85
MODULATION_NEW_FIT_FLOOR = 5.0
TONIC_DEGREES = {0, 5, 7}
BRIDGE_FRACTION = 0.60
SHORT_SECTION_SPAN = 2
