"""CQT feature extraction + chord-index maps for BTC.

Trimmed from the original utils/mir_eval_modules.py — the `mir_eval` dependency
(only used for evaluation metrics, not inference) is removed.
"""
from __future__ import annotations
import numpy as np
import librosa

# --- chord index -> label maps -------------------------------------------------
idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min',
             'F', 'F:min', 'F#', 'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min',
             'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7',
                '7', 'dim7', 'hdim7', 'sus2', 'sus4']


def idx2voca_chord():
    """169-index large-vocabulary map (168 root×quality + 'X' + 'N')."""
    m = {169: 'N', 168: 'X'}
    for i in range(168):
        root = root_list[i // 14]
        quality = quality_list[i % 14]
        m[i] = root if (i % 14) == 1 else root + ':' + quality
    return m


def audio_to_features(audio_path_or_array, sr_target=22050, inst_len=10.0,
                      n_bins=144, bins_per_octave=24, hop_length=2048, tuning=0.0):
    """Load audio (path or 1-D np array) and compute the log-CQT feature matrix.

    Returns (feature [n_bins, T], feature_per_frame_seconds).
    Mirrors the original audio_file_to_features windowing exactly.
    ``tuning`` is the offset in fractions of one CQT bin; the released weights
    were trained with 0.0.
    """
    if isinstance(audio_path_or_array, (str, bytes)) or hasattr(audio_path_or_array, "__fspath__"):
        wav, sr = librosa.load(str(audio_path_or_array), sr=sr_target, mono=True)
    else:
        wav = np.asarray(audio_path_or_array, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        sr = sr_target

    def _cqt(y):
        return librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave,
                           hop_length=hop_length, tuning=tuning)

    win = int(sr_target * inst_len)
    feature = None
    cur = 0
    while len(wav) > cur + win:
        tmp = _cqt(wav[cur:cur + win])
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
        cur += win
    tmp = _cqt(wav[cur:])
    feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)
    # timestep is fixed at 108 for BTC; feature_per_second = inst_len / timestep
    return feature
