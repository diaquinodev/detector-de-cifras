"""Reconhecimento de acordes com o BTC (ISMIR 2019).

O modelo decide a harmonia. A grade de tempos e compassos continua no
pipeline de pulso. O carregamento é local e usa um checkpoint sanitizado que o
PyTorch abre com ``weights_only=True``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parent / "vendor" / "btc-chord"
CHECKPOINT = MODEL_ROOT / "btc_model_large_voca_safe.pt"

_model = None


def available() -> bool:
    return CHECKPOINT.is_file()


def model():
    global _model
    if _model is None:
        if not available():
            raise FileNotFoundError(f"Pesos BTC ausentes: {CHECKPOINT}")
        if str(MODEL_ROOT) not in sys.path:
            sys.path.insert(0, str(MODEL_ROOT))
        from modeling_btc import BTCForChordRecognition

        _model = BTCForChordRecognition.from_pretrained(
            MODEL_ROOT,
            device="cpu",
            large_voca=True,
        )
    return _model


def recognize(audio: Path | np.ndarray, tuning_cents: float = 0.0) -> list[dict]:
    return model().predict(audio, tuning_cents=tuning_cents)
