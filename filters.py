"""
filters.py

Construcción y aplicación de filtros digitales (Butterworth + Notch) en formato SOS.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from brain_engine import SAMPLE_RATE

# Filtros individuales

def build_bandpass_sos(
    low: float = 1.0,
    high: float = 10.0,
    order: int = 4,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

def build_notch_sos(
    freq: float = 60.0,
    quality: float = 30.0,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    b, a = iirnotch(freq / (fs / 2.0), quality)
    return tf2sos(b, a)


def build_passthought_sos(
    low: float = 8.0,
    high: float = 30.0,
    order: int = 4,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

# Cadenas preconfiguradas

def build_p300_chain(
    notch_freq: float = 60.0,
    bp_low: float = 1.0,
    bp_high: float = 10.0,
) -> list[np.ndarray]:

    return [
        build_notch_sos(freq=notch_freq),
        build_bandpass_sos(low=bp_low, high=bp_high),
    ]

def build_mu_beta_chain(notch_freq: float = 60.0) -> list[np.ndarray]:
    return [
        build_notch_sos(freq=notch_freq),
        build_passthought_sos(),
    ]

# Aplicacion

def apply_filter_chain(
    data: np.ndarray,
    sos_chain: list[np.ndarray],
    axis: int = 0,
) -> np.ndarray:
    
    out = data.astype(np.float64)
    for sos in sos_chain:
        out = sosfilt(sos, out, axis=axis)
    return out