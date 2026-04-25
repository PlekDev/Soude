"""
filters.py — Neuro-Lock Filter Construction
Builds the SOS filter chain used by signal_processing.py.
Sub-team 2 (Signal/Math) owns this file.
"""

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos

from brain_engine import SAMPLE_RATE


def build_bandpass_sos(low: float = 1.0, high: float = 10.0, order: int = 4) -> np.ndarray:
    """4th-order Butterworth bandpass as second-order sections."""
    nyq = SAMPLE_RATE / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sos


def build_notch_sos(freq: float = 60.0, quality: float = 30.0) -> np.ndarray:
    """IIR notch filter at `freq` Hz (use 50 Hz for Europe/Mexico)."""
    b, a = iirnotch(freq / (SAMPLE_RATE / 2.0), quality)
    return tf2sos(b, a)


def build_p300_chain() -> list[np.ndarray]:
    """
    Returns the ordered list of SOS filters applied to every epoch.
    Currently: [bandpass 1-10 Hz, notch 60 Hz].
    """
    return [
        build_bandpass_sos(low=1.0, high=10.0, order=4),
        build_notch_sos(freq=60.0, quality=30.0),
    ]
