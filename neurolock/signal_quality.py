"""
signal_quality.py — Soude Signal Quality Heuristics
Per-channel signal quality estimation from the live ring buffer.
(Extraído de data_logger.py: esto es calidad de señal, no logging.)
"""

import logging

import numpy as np

from neurolock.brain_engine import N_CHANNELS, CHANNEL_NAMES

logger = logging.getLogger(__name__)


class ImpedanceChecker:
    """
    Estimates per-channel impedance quality from the live ring buffer by
    measuring signal variance.  High variance relative to typical EEG suggests
    good contact; near-zero variance suggests electrode off / bridge.

    This is a heuristic — not a substitute for the Unicorn's built-in
    impedance check, which should be run before each session.
    """

    GOOD_VARIANCE_UV2 = 10.0    # µV² lower bound for "live" channel
    BAD_VARIANCE_UV2  = 1e5     # µV² upper bound (above = noise / artifact)

    def check(self, snapshot: np.ndarray) -> list[dict]:
        """
        snapshot: (BUFFER_SAMPLES, N_CHANNELS) from RingBuffer.snapshot()
        Returns list of dicts per channel with keys: name, variance, status
        """
        results = []
        for ch_idx in range(N_CHANNELS):
            var = float(np.var(snapshot[:, ch_idx]))
            if var < self.GOOD_VARIANCE_UV2:
                status = "POOR"
            elif var > self.BAD_VARIANCE_UV2:
                status = "SATURATED"
            else:
                status = "OK"
            results.append({
                "name":     CHANNEL_NAMES[ch_idx],
                "variance": var,
                "status":   status,
            })
        return results
