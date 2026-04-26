"""
data_logger.py — Soude Session Logger
Records raw epochs, markers, and auth results for debugging and demo replay.
Sub-team 4 (Integration/Demo) owns this file.
"""

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from brain_engine import StimulusMarker, SAMPLE_RATE, N_CHANNELS
from Fase1.signal_processing import AuthResult

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SessionLogger:
    """
    Writes session data to a timestamped folder:
        logs/<session_id>/
            markers.csv       — stimulus event log
            epochs.npz        — raw epoch array per marker
            auth_result.json  — final auth decision
            summary.txt       — human-readable summary
    """

    def __init__(self, session_id: Optional[str] = None):
        sid = session_id or _session_id()
        self._dir = LOG_DIR / sid
        self._dir.mkdir(parents=True, exist_ok=True)

        self._marker_rows: list[dict] = []
        self._epochs: dict[str, np.ndarray] = {}   # key: "epoch_<idx>"
        self._auth_result: Optional[AuthResult] = None
        self._erp_data: Optional[dict] = None

        logger.info("Session log directory: %s", self._dir)

    # ── Markers ────────────────────────────────────────────────────────────────

    def log_marker(self, marker: StimulusMarker, epoch: Optional[np.ndarray]) -> None:
        idx = len(self._marker_rows)
        row = {
            "index":        idx,
            "image_id":     marker.image_id,
            "is_target":    int(marker.is_target),
            "buffer_index": marker.buffer_index,
            "timestamp":    f"{marker.timestamp:.6f}",
        }
        self._marker_rows.append(row)
        if epoch is not None:
            self._epochs[f"epoch_{idx:04d}"] = epoch

    def log_markers_bulk(
        self,
        markers: list[StimulusMarker],
        epochs:  Optional[list[Optional[np.ndarray]]] = None,
    ) -> None:
        for i, m in enumerate(markers):
            ep = epochs[i] if (epochs and i < len(epochs)) else None
            self.log_marker(m, ep)

    # ── Auth Result ────────────────────────────────────────────────────────────

    def log_auth_result(self, result: AuthResult, erp_data: Optional[dict] = None) -> None:
        self._auth_result = result
        self._erp_data    = erp_data

    # ── Flush to Disk ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write all buffered data to disk."""
        self._write_markers_csv()
        self._write_epochs_npz()
        self._write_auth_json()
        self._write_summary()
        logger.info("Session flushed to %s", self._dir)

    def _write_markers_csv(self) -> None:
        path = self._dir / "markers.csv"
        if not self._marker_rows:
            return
        fields = list(self._marker_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._marker_rows)

    def _write_epochs_npz(self) -> None:
        if not self._epochs:
            return
        path = self._dir / "epochs.npz"
        np.savez_compressed(str(path), **self._epochs)

    def _write_auth_json(self) -> None:
        if self._auth_result is None:
            return
        payload = {
            "granted":          self._auth_result.granted,
            "target_peak_uv":   self._auth_result.target_peak_uv,
            "nontarget_peak_uv": self._auth_result.nontarget_peak_uv,
            "snr_db":           self._auth_result.snr_db,
            "message":          self._auth_result.message,
            "erp_data":         self._erp_data,
        }
        path = self._dir / "auth_result.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _write_summary(self) -> None:
        path = self._dir / "summary.txt"
        lines = [
            "=" * 60,
            f"SOUDE SESSION — {self._dir.name}",
            "=" * 60,
            f"Total stimuli:   {len(self._marker_rows)}",
            f"Epochs saved:    {len(self._epochs)}",
        ]
        if self._auth_result:
            lines += [
                "",
                f"AUTH RESULT:     {'GRANTED ✓' if self._auth_result.granted else 'DENIED ✗'}",
                f"Target P300:     {self._auth_result.target_peak_uv:.3f} µV",
                f"Non-target P300: {self._auth_result.nontarget_peak_uv:.3f} µV",
                f"SNR:             {self._auth_result.snr_db:.2f} dB",
                f"Message:         {self._auth_result.message}",
            ]
        lines.append("=" * 60)
        path.write_text("\n".join(lines), encoding="utf-8")

    @property
    def session_dir(self) -> Path:
        return self._dir


# ── Impedance Checker ──────────────────────────────────────────────────────────

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

    CHANNEL_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

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
                "name":     self.CHANNEL_NAMES[ch_idx],
                "variance": var,
                "status":   status,
            })
        return results
