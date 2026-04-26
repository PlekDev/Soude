"""
signal_processing.py — Soude Signal Processing Pipeline
Real-time filtering, epoch extraction, P300 detection, and authentication.
Sub-team 2 (Signal/Math) owns this file.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import sosfiltfilt, sosfilt_zi, sosfilt
from filters import *

from brain_engine import (
    SAMPLE_RATE,
    N_CHANNELS,
    P300_CHANNELS,
    BrainEngine,
    StimulusMarker,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EPOCH_DURATION_S  = 0.800     # 800 ms post-stimulus window
BASELINE_DURATION_S = 0.100   # 100 ms pre-stimulus baseline correction
EPOCH_SAMPLES     = int(EPOCH_DURATION_S * SAMPLE_RATE)
BASELINE_SAMPLES  = int(BASELINE_DURATION_S * SAMPLE_RATE)

# P300 detection window (samples within epoch)
P300_ONSET_S   = 0.250
P300_OFFSET_S  = 0.500
P300_ONSET     = int(P300_ONSET_S  * SAMPLE_RATE)
P300_OFFSET    = int(P300_OFFSET_S * SAMPLE_RATE)

# Authentication threshold — mean(target − non-target) across 250–500 ms window.
# With visually equated stimuli, this mean is ≈ 0 µV for non-attending subjects
# and ≈ 3–8 µV for a genuine cognitive P300.
AUTH_THRESHOLD_UV = 1.5    # µV — a genuine P300 mean should comfortably exceed this.

# Minimum required epochs per class for reliable averaging
MIN_EPOCHS = 5             # Need at least 5 clean epochs per class for a reliable average

@dataclass
class AuthResult:
    granted: bool
    target_peak_uv: float
    nontarget_peak_uv: float
    snr_db: float
    message: str


# ── Filter Construction ────────────────────────────────────────────────────────

# Pre-build filters once at module load (cheap, avoids repeated construction)
_P300_CHAIN = build_p300_chain()

# ── Stateful Online Filter ─────────────────────────────────────────────────────

class OnlineFilter:
    """
    Applies the P300 filter chain using stateful SOS filtering so it can process
    streaming chunks without phase discontinuities.
    Operates on shape (n_samples, N_CHANNELS).
    """

    def __init__(self):
        # Creamos una lista de estados (zi) para cada filtro en la cadena
        self._zis = []
        for sos in _P300_CHAIN:
            zi_1ch = sosfilt_zi(sos)  # (sections, 2)
            # Expand to (sections, N_CHANNELS, 2)
            zi_expanded = np.repeat(zi_1ch[:, np.newaxis, :], N_CHANNELS, axis=1)
            self._zis.append(zi_expanded)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        chunk: (n_samples, N_CHANNELS)
        Returns filtered chunk of the same shape.
        """
        x = chunk.T   # (N_CHANNELS, n_samples)
        
        # Aplicamos cada filtro de la cadena secuencialmente, guardando su estado
        for i, sos in enumerate(_P300_CHAIN):
            x, self._zis[i] = sosfilt(sos, x, zi=self._zis[i])
            
        return x.T   # back to (n_samples, N_CHANNELS)

# ── Offline / Epoch Filter ─────────────────────────────────────────────────────

def filter_epoch(epoch: np.ndarray) -> np.ndarray:
    """
    Zero-phase filter a single epoch (n_samples, N_CHANNELS).
    Uses sosfiltfilt (forward-backward) — suitable for offline per-epoch processing.
    Safe for short epochs (>= 3× filter order satisfied at 200 samples).
    """
    filtered = epoch.T
    
    # Aplicamos cada filtro de la cadena hacia adelante y hacia atrás (zero-phase)
    for sos in _P300_CHAIN:
        filtered = sosfiltfilt(sos, filtered)
        
    return filtered.T


# ── Baseline Correction ────────────────────────────────────────────────────────

def baseline_correct(epoch: np.ndarray, baseline_samples: int = BASELINE_SAMPLES) -> np.ndarray:
    """
    Subtract the mean of the first `baseline_samples` from the entire epoch.
    epoch shape: (n_samples, N_CHANNELS)
    """
    if baseline_samples <= 0 or baseline_samples >= len(epoch):
        return epoch
    baseline_mean = epoch[:baseline_samples].mean(axis=0, keepdims=True)
    return epoch - baseline_mean


# ── Artifact Rejection ─────────────────────────────────────────────────────────

def is_artifact(epoch: np.ndarray, threshold_uv: float = 100.0) -> bool:
    """
    Reject epoch if any channel exceeds threshold_uv peak-to-peak.
    After bandpass filtering (0.5–30 Hz), genuine EEG should be 10–100 µV p-p.
    Artifacts from poor contact, cable movement, or mains interference appear
    as bursts well above 100 µV and must be rejected.
    Typical blink artifact: >150 µV on frontal channels.
    """
    pp = epoch.max(axis=0) - epoch.min(axis=0)
    worst = float(pp.max())
    if worst > threshold_uv:
        logger.warning("Artifact rejected: peak-to-peak %.1f µV > %.0f µV threshold", worst, threshold_uv)
        return True
    return False


# ── Epoch Extractor ────────────────────────────────────────────────────────────

class EpochExtractor:
    """
    Extracts, filters, and baseline-corrects epochs from the BrainEngine buffer.
    """

    def __init__(self, engine: BrainEngine):
        self._engine = engine

    def extract(
        self,
        marker: StimulusMarker,
        pre_samples: int = BASELINE_SAMPLES,
        reject_artifacts: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Pull EPOCH_SAMPLES starting at (marker.buffer_index - pre_samples),
        apply filtering + baseline correction.

        Returns (EPOCH_SAMPLES, N_CHANNELS) or None if unavailable / artifact.
        """
        total_samples = EPOCH_SAMPLES + pre_samples
        start_index   = marker.buffer_index - pre_samples

        # Guard: if the start falls before the buffer has any data, skip
        if start_index < 0:
            logger.warning("Epoch id=%d skipped: start_index=%d < 0 (device not warmed up).", marker.image_id, start_index)
            return None

        # Guard: ensure the full epoch has been acquired (post-stimulus data exists)
        end_index = marker.buffer_index + EPOCH_SAMPLES
        if end_index > self._engine.buffer.total_written:
            logger.warning("Epoch id=%d not yet complete: need %d samples, have %d.", marker.image_id, end_index, self._engine.buffer.total_written)
            return None

        raw = self._engine.buffer.read_from(start_index, total_samples)
        if raw is None:
            logger.warning("Epoch id=%d unavailable (buffer overrun).", marker.image_id)
            return None

        filtered   = filter_epoch(raw)
        corrected  = baseline_correct(filtered, baseline_samples=pre_samples)
        epoch      = corrected[pre_samples:]   # discard pre-stimulus portion

        if reject_artifacts and is_artifact(epoch):
            logger.debug("Epoch for id=%d rejected (artifact).", marker.image_id)
            return None

        return epoch   # (EPOCH_SAMPLES, N_CHANNELS)


# ── Signal Averager ────────────────────────────────────────────────────────────

class SignalAverager:
    """
    Accumulates epochs for target and non-target classes and computes the
    grand-average ERP for P300 detection.
    """

    def __init__(self):
        self._target_epochs:     list[np.ndarray] = []
        self._nontarget_epochs:  list[np.ndarray] = []

    def reset(self) -> None:
        self._target_epochs.clear()
        self._nontarget_epochs.clear()

    def add_epoch(self, epoch: np.ndarray, is_target: bool) -> None:
        if is_target:
            self._target_epochs.append(epoch)
        else:
            self._nontarget_epochs.append(epoch)

    @property
    def n_target(self) -> int:
        return len(self._target_epochs)

    @property
    def n_nontarget(self) -> int:
        return len(self._nontarget_epochs)

    def target_average(self) -> Optional[np.ndarray]:
        """Returns (EPOCH_SAMPLES, N_CHANNELS) grand average or None."""
        if not self._target_epochs:
            return None
        return np.mean(self._target_epochs, axis=0)

    def nontarget_average(self) -> Optional[np.ndarray]:
        if not self._nontarget_epochs:
            return None
        return np.mean(self._nontarget_epochs, axis=0)

    def target_sem(self) -> Optional[np.ndarray]:
        """Standard error of the mean across target epochs."""
        n = len(self._target_epochs)
        if n < 2:
            return None
        return np.std(self._target_epochs, axis=0) / np.sqrt(n)


# ── P300 Detector ──────────────────────────────────────────────────────────────

def compute_p300_peak(average: np.ndarray) -> float:
    """
    Given a grand-average epoch (EPOCH_SAMPLES, N_CHANNELS),
    return the mean amplitude (µV) across P300_CHANNELS in the 250–500 ms window.
    """
    window = average[P300_ONSET:P300_OFFSET, :]
    p300_channels_data = window[:, P300_CHANNELS]   # (window_samples, 3)
    return float(p300_channels_data.mean())


def compute_snr_db(target_peak: float, nontarget_peak: float, noise_std: float) -> float:
    """Signal-to-noise ratio in dB.  noise_std from non-target variability."""
    signal_power = (target_peak - nontarget_peak) ** 2
    noise_power  = max(noise_std ** 2, 1e-9)
    return float(10.0 * np.log10(signal_power / noise_power))


# ── Authentication Pipeline ────────────────────────────────────────────────────

class AuthenticationPipeline:
    """
    Ties together epoch extraction, averaging, and P300 decision.

    Usage:
        pipe = AuthenticationPipeline(engine)
        pipe.set_targets([3, 11, 17])          # password image IDs

        # ... oddball paradigm runs, marks are recorded in engine ...

        result = pipe.evaluate()
        if result.granted:
            unlock_vault()
    """

    def __init__(self, engine: BrainEngine, target_ids: Optional[list[int]] = None):
        self._engine     = engine
        self._extractor  = EpochExtractor(engine)
        self._averager   = SignalAverager()
        self._target_ids: set[int] = set(target_ids or [])

    def set_targets(self, target_ids: list[int]) -> None:
        self._target_ids = set(target_ids)
        self._engine.set_targets(target_ids)

    def reset(self) -> None:
        self._averager.reset()
        self._engine.clear_markers()

    def process_all_markers(self) -> dict:
        """
        Extract and average all recorded markers.
        Returns a stats dict for debugging / live display.
        """
        self._averager.reset()
        markers = self._engine.get_markers()
        accepted = rejected = 0

        for marker in markers:
            epoch = self._extractor.extract(marker)
            if epoch is None:
                rejected += 1
                continue
            is_target = marker.image_id in self._target_ids
            self._averager.add_epoch(epoch, is_target)
            accepted += 1

        logger.info(
            "Epochs: %d accepted, %d rejected | target=%d non-target=%d",
            accepted, rejected,
            self._averager.n_target, self._averager.n_nontarget,
        )
        return {"accepted": accepted, "rejected": rejected,
                "n_target": self._averager.n_target,
                "n_nontarget": self._averager.n_nontarget}

    def evaluate(self) -> AuthResult:
        """
        Run full pipeline and return authentication decision.
        """
        stats = self.process_all_markers()

        if self._averager.n_target < MIN_EPOCHS:
            return AuthResult(
                granted=False,
                target_peak_uv=0.0,
                nontarget_peak_uv=0.0,
                snr_db=-999.0,
                message=(
                    f"Insufficient target epochs: "
                    f"{self._averager.n_target}/{MIN_EPOCHS} required."
                ),
            )
        if self._averager.n_nontarget < MIN_EPOCHS:
            return AuthResult(
                granted=False,
                target_peak_uv=0.0,
                nontarget_peak_uv=0.0,
                snr_db=-999.0,
                message=(
                    f"Insufficient non-target epochs: "
                    f"{self._averager.n_nontarget}/{MIN_EPOCHS} required."
                ),
            )

        target_avg    = self._averager.target_average()
        nontarget_avg = self._averager.nontarget_average()

        # ── Mean-difference detection ─────────────────────────────────────────
        # With visually equated stimuli (all images same background/luminance),
        # no Visual Evoked Potential (VEP) component remains in the difference
        # waveform — only a cognitive P300 survives averaging.  The mean of the
        # difference across the 250–500 ms window is therefore the right measure:
        #
        #   • For a non-attending subject: target − non-target ≈ 0 at every
        #     sample → mean ≈ 0 µV → DENIED.
        #
        #   • For a genuine user: their cognitive P300 adds a consistent
        #     positive deflection at ~300–500 ms only in target epochs → mean > 0.
        #
        # WHY NOT PEAK:  peak-of-difference searches 63 samples for the largest
        # single value.  Pure Gaussian noise with σ ≈ 5 µV (grand-average
        # noise with 15 target epochs) produces an expected maximum of ≈ 15 µV
        # — guaranteed false positives regardless of stimulus design.
        t_series  = target_avg   [P300_ONSET:P300_OFFSET, :][:, P300_CHANNELS].mean(axis=1)
        nt_series = nontarget_avg[P300_ONSET:P300_OFFSET, :][:, P300_CHANNELS].mean(axis=1)
        diff_series = t_series - nt_series

        # Grand-average mean difference across the P300 window
        delta          = float(diff_series.mean())
        target_peak    = float(t_series.mean())
        nontarget_peak = float(nt_series.mean())

        # Noise estimate from the pre-P300 window (0–250 ms), where no
        # cognitive ERP is expected.  This gives an in-session noise floor.
        pre_t   = target_avg   [0:P300_ONSET, :][:, P300_CHANNELS].mean(axis=1)
        pre_nt  = nontarget_avg[0:P300_ONSET, :][:, P300_CHANNELS].mean(axis=1)
        pre_std = float(np.std(pre_t - pre_nt))          # σ of pre-P300 diff
        noise_std = max(pre_std, 1e-6)

        snr = compute_snr_db(target_peak, nontarget_peak, noise_std)

        # Dual-criteria grant: amplitude AND pre-window SNR must both pass.
        # - abs(delta) >= AUTH_THRESHOLD_UV guards against large-noise sessions.
        # - abs(delta) >= PRE_SNR_RATIO * noise_std normalises to in-session noise,
        #   ensuring the P300 window is meaningfully above the pre-stimulus baseline.
        PRE_SNR_RATIO = 1.5          # P300 window must be 1.5× the pre-window σ
        granted = (abs(delta) >= AUTH_THRESHOLD_UV and
                   abs(delta) >= PRE_SNR_RATIO * noise_std)

        msg = (
            f"GRANTED — mean|ΔP300|={abs(delta):.2f} µV, "
            f"pre_σ={noise_std:.2f} µV, SNR={snr:.1f} dB"
            if granted else
            f"DENIED — mean|ΔP300|={abs(delta):.2f} µV "
            f"(need ≥{AUTH_THRESHOLD_UV} µV AND ≥{PRE_SNR_RATIO}×pre_σ={PRE_SNR_RATIO*noise_std:.2f} µV)"
        )
        logger.info("Auth result: %s", msg)

        return AuthResult(
            granted=granted,
            target_peak_uv=target_peak,
            nontarget_peak_uv=nontarget_peak,
            snr_db=snr,
            message=msg,
        )

    def get_erp_data(self) -> dict:
        """
        Returns time axis and averaged ERPs for live plotting in the UI.
        """
        t = np.linspace(0, EPOCH_DURATION_S * 1000, EPOCH_SAMPLES)  # ms
        target_avg    = self._averager.target_average()
        nontarget_avg = self._averager.nontarget_average()
        target_sem    = self._averager.target_sem()

        def _p300_series(avg):
            if avg is None:
                return np.zeros(EPOCH_SAMPLES)
            return avg[:, P300_CHANNELS].mean(axis=1)

        return {
            "t_ms":          t.tolist(),
            "target":        _p300_series(target_avg).tolist(),
            "nontarget":     _p300_series(nontarget_avg).tolist(),
            "target_sem":    (_p300_series(target_sem).tolist() if target_sem is not None
                              else np.zeros(EPOCH_SAMPLES).tolist()),
            "p300_onset_ms": P300_ONSET_S  * 1000,
            "p300_offset_ms": P300_OFFSET_S * 1000,
        }
