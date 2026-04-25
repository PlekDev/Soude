"""
brain_engine.py — Neuro-Lock BCI Core Controller
Manages UnicornPy device lifecycle, ring buffer, and stimulus synchronization.
Sub-team 1 (Hardware/API) owns this file.
"""

import threading
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import sys
import os
from pylsl import StreamInfo, StreamOutlet

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 250          # Unicorn Hybrid Black native sample rate (Hz)
N_CHANNELS = 8             # EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
BUFFER_SECONDS = 120       # Ring buffer duration — must exceed full paradigm (~57 s) + enrollment time + margin
BUFFER_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS

# Channel indices (0-based) matching Unicorn Hybrid Black layout
CH_1_FZ  = 0
CH_2_C3  = 1
CH_3_CZ  = 2
CH_4_C4  = 3
CH_5_PZ  = 4
CH_6_PO7 = 5
CH_7_OZ  = 6
CH_8_PO8 = 7

# Alias para mantener la compatibilidad con el resto del proyecto 
# (Así no tienes que cambiar los otros archivos)
CH_FZ  = CH_1_FZ
CH_C3  = CH_2_C3
CH_CZ  = CH_3_CZ
CH_C4  = CH_4_C4
CH_PZ  = CH_5_PZ
CH_PO7 = CH_6_PO7
CH_OZ  = CH_7_OZ
CH_PO8 = CH_8_PO8

# Channels used for P300 detection (centroparietal focus)
P300_CHANNELS = [CH_CZ, CH_PZ, CH_OZ]

# GetData block size: samples pulled per call
GETDATA_BLOCK = 4          # ~16 ms at 250 Hz — keeps latency low

# Total data columns returned by UnicornPy.GetData() per sample (Unicorn Hybrid Black)
# Layout: 8 EEG | 3 Accelerometer | 3 Gyroscope | 1 Battery | 1 Counter | 1 Validation
# Reference: g.tec Unicorn Python API, GetNumberOfAcquiredChannels() == 17
UNICORN_TOTAL_COLS = 17

ruta_unicorn = r"C:\Users\joldo\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib"
sys.path.append(ruta_unicorn)

@dataclass
class StimulusMarker:
    """Records a single stimulus event with its buffer position and wall-clock time."""
    image_id: int
    buffer_index: int          # Write-head position in ring buffer at flash time
    timestamp: float           # time.perf_counter() value for external alignment
    is_target: bool = False    # Set later during authentication evaluation


class UnicornInterface(ABC):
    """Abstract base so MockUnicorn and RealUnicorn share the same contract."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_data(self, n_samples: int) -> np.ndarray: ...
    """Returns shape (n_samples, N_CHANNELS) in µV."""


# ── Real Device ────────────────────────────────────────────────────────────────
class RealUnicorn(UnicornInterface):
    """
    Thin wrapper around UnicornPy that enforces the UnicornInterface contract.
    Import UnicornPy lazily so the rest of the codebase loads on machines
    that only have MockUnicorn available.
    """

    def __init__(self, serial: str):
        self.serial = serial
        self._device = None

    def open(self) -> None:
        try:
            import UnicornPy  # type: ignore
            available = UnicornPy.GetAvailableDevices(True)
            if self.serial not in available:
                raise RuntimeError(
                    f"Device {self.serial} not found. Available: {available}"
                )
            self._device = UnicornPy.Unicorn(self.serial)
            self._device.StartAcquisition(False)   # False = signal mode (not test)
            logger.info("Unicorn %s acquisition started.", self.serial)
        except ImportError:
            raise RuntimeError(
                "UnicornPy is not installed. Run from the Unicorn SDK environment, "
                "or use MockUnicorn for development."
            )

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.StopAcquisition()
            except Exception as exc:
                logger.warning("Error stopping acquisition: %s", exc)
            finally:
                del self._device
                self._device = None
                logger.info("Unicorn device released.")

    def get_data(self, n_samples: int) -> np.ndarray:
        """
        Pull n_samples from the device.  UnicornPy.GetData fills a flat buffer
        ordered as [ch0_s0, ch1_s0, …, ch7_s0, ch0_s1, …].
        Returns (n_samples, N_CHANNELS) float64 array in µV.
        """
        import UnicornPy  # type: ignore
        # Unicorn SDK requires a buffer sized: total_cols * samples * sizeof(float32)
        # UNICORN_TOTAL_COLS = 17: 8 EEG + 3 Accel + 3 Gyro + 1 Battery + 1 Counter + 1 Valid
        n_cols = UNICORN_TOTAL_COLS
        raw = bytearray(n_samples * n_cols * 4)
        try:
            data = self._device.GetData(n_samples, raw, len(raw))
        except UnicornPy.DeviceException as exc:
            raise RuntimeError(f"Unicorn GetData failed: {exc}") from exc

        arr = np.frombuffer(raw, dtype=np.float32).reshape(n_samples, n_cols)
        return arr[:, :N_CHANNELS].astype(np.float64)   # drop non-EEG columns


# ── Mock Device ────────────────────────────────────────────────────────────────
class MockUnicorn(UnicornInterface):
    """
    Generates synthetic EEG.  When notify_target() is called it injects a
    realistic P300-shaped response (positive peak ~300 ms post-stimulus) on
    P300_CHANNELS so the signal processing pipeline can be validated without
    hardware.

    Usage:
        engine = BrainEngine(device=MockUnicorn())
    """

    _P300_LATENCY_SAMPLES = int(0.30 * SAMPLE_RATE)   # 300 ms
    _P300_WIDTH_SAMPLES   = int(0.10 * SAMPLE_RATE)   # ~100 ms FWHM
    _P300_AMPLITUDE_UV    = 8.0                        # µV peak

    def __init__(self):
        self._rng = np.random.default_rng(seed=42)
        self._pending_p300: list[int] = []   # samples until P300 peak injection
        self._lock = threading.Lock()
        self._sample_counter = 0

    def open(self) -> None:
        logger.info("MockUnicorn opened (simulation mode).")

    def close(self) -> None:
        logger.info("MockUnicorn closed.")

    def notify_target(self) -> None:
        """Call this from BrainEngine.mark_stimulus when image is a target."""
        with self._lock:
            self._pending_p300.append(self._P300_LATENCY_SAMPLES)

    def get_data(self, n_samples: int) -> np.ndarray:
        """
        Produce pink-ish noise baseline with optional P300 bump injected on
        Cz, Pz, Oz.  Sleeps to pace at real hardware sample rate so the ring
        buffer doesn't overflow in test/simulator mode.
        """
        import time
        # Block for the natural duration of n_samples (mimics hardware behaviour)
        time.sleep(n_samples / SAMPLE_RATE)
        # 1/f noise approximation: white noise low-passed in frequency
        white = self._rng.standard_normal((n_samples, N_CHANNELS)) * 6.0
        # Simple IIR to tint toward pink (b=1, a=[1, -0.98])
        out = np.zeros_like(white)
        prev = np.zeros(N_CHANNELS)
        for i in range(n_samples):
            out[i] = white[i] + 0.98 * prev
            prev = out[i]

        # Inject P300 for pending targets
        with self._lock:
            still_pending = []
            for remaining in self._pending_p300:
                for s in range(n_samples):
                    abs_pos = (self._sample_counter + s)
                    peak_pos = abs_pos + remaining - s  # align relative position
                    dist = abs(s - (n_samples - remaining))
                    if dist < self._P300_WIDTH_SAMPLES:
                        sigma = self._P300_WIDTH_SAMPLES / 2.5
                        amp = self._P300_AMPLITUDE_UV * np.exp(
                            -0.5 * (dist / sigma) ** 2
                        )
                        for ch in P300_CHANNELS:
                            out[s, ch] += amp
                new_remaining = remaining - n_samples
                if new_remaining > -self._P300_WIDTH_SAMPLES:
                    still_pending.append(new_remaining)
            self._pending_p300 = still_pending

        self._sample_counter += n_samples
        return out


# ── Ring Buffer ────────────────────────────────────────────────────────────────
class RingBuffer:
    """
    Thread-safe circular buffer for continuous EEG storage.
    Shape: (BUFFER_SAMPLES, N_CHANNELS).
    write_head points to the NEXT slot to be written.
    """

    def __init__(self):
        self._buf = np.zeros((BUFFER_SAMPLES, N_CHANNELS), dtype=np.float64)
        self._write_head = 0
        self._total_written = 0
        self._lock = threading.RLock()

    @property
    def write_head(self) -> int:
        return self._write_head

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(self, samples: np.ndarray) -> None:
        """
        Write (n, N_CHANNELS) samples.  Wraps around automatically.
        """
        n = len(samples)
        with self._lock:
            end = self._write_head + n
            if end <= BUFFER_SAMPLES:
                self._buf[self._write_head:end] = samples
            else:
                first = BUFFER_SAMPLES - self._write_head
                self._buf[self._write_head:] = samples[:first]
                self._buf[:n - first] = samples[first:]
            self._write_head = end % BUFFER_SAMPLES
            self._total_written += n

    def read_from(self, start_index: int, n_samples: int) -> Optional[np.ndarray]:
        """
        Return n_samples starting from start_index (ring-wrapped).
        Returns None if start_index is older than BUFFER_SAMPLES ago.
        """
        with self._lock:
            # When buffer hasn't filled once yet, oldest valid index is 0
            if self._total_written <= BUFFER_SAMPLES:
                oldest = 0
            else:
                oldest = self._total_written - BUFFER_SAMPLES

            if start_index < oldest:
                logger.warning(
                    "Requested index %d is before oldest buffered sample %d.",
                    start_index, oldest
                )
                return None
            if start_index + n_samples > self._total_written:
                logger.debug(
                    "Requested index %d + %d samples not yet acquired (have %d).",
                    start_index, n_samples, self._total_written
                )
                return None
            out = np.empty((n_samples, N_CHANNELS), dtype=np.float64)
            for i in range(n_samples):
                ring_pos = (start_index + i) % BUFFER_SAMPLES
                out[i] = self._buf[ring_pos]
            return out

    def snapshot(self) -> np.ndarray:
        """Return a copy of the entire ring buffer in chronological order."""
        with self._lock:
            return np.roll(self._buf.copy(), -self._write_head, axis=0)


# ── Brain Engine ───────────────────────────────────────────────────────────────
class BrainEngine:
    """
    High-level controller.  Spawns a high-priority acquisition thread,
    manages the ring buffer, and records stimulus markers.

    Example:
        engine = BrainEngine(serial="UN-2023.10.01")
        engine.start()
        ...
        engine.mark_stimulus(image_id=7)
        ...
        markers = engine.get_markers()
        engine.stop()
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        device: Optional[UnicornInterface] = None,
    ):
        if device is not None:
            self._device = device
        elif serial is not None:
            self._device = RealUnicorn(serial)
        else:
            logger.info("No serial or device provided — using MockUnicorn.")
            self._device = MockUnicorn()

        self._buffer = RingBuffer()
        self._markers: list[StimulusMarker] = []
        self._markers_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._acq_error: Optional[Exception] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open device and start background acquisition thread."""
        self._device.open()
        self._running = True
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="EEG-Acquisition",
            daemon=True,
        )
        self._thread.start()
        # Elevate OS thread priority on Windows
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenThread(0x0020, False, self._thread.ident)
            ctypes.windll.kernel32.SetThreadPriority(handle, 2)  # THREAD_PRIORITY_HIGHEST
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass  # Non-Windows or permission denied — acceptable
        logger.info("BrainEngine acquisition started.")

    def stop(self) -> None:
        """Signal acquisition thread to stop and release device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._device.close()
        logger.info("BrainEngine stopped.")

    def check_health(self) -> None:
        """Raise if acquisition thread encountered a fatal error."""
        if self._acq_error is not None:
            raise self._acq_error

    # ── Acquisition Loop ───────────────────────────────────────────────────────

    def _acquisition_loop(self) -> None:
        """Bucle de adquisición que también transmite por LSL."""
        
        # 1. Configurar el canal de transmisión LSL
        info = StreamInfo(
            name='Unicorn_EEG', 
            type='EEG', 
            channel_count=N_CHANNELS, 
            nominal_srate=SAMPLE_RATE, 
            channel_format='float32', 
            source_id=self._device.serial if hasattr(self._device, 'serial') else 'mock_123'
        )
        outlet = StreamOutlet(info)
        print("Transmisión LSL iniciada. Otras computadoras ya pueden escuchar.")

        consecutive_errors = 0
        while self._running:
            try:
                # Obtenemos los datos (ej. 4 muestras)
                samples = self._device.get_data(GETDATA_BLOCK)
                
                # Guardamos localmente (tu código original)
                self._buffer.write(samples)
                
                # ¡NUEVO! Enviamos por WiFi a cualquier PC que esté escuchando
                # pylsl espera una lista de listas, así que convertimos el array de numpy
                outlet.push_chunk(samples.tolist())
                
                consecutive_errors = 0
            except RuntimeError as exc:
                consecutive_errors += 1
                logger.error("Acquisition error #%d: %s", consecutive_errors, exc)
                if consecutive_errors >= 10:
                    self._acq_error = exc
                    self._running = False
                    logger.critical("Too many consecutive errors. Stopping acquisition.")
                    return
                time.sleep(0.01)

    # ── Stimulus Marking ───────────────────────────────────────────────────────

    def mark_stimulus(self, image_id: int) -> StimulusMarker:
        """
        Record the exact ring-buffer position when an image is flashed.
        Must be called from the UI thread immediately after the screen update.

        Returns the StimulusMarker so the caller can store it if needed.
        """
        # Capture buffer write head BEFORE any further samples arrive
        buf_idx = self._buffer.total_written
        ts = time.perf_counter()

        marker = StimulusMarker(
            image_id=image_id,
            buffer_index=buf_idx,
            timestamp=ts,
        )
        with self._markers_lock:
            self._markers.append(marker)

        # If using mock device, we need to inform it for P300 injection.
        # is_target is determined later; but we can hook this after set_targets().
        logger.debug("Stimulus marked: id=%d  buf=%d  t=%.6f", image_id, buf_idx, ts)
        return marker

    # ── Marker Management ──────────────────────────────────────────────────────

    def set_targets(self, target_ids: list[int]) -> None:
        """
        Designate which image_ids are the password (target) images.
        Must be called before evaluation begins.
        """
        with self._markers_lock:
            for m in self._markers:
                m.is_target = m.image_id in target_ids

        # Let MockUnicorn inject P300s retroactively for already-marked targets
        if isinstance(self._device, MockUnicorn):
            with self._markers_lock:
                for m in self._markers:
                    if m.is_target:
                        self._device.notify_target()

    def get_markers(self) -> list[StimulusMarker]:
        """Return a copy of all recorded stimulus markers."""
        with self._markers_lock:
            return list(self._markers)

    def clear_markers(self) -> None:
        with self._markers_lock:
            self._markers.clear()

    # ── Data Access ────────────────────────────────────────────────────────────

    def get_epoch(self, marker: StimulusMarker, duration_s: float = 0.8) -> Optional[np.ndarray]:
        """
        Extract a single epoch starting at marker.buffer_index.
        Returns (n_samples, N_CHANNELS) or None if data is no longer in buffer.
        """
        n_samples = int(duration_s * SAMPLE_RATE)
        return self._buffer.read_from(marker.buffer_index, n_samples)

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer
