"""
brain_engine.py — Soude BCI Core Controller
Manages UnicornPy device lifecycle, ring buffer, and stimulus synchronization.
Sub-team 1 (Hardware/API) owns this file.
"""

import threading
import time
import struct
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sys
import os
from pylsl import StreamInfo, StreamOutlet
import serial

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 250          # Unicorn Hybrid Black native sample rate (Hz)
N_CHANNELS = 8             # EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
BUFFER_SECONDS = 120       # Ring buffer duration
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

# Alias de compatibilidad
CH_FZ  = CH_1_FZ
CH_C3  = CH_2_C3
CH_CZ  = CH_3_CZ
CH_C4  = CH_4_C4
CH_PZ  = CH_5_PZ
CH_PO7 = CH_6_PO7
CH_OZ  = CH_7_OZ
CH_PO8 = CH_8_PO8

P300_CHANNELS = [CH_CZ, CH_PZ, CH_OZ]
CHANNEL_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

GETDATA_BLOCK = 4          # samples pulled per call (~16 ms at 250 Hz)

# Unicorn Hybrid Black native stream layout (17 cols):
# 0..7: EEG (8) | 8..10: Accel (3) | 11..13: Gyro (3) | 14: Battery (1) | 15: Counter (1) | 16: Validation (1)
UNICORN_TOTAL_COLS = 17

# LSL stream layout for all kinematic + physiological data (15 channels):
LSL_TOTAL_COLS = 15  # 8 EEG + 3 Accel + 3 Gyro + 1 Battery


# ── Estructuras de Datos Estandarizadas ────────────────────────────────────────

@dataclass
class UnicornDataPacket:
    """
    Estructura estándar que encapsula un bloque de adquisición de n_samples.
    
    Attributes:
        eeg:           Array (n_samples, 8) en µV.
        accelerometer: Array (n_samples, 3) en g (X, Y, Z).
        gyroscope:     Array (n_samples, 3) en deg/s (X, Y, Z).
        battery:       Array (n_samples,) porcentaje [0.0, 100.0].
    """
    eeg: np.ndarray
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    battery: np.ndarray

    def to_matrix(self) -> np.ndarray:
        """Serializa a array 2D shape (n_samples, 15) para red (LSL) o almacenamiento plano."""
        return np.column_stack([
            self.eeg,
            self.accelerometer,
            self.gyroscope,
            self.battery[:, np.newaxis]
        ])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "UnicornDataPacket":
        """Reconstruye un UnicornDataPacket a partir de un array 2D de 15 columnas."""
        return cls(
            eeg=matrix[:, :8],
            accelerometer=matrix[:, 8:11],
            gyroscope=matrix[:, 11:14],
            battery=matrix[:, 14],
        )


@dataclass
class StimulusMarker:
    """Records a single stimulus event with its buffer position and wall-clock time."""
    image_id: int
    buffer_index: int          # Write-head position in ring buffer at flash time
    timestamp: float           # time.perf_counter() value for external alignment
    is_target: bool = False


class UnicornInterface(ABC):
    """Abstract base so MockUnicorn, RealUnicorn, and LSLUnicorn share the same contract."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_data(self, n_samples: int) -> UnicornDataPacket:
        """Returns normalized UnicornDataPacket containing EEG, kinematics, and battery."""


# ── Real Device With API ────────────────────────────────────────────────────────────────
class RealUnicorn(UnicornInterface):
    """Wrapper around UnicornPy mapping the raw 17 columns to UnicornDataPacket."""

    def __init__(self, serial: str):
        self.serial = serial
        self._device = None

    def open(self) -> None:
        try:
            sdk_path = os.environ.get("UNICORN_SDK_PATH", "").strip()
            if sdk_path and sdk_path not in sys.path:
                sys.path.append(sdk_path)
            import UnicornPy  # type: ignore
            available = UnicornPy.GetAvailableDevices(True)
            if self.serial not in available:
                raise RuntimeError(
                    f"Device {self.serial} not found. Available: {available}"
                )
            self._device = UnicornPy.Unicorn(self.serial)
            self._device.StartAcquisition(False)
            logger.info("Unicorn %s acquisition started.", self.serial)
        except ImportError:
            raise RuntimeError(
                "UnicornPy is not available. Install the g.tec Unicorn Suite and set "
                "UNICORN_SDK_PATH in your .env, or leave UNICORN_SERIAL empty to use MockUnicorn."
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

    def get_data(self, n_samples: int) -> UnicornDataPacket:
        import UnicornPy  # type: ignore
        raw = bytearray(n_samples * UNICORN_TOTAL_COLS * 4)
        try:
            self._device.GetData(n_samples, raw, len(raw))
        except UnicornPy.DeviceException as exc:
            raise RuntimeError(f"Unicorn GetData failed: {exc}") from exc

        arr = np.frombuffer(raw, dtype=np.float32).reshape(n_samples, UNICORN_TOTAL_COLS)
        
        return UnicornDataPacket(
            eeg=arr[:, 0:8].astype(np.float64),
            accelerometer=arr[:, 8:11].astype(np.float64),
            gyroscope=arr[:, 11:14].astype(np.float64),
            battery=arr[:, 14].astype(np.float64),
        )

# ── Real Device With FREEAPI (Direct Serial) ──────────────────────────────────
class FreeUnicorn(UnicornInterface):
    """
    Controlador directo vía puerto serie (UART/Bluetooth) sin dependencia de UnicornPy.
    Parsea tramas binarias de 45 bytes (1 muestra por trama a 250 Hz).
    """

    START_ACQ = bytes([0x61, 0x7C, 0x87])
    STOP_ACQ  = bytes([0x63, 0x5C, 0xC5])
    FRAME_LEN = 45

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.serial = port
        self.baudrate = baudrate
        self.device: Optional[serial.Serial] = None

    def open(self) -> None:
        try:
            self.device = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0
            )
            self.device.reset_input_buffer()
            self.device.reset_output_buffer()

            # Comando de inicio de adquisición
            self.device.write(self.START_ACQ)
            response = self.device.read(3)

            if response != b'\x00\x00\x00':
                logger.warning(
                    "FreeUnicorn (%s): Respuesta de inicio no estándar: %s",
                    self.port, response.hex()
                )
            logger.info("FreeUnicorn en %s iniciado correctamente.", self.port)
        except Exception as exc:
            if self.device and self.device.is_open:
                self.device.close()
            self.device = None
            raise RuntimeError(f"Error al abrir Unicorn en puerto serie {self.port}: {exc}") from exc

    def close(self) -> None:
        if self.device is not None:
            try:
                if self.device.is_open:
                    self.device.write(self.STOP_ACQ)
                    time.sleep(0.05)
                    self.device.close()
                logger.info("FreeUnicorn (%s) cerrado.", self.port)
            except Exception as exc:
                logger.warning("Error cerrando FreeUnicorn: %s", exc)
            finally:
                self.device = None

    def _read_exact(self, n_bytes: int) -> bytes:
        """Lee exactamente n_bytes bloqueando hasta completarlos o dar timeout."""
        data = bytearray()
        while len(data) < n_bytes:
            chunk = self.device.read(n_bytes - len(data))
            if not chunk:
                raise TimeoutError("Timeout en lectura serie de FreeUnicorn.")
            data.extend(chunk)
        return bytes(data)

    def _read_frame(self) -> bytes:
        """Sincroniza y extrae una trama válida de 45 bytes (0xC0 0x00 ... 0x0D 0x0A)."""
        while True:
            # Buscar byte de cabecera 0xC0
            b = self.device.read(1)
            if not b:
                raise TimeoutError("Timeout buscando cabecera de trama Unicorn.")
            if b == b'\xC0':
                b2 = self.device.read(1)
                if b2 == b'\x00':
                    # Cabecera confirmada; leer los 43 bytes restantes del payload
                    rest = self._read_exact(self.FRAME_LEN - 2)
                    payload = b'\xC0\x00' + rest
                    # Verificar pie de paquete (0x0D 0x0A -> CR LF)
                    if payload[43:45] == b'\x0D\x0A':
                        return payload
                    # Si el footer no coincide, se perdió sincronización; continúa el escaneo

    def get_data(self, n_samples: int) -> UnicornDataPacket:
        if self.device is None or not self.device.is_open:
            raise RuntimeError("Dispositivo FreeUnicorn no está abierto.")

        eeg_arr   = np.zeros((n_samples, 8), dtype=np.float64)
        accel_arr = np.zeros((n_samples, 3), dtype=np.float64)
        gyro_arr  = np.zeros((n_samples, 3), dtype=np.float64)
        batt_arr  = np.zeros(n_samples, dtype=np.float64)

        for s in range(n_samples):
            payload = self._read_frame()

            # Batería (nibble bajo de byte 2, rango 0..15 -> 0..100%)
            batt_arr[s] = 100.0 * float(payload[2] & 0x0F) / 15.0

            # 8 Canales EEG (3 bytes cada uno en Big-Endian con signo)
            for ch in range(8):
                idx = 3 + ch * 3
                raw_bytes = b'\x00' + payload[idx:idx + 3]
                val = struct.unpack('>i', raw_bytes)[0]
                if val & 0x00800000:
                    val -= 0x01000000
                # Conversión de cuentas del ADC a µV
                eeg_arr[s, ch] = float(val) * 4500000.0 / 50331642.0

            # Acelerómetro (Little-Endian, 16-bit signed, escala ±8g -> 4096 LSB/g)
            accel_arr[s, 0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.0
            accel_arr[s, 1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.0
            accel_arr[s, 2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.0

            # Giroscopio (Little-Endian, 16-bit signed, escala ±1000 deg/s -> 32.8 LSB/(deg/s))
            gyro_arr[s, 0] = float(struct.unpack('<h', payload[33:35])[0]) / 32.8
            gyro_arr[s, 1] = float(struct.unpack('<h', payload[35:37])[0]) / 32.8
            gyro_arr[s, 2] = float(struct.unpack('<h', payload[37:39])[0]) / 32.8

        return UnicornDataPacket(
            eeg=eeg_arr,
            accelerometer=accel_arr,
            gyroscope=gyro_arr,
            battery=batt_arr,
        )

# ── Mock Device ────────────────────────────────────────────────────────────────
class MockUnicorn(UnicornInterface):
    """Generates synthetic EEG + kinematics + battery with P300 injection capabilities."""

    _P300_LATENCY_SAMPLES = int(0.30 * SAMPLE_RATE)
    _P300_WIDTH_SAMPLES   = int(0.10 * SAMPLE_RATE)
    _P300_AMPLITUDE_UV    = 8.0

    def __init__(self, p300_amplitude_uv: float = _P300_AMPLITUDE_UV):
        self._amplitude = p300_amplitude_uv
        self._rng = np.random.default_rng(seed=42)
        self._pending_p300: list[int] = []
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._served = 0
        self._simulated_battery = 95.0

    def open(self) -> None:
        logger.info("MockUnicorn opened (simulation mode).")

    def close(self) -> None:
        logger.info("MockUnicorn closed.")

    def notify_target(self) -> None:
        with self._lock:
            self._pending_p300.append(self._P300_LATENCY_SAMPLES)

    def get_data(self, n_samples: int) -> UnicornDataPacket:
        if self._t0 is None:
            self._t0 = time.monotonic()
        self._served += n_samples
        delay = self._t0 + self._served / SAMPLE_RATE - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        # 1/f Pink-noise baseline for EEG
        white = self._rng.standard_normal((n_samples, N_CHANNELS)) * 6.0
        eeg_out = np.zeros_like(white)
        prev = np.zeros(N_CHANNELS)
        for i in range(n_samples):
            eeg_out[i] = white[i] + 0.98 * prev
            prev = eeg_out[i]

        # Inject P300 for pending targets
        with self._lock:
            still_pending = []
            for remaining in self._pending_p300:
                for s in range(n_samples):
                    dist = abs(s - remaining)
                    if dist < self._P300_WIDTH_SAMPLES:
                        sigma = self._P300_WIDTH_SAMPLES / 2.5
                        amp = self._amplitude * np.exp(-0.5 * (dist / sigma) ** 2)
                        for ch in P300_CHANNELS:
                            eeg_out[s, ch] += amp
                new_remaining = remaining - n_samples
                if new_remaining > -self._P300_WIDTH_SAMPLES:
                    still_pending.append(new_remaining)
            self._pending_p300 = still_pending

        # Synthetic Accelerometer (gravity on Z ~1.0g + small tremor)
        accel = np.zeros((n_samples, 3), dtype=np.float64)
        accel[:, 2] = 1.0
        accel += self._rng.normal(0.0, 0.02, size=(n_samples, 3))

        # Synthetic Gyroscope (rest state ~0 deg/s + micro-movements)
        gyro = self._rng.normal(0.0, 0.1, size=(n_samples, 3))

        # Synthetic Battery (slow drain)
        self._simulated_battery = max(0.0, self._simulated_battery - 0.0001 * n_samples)
        battery = np.full(n_samples, self._simulated_battery, dtype=np.float64)

        return UnicornDataPacket(
            eeg=eeg_out,
            accelerometer=accel,
            gyroscope=gyro,
            battery=battery,
        )

# ── LSL Network Receiver ──────────────────────────────────────────────────────
class LSLUnicorn(UnicornInterface):
    """Receives normalized streams (15 channels) over LSL."""

    LSL_STREAM_NAME  = "Unicorn_EEG"
    RESOLVE_TIMEOUT  = 15.0
    PULL_TIMEOUT     = 0.05
    GET_DATA_TIMEOUT = 5.0

    def __init__(self):
        self._inlet = None
        self._pending: list[list[float]] = []
        self._lock = threading.Lock()

    def open(self) -> None:
        from pylsl import resolve_byprop, StreamInlet
        logger.info(
            "LSLUnicorn: searching for '%s' on the network (timeout %.0f s)…",
            self.LSL_STREAM_NAME, self.RESOLVE_TIMEOUT,
        )
        streams = resolve_byprop(
            "name", self.LSL_STREAM_NAME, timeout=self.RESOLVE_TIMEOUT
        )
        if not streams:
            raise RuntimeError(
                f"No LSL stream named '{self.LSL_STREAM_NAME}' found on the network."
            )
        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        logger.info(
            "LSLUnicorn connected: name='%s' channels=%d rate=%.0f Hz",
            info.name(), info.channel_count(), info.nominal_srate(),
        )

    def close(self) -> None:
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception as exc:
                logger.debug("Error closing LSL inlet: %s", exc)
            self._inlet = None
        logger.info("LSLUnicorn: stream closed.")

    def get_data(self, n_samples: int) -> UnicornDataPacket:
        deadline = time.monotonic() + self.GET_DATA_TIMEOUT
        with self._lock:
            while len(self._pending) < n_samples:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"LSLUnicorn timeout: waited {self.GET_DATA_TIMEOUT}s for "
                        f"{n_samples} samples. Check sender status."
                    )
                chunk, _ = self._inlet.pull_chunk(
                    max_samples=n_samples - len(self._pending),
                    timeout=self.PULL_TIMEOUT,
                )
                if chunk:
                    self._pending.extend(chunk)

            out = np.array(self._pending[:n_samples], dtype=np.float64)
            self._pending = self._pending[n_samples:]

        # Si el stream LSL tiene 15 canales, parseamos el paquete completo
        if out.shape[1] >= LSL_TOTAL_COLS:
            return UnicornDataPacket.from_matrix(out[:, :LSL_TOTAL_COLS])
        else:
            # Compatibilidad fallback: si un emisor legacy solo manda 8 canales EEG
            return UnicornDataPacket(
                eeg=out[:, :N_CHANNELS],
                accelerometer=np.zeros((n_samples, 3), dtype=np.float64),
                gyroscope=np.zeros((n_samples, 3), dtype=np.float64),
                battery=np.zeros(n_samples, dtype=np.float64),
            )

# ── Ring Buffer ────────────────────────────────────────────────────────────────
class RingBuffer:
    """
    Thread-safe circular buffer for continuous EEG + auxiliary data.
    Stores all 15 dimensions: shape (BUFFER_SAMPLES, 15).
    """

    def __init__(self):
        self._buf = np.zeros((BUFFER_SAMPLES, LSL_TOTAL_COLS), dtype=np.float64)
        self._write_head = 0
        self._total_written = 0
        self._lock = threading.RLock()

    @property
    def write_head(self) -> int:
        return self._write_head

    @property
    def total_written(self) -> int:
        return self._total_written

    def write_packet(self, packet: UnicornDataPacket) -> None:
        """Escribe un UnicornDataPacket serializándolo internamente."""
        matrix = packet.to_matrix()
        n = len(matrix)
        with self._lock:
            end = self._write_head + n
            if end <= BUFFER_SAMPLES:
                self._buf[self._write_head:end] = matrix
            else:
                first = BUFFER_SAMPLES - self._write_head
                self._buf[self._write_head:] = matrix[:first]
                self._buf[:n - first] = matrix[first:]
            self._write_head = end % BUFFER_SAMPLES
            self._total_written += n

    def read_from(self, start_index: int, n_samples: int) -> Optional[UnicornDataPacket]:
        """Extrae n_samples en un UnicornDataPacket estructurado."""
        with self._lock:
            oldest = 0 if self._total_written <= BUFFER_SAMPLES else self._total_written - BUFFER_SAMPLES
            if start_index < oldest or start_index + n_samples > self._total_written:
                return None
            idx = (start_index + np.arange(n_samples)) % BUFFER_SAMPLES
            chunk = self._buf[idx]
            return UnicornDataPacket.from_matrix(chunk)

    def read_eeg_from(self, start_index: int, n_samples: int) -> Optional[np.ndarray]:
        """Atajo para algoritmos que sólo necesitan los canales EEG (n_samples, 8)."""
        packet = self.read_from(start_index, n_samples)
        return packet.eeg if packet is not None else None

    def snapshot(self) -> UnicornDataPacket:
        """Copia ordenada cronológicamente de todo el búfer."""
        with self._lock:
            rolled = np.roll(self._buf.copy(), -self._write_head, axis=0)
            return UnicornDataPacket.from_matrix(rolled)

# ── Brain Engine ───────────────────────────────────────────────────────────────
class BrainEngine:
    """High-level controller."""

    def __init__(
        self,
        serial: Optional[str] = None,
        device: Optional[UnicornInterface] = None,
    ):
        if device is not None:
            self._device = device
        elif serial is not None and serial.upper() == "LSL":
            logger.info("UNICORN_SERIAL=LSL — using LSLUnicorn (network receiver mode).")
            self._device = LSLUnicorn()
        elif serial is not None and (serial.upper().startswith("COM") or serial.startswith("/dev/")):
            logger.info("Puerto serial detectado (%s) — using FreeUnicorn (Direct Serial).", serial)
            self._device = FreeUnicorn(port=serial)
        elif serial is not None:
            logger.info("Serial de hardware detectado (%s) — using RealUnicorn (UnicornPy).", serial)
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

    def start(self) -> None:
        self._device.open()
        self._running = True
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="EEG-Acquisition",
            daemon=True,
        )
        self._thread.start()
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenThread(0x0020, False, self._thread.ident)
            ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass
        logger.info("BrainEngine acquisition started.")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._device.close()
        logger.info("BrainEngine stopped.")

    def check_health(self) -> None:
        if self._acq_error is not None:
            raise self._acq_error

    def _acquisition_loop(self) -> None:
        outlet = None
        if not isinstance(self._device, LSLUnicorn):
            # Transmitimos 15 canales: 8 EEG + 3 Acc + 3 Gyro + 1 Battery
            info = StreamInfo(
                name='Unicorn_EEG',
                type='Multimodal',
                channel_count=LSL_TOTAL_COLS,
                nominal_srate=SAMPLE_RATE,
                channel_format='float32',
                source_id=self._device.serial if hasattr(self._device, 'serial') else 'soude_mock',
            )
            outlet = StreamOutlet(info)
            logger.info("LSL broadcast started with %d channels.", LSL_TOTAL_COLS)

        consecutive_errors = 0
        while self._running:
            try:
                # Ahora devuelve un UnicornDataPacket normalizado
                packet: UnicornDataPacket = self._device.get_data(GETDATA_BLOCK)
                
                # Almacena en el búfer circular
                self._buffer.write_packet(packet)

                # Transmite por LSL
                if outlet is not None:
                    outlet.push_chunk(packet.to_matrix().tolist())
                
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

    def mark_stimulus(self, image_id: int) -> StimulusMarker:
        buf_idx = self._buffer.total_written
        ts = time.perf_counter()

        marker = StimulusMarker(
            image_id=image_id,
            buffer_index=buf_idx,
            timestamp=ts,
        )
        with self._markers_lock:
            self._markers.append(marker)

        logger.debug("Stimulus marked: id=%d  buf=%d  t=%.6f", image_id, buf_idx, ts)
        return marker

    def set_targets(self, target_ids: list[int]) -> None:
        with self._markers_lock:
            for m in self._markers:
                m.is_target = m.image_id in target_ids

        if isinstance(self._device, MockUnicorn):
            with self._markers_lock:
                for m in self._markers:
                    if m.is_target:
                        self._device.notify_target()

    def get_markers(self) -> list[StimulusMarker]:
        with self._markers_lock:
            return list(self._markers)

    def clear_markers(self) -> None:
        with self._markers_lock:
            self._markers.clear()

    def get_epoch(self, marker: StimulusMarker, duration_s: float = 0.8) -> Optional[np.ndarray]:
        """
        Retorna la ventana EEG correspondiente a un marcador (n_samples, 8)
        manteniendo compatibilidad con el pipeline de clasificación P300.
        """
        n_samples = int(duration_s * SAMPLE_RATE)
        return self._buffer.read_eeg_from(marker.buffer_index, n_samples)

    def get_epoch_packet(self, marker: StimulusMarker, duration_s: float = 0.8) -> Optional[UnicornDataPacket]:
        """Retorna el paquete completo (EEG + Kinematics + Battery) para un marcador dado."""
        n_samples = int(duration_s * SAMPLE_RATE)
        return self._buffer.read_from(marker.buffer_index, n_samples)

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer