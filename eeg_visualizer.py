"""
eeg_visualizer.py  —  Unicorn Live EEG Visualizer
Standalone PyQt6 app.  Auto-detects the first available Unicorn Hybrid Black
via UnicornPy; falls back to MockUnicorn only when the SDK or hardware is
unavailable (with a visible warning in the status bar).

Usage:
    python eeg_visualizer.py                    # auto-detect
    python eeg_visualizer.py UN-2023.10.01      # force serial number
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))
from brain_engine import (
    BrainEngine, MockUnicorn, RealUnicorn,
    SAMPLE_RATE, N_CHANNELS,
)
from filters import build_bandpass_sos, apply_filter_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Colour palette (matches app.py / Colors class) ───────────────────────────
class C:
    BG_DEEP  = "#060a10"
    BG_PANEL = "#0d1520"
    BG_CARD  = "#101828"
    ACCENT   = "#00e5ff"
    ACCENT2  = "#7c4dff"
    SUCCESS  = "#00e676"
    DANGER   = "#ff1744"
    WARN     = "#ffab40"
    TEXT_HI  = "#e8f4fd"
    TEXT_LO  = "#4a6478"
    BORDER   = "#1a2e42"


CH_NAMES  = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CH_COLORS = [
    "#00e5ff", "#7c4dff", "#00e676", "#ff5252",
    "#ffab40", "#ea80fc", "#40c4ff", "#b2ff59",
]

# Band definitions: (name, low_hz, high_hz, colour)
BANDS = [
    ("Delta", 0.5,  4.0,  "#7c4dff"),
    ("Theta", 4.0,  8.0,  "#40c4ff"),
    ("Alpha", 8.0,  13.0, "#00e676"),
    ("Beta",  13.0, 30.0, "#ffab40"),
    ("Gamma", 30.0, 45.0, "#ea80fc"),
]

# Pre-build SOS band-pass filters (reused every frame — building is slow)
_BAND_SOS = {
    name: build_bandpass_sos(lo, hi)
    for name, lo, hi, _ in BANDS
}


# ─────────────────────────────────────────────────────────────────────────────
#  Device detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_device(forced_serial: str = "") -> tuple:
    """
    Returns (UnicornInterface_instance, label_str, is_real_bool).

    Priority:
      1. forced_serial argument
      2. UNICORN_SERIAL environment variable
      3. First device reported by UnicornPy.GetAvailableDevices()
      4. MockUnicorn fallback (logs a warning)
    """
    serial = forced_serial or os.environ.get("UNICORN_SERIAL", "")
    try:
        import UnicornPy  # type: ignore   # only present in the g.tec SDK env
        devices = UnicornPy.GetAvailableDevices(True)
        logger.info("Unicorn devices detected: %s", devices)
        if not devices:
            raise RuntimeError("No Unicorn devices found over Bluetooth.")
        target = serial if serial in devices else devices[0]
        return RealUnicorn(target), target, True
    except ImportError:
        logger.warning("UnicornPy not installed — using MockUnicorn.")
    except Exception as exc:
        logger.warning("Device detection failed (%s) — using MockUnicorn.", exc)
    return MockUnicorn(), "SIMULATOR", False


# ─────────────────────────────────────────────────────────────────────────────
#  Reusable style helpers
# ─────────────────────────────────────────────────────────────────────────────
def mono_label(text: str, size: int = 10, color: str = C.TEXT_LO) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:'Share Tech Mono','Courier New',monospace; "
        f"font-size:{size}px; background:transparent; border:none;"
    )
    return lbl


def divider() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{C.BORDER};")
    return f


def glow_button(text: str, accent: str = C.ACCENT) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(34)
    btn.setStyleSheet(f"""
        QPushButton {{
            background:transparent; border:1px solid {accent};
            border-radius:5px; color:{accent};
            font-family:'Share Tech Mono','Courier New',monospace;
            font-size:11px; letter-spacing:2px; padding:0 14px;
        }}
        QPushButton:hover  {{ background:{accent}22; color:{C.TEXT_HI}; }}
        QPushButton:pressed{{ background:{accent}44; }}
        QPushButton:disabled{{ border-color:{C.TEXT_LO}; color:{C.TEXT_LO}; }}
    """)
    return btn


# ─────────────────────────────────────────────────────────────────────────────
#  Waveform widget  (8-channel scrolling oscilloscope)
# ─────────────────────────────────────────────────────────────────────────────
class WaveformWidget(QWidget):
    """
    Paints the last ``window_s`` seconds of all 8 EEG channels using data
    fetched directly from engine.buffer.read_from().

    Scale modes:
      • Fixed µV  — each channel uses the same ±scale µV range.
      • Auto      — each channel auto-scaled to ±3σ.
    """

    _WINDOW_OPTIONS = {2: 2 * SAMPLE_RATE, 4: 4 * SAMPLE_RATE, 8: 8 * SAMPLE_RATE}
    ML, MR, MT, MB = 44, 8, 6, 22   # margins: left, right, top, bottom

    def __init__(self, engine: BrainEngine, parent=None):
        super().__init__(parent)
        self._engine    = engine
        self._data: Optional[np.ndarray] = None
        self._scale     = 150.0
        self._autoscale = False
        self._window_s  = 4
        self._n_samples = self._WINDOW_OPTIONS[4]
        self._paused    = False

        self.setMinimumSize(640, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:{C.BG_PANEL};")

    # ── Public controls ───────────────────────────────────────────────────────
    def set_scale(self, uv: float):
        self._scale     = float(uv)
        self._autoscale = False

    def set_autoscale(self, on: bool):
        self._autoscale = on

    def set_window(self, seconds: int):
        self._window_s  = int(seconds)
        self._n_samples = self._WINDOW_OPTIONS.get(seconds, 4 * SAMPLE_RATE)

    def set_paused(self, paused: bool):
        self._paused = paused

    def current_data(self) -> Optional[np.ndarray]:
        return self._data

    # ── Data fetch (called by main timer) ─────────────────────────────────────
    def fetch(self) -> None:
        if self._paused:
            return
        try:
            buf   = self._engine.buffer
            total = buf.total_written
            n     = self._n_samples
            if total < n:
                return
            data = buf.read_from(total - n, n)
            if data is not None:
                self._data = data
                self.update()   # schedule repaint
        except Exception as exc:
            logger.debug("WaveformWidget.fetch: %s", exc)

    # ── Painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ML, MR, MT, MB = self.ML, self.MR, self.MT, self.MB

        painter.fillRect(0, 0, w, h, QColor(C.BG_PANEL))

        if self._data is None:
            painter.setPen(QPen(QColor(C.TEXT_LO)))
            painter.setFont(QFont("Share Tech Mono", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Waiting for EEG data…")
            painter.end()
            return

        data   = self._data
        n_samp = len(data)
        plot_w = w - ML - MR
        plot_h = h - MT - MB
        ch_h   = plot_h / N_CHANNELS

        # Border
        painter.setPen(QPen(QColor(C.BORDER)))
        painter.drawRect(ML, MT, plot_w, plot_h)

        # One pixel per horizontal step keeps painting fast
        x_step = max(1, n_samp // plot_w)

        for ch in range(N_CHANNELS):
            y_top    = MT + ch * ch_h
            y_center = y_top + ch_h / 2

            # Lane separator
            if ch > 0:
                painter.setPen(QPen(QColor(C.BORDER)))
                painter.drawLine(ML, int(y_top), ML + plot_w, int(y_top))

            # Channel label (left gutter)
            painter.setPen(QPen(QColor(CH_COLORS[ch])))
            painter.setFont(QFont("Share Tech Mono", 8))
            painter.drawText(2, int(y_center) + 4, CH_NAMES[ch])

            # Zero baseline
            base_pen = QPen(QColor(26, 46, 66, 70))
            painter.setPen(base_pen)
            painter.drawLine(ML, int(y_center), ML + plot_w, int(y_center))

            # Scale factor
            signal = data[:, ch]
            if self._autoscale:
                std   = float(np.std(signal)) or 1.0
                scale = (ch_h * 0.42) / (3.0 * std)
            else:
                scale = (ch_h * 0.42) / self._scale

            # Build QPainterPath (much faster than individual drawLine calls)
            sig_pen = QPen(QColor(CH_COLORS[ch]))
            sig_pen.setWidthF(1.2)
            painter.setPen(sig_pen)

            path = QPainterPath()
            first = True
            for i in range(0, n_samp, x_step):
                x = ML + (i / n_samp) * plot_w
                y = y_center - float(signal[i]) * scale
                y = max(y_top + 1.0, min(y_top + ch_h - 1.0, y))
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)

        # ── Time axis ─────────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        for t in range(self._window_s + 1):
            x = ML + (t / self._window_s) * plot_w
            painter.drawLine(int(x), MT + plot_h, int(x), MT + plot_h + 4)
            lbl = f"-{self._window_s - t}s" if t < self._window_s else "now"
            painter.drawText(int(x) - 14, MT + plot_h + 16, lbl)

        # ── Scale badge ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(C.ACCENT)))
        painter.setFont(QFont("Share Tech Mono", 8))
        scale_txt = "AUTO" if self._autoscale else f"±{self._scale:.0f} µV"
        painter.drawText(ML + 4, MT + 12, scale_txt)

        painter.end()


# ─────────────────────────────────────────────────────────────────────────────
#  FFT spectrum widget  (Cz channel, 0–50 Hz)
# ─────────────────────────────────────────────────────────────────────────────
class FFTWidget(QWidget):
    """Draws a bar-chart FFT of the Cz channel coloured by brainwave band."""

    CZ_IDX = 2   # Cz is channel index 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mags: Optional[np.ndarray] = None
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background:{C.BG_CARD};")

    def update_data(self, data: np.ndarray) -> None:
        """Compute FFT from data shape (n_samples, N_CHANNELS)."""
        if data is None or len(data) < 64:
            return
        signal = data[:, self.CZ_IDX]
        # Windowed FFT
        window = np.hanning(len(signal))
        fft    = np.abs(np.fft.rfft(signal * window))
        freqs  = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)
        # Keep only 0–50 Hz
        mask     = freqs <= 50.0
        self._mags  = fft[mask]
        self._freqs = freqs[mask]
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(C.BG_CARD))

        if self._mags is None or len(self._mags) == 0:
            painter.end()
            return

        mags  = self._mags
        freqs = self._freqs
        n_bins = len(mags)
        max_m  = float(np.max(mags)) or 1.0
        bw     = w / n_bins

        # Colour lookup by frequency
        def band_color(f: float) -> str:
            for _, lo, hi, col in BANDS:
                if lo <= f < hi:
                    return col
            return C.TEXT_LO

        for i, (f, m) in enumerate(zip(freqs, mags)):
            bar_h = int((m / max_m) * (h - 14))
            col   = QColor(band_color(float(f)))
            col.setAlpha(180)
            painter.fillRect(int(i * bw), h - bar_h - 12, max(1, int(bw) - 1), bar_h, col)

        # Axis labels
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        for f_mark in [4, 8, 13, 30, 50]:
            idx = int(f_mark / 50.0 * n_bins)
            if idx < n_bins:
                x = int(idx * bw)
                painter.drawLine(x, h - 12, x, h - 8)
                painter.drawText(x - 10, h - 1, f"{f_mark}Hz")

        painter.end()


# ─────────────────────────────────────────────────────────────────────────────
#  Right-panel sidebar
# ─────────────────────────────────────────────────────────────────────────────
class SidebarWidget(QWidget):
    """
    Compact sidebar with:
      • Signal quality  — per-channel RMS bar (last 1 s)
      • Band power      — Delta/Theta/Alpha/Beta/Gamma power bars (Cz)
      • Live stats      — peak µV, RMS, SNR, sample count
      • FFT spectrum    — Cz channel 0–50 Hz
    """

    _RMS_WINDOW = SAMPLE_RATE       # 1 second for quality bars
    _BP_WINDOW  = 2 * SAMPLE_RATE   # 2 seconds for band power

    def __init__(self, engine: BrainEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self.setFixedWidth(260)
        self.setStyleSheet(f"background:{C.BG_DEEP}; border-left:1px solid {C.BORDER};")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Signal quality ────────────────────────────────────────────────────
        root.addWidget(mono_label("SIGNAL QUALITY", 9, C.ACCENT))
        root.addWidget(divider())
        self._q_bars:  list[QProgressBar] = []
        self._q_vals:  list[QLabel]       = []
        for i, ch in enumerate(CH_NAMES):
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(mono_label(ch, 9, CH_COLORS[i]))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background:{C.BG_CARD}; border:none; border-radius:4px;
                }}
                QProgressBar::chunk {{
                    background:{CH_COLORS[i]}; border-radius:4px;
                }}
            """)
            row.addWidget(bar, stretch=1)
            val = mono_label("—", 9, C.TEXT_LO)
            val.setFixedWidth(40)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            self._q_bars.append(bar)
            self._q_vals.append(val)
            root.addLayout(row)

        root.addSpacing(4)

        # ── Band power ────────────────────────────────────────────────────────
        root.addWidget(mono_label("BAND POWER  (Cz)", 9, C.ACCENT))
        root.addWidget(divider())
        self._bp_bars: list[QProgressBar] = []
        self._bp_vals: list[QLabel]       = []
        for name, lo, hi, col in BANDS:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = mono_label(name, 9, col)
            lbl.setFixedWidth(42)
            row.addWidget(lbl)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background:{C.BG_CARD}; border:none; border-radius:4px;
                }}
                QProgressBar::chunk {{
                    background:{col}; border-radius:4px;
                }}
            """)
            row.addWidget(bar, stretch=1)
            val = mono_label("—", 9, C.TEXT_LO)
            val.setFixedWidth(40)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            self._bp_bars.append(bar)
            self._bp_vals.append(val)
            root.addLayout(row)

        root.addSpacing(4)

        # ── Live stats ────────────────────────────────────────────────────────
        root.addWidget(mono_label("LIVE STATS", 9, C.ACCENT))
        root.addWidget(divider())
        stats_grid = QGridLayout()
        stats_grid.setSpacing(6)

        self._stat_labels: dict[str, QLabel] = {}
        stat_defs = [
            ("Peak",    "µV",  0, 0),
            ("RMS",     "µV",  0, 1),
            ("SNR",     "dB",  1, 0),
            ("Samples", "",    1, 1),
        ]
        for key, unit, row_i, col_i in stat_defs:
            card = QFrame()
            card.setStyleSheet(
                f"background:{C.BG_CARD}; border:1px solid {C.BORDER};"
                f"border-radius:5px;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(6, 5, 6, 5)
            cl.setSpacing(2)
            cl.addWidget(mono_label(key.upper(), 8, C.TEXT_LO))
            val_lbl = mono_label("—", 15, C.TEXT_HI)
            unit_lbl = mono_label(unit, 8, C.TEXT_LO)
            row_w = QHBoxLayout()
            row_w.setSpacing(3)
            row_w.addWidget(val_lbl)
            row_w.addWidget(unit_lbl)
            row_w.addStretch()
            cl.addLayout(row_w)
            stats_grid.addWidget(card, row_i, col_i)
            self._stat_labels[key] = val_lbl
        root.addLayout(stats_grid)

        root.addSpacing(4)

        # ── FFT spectrum ──────────────────────────────────────────────────────
        root.addWidget(mono_label("SPECTRUM  (Cz, 0–50 Hz)", 9, C.ACCENT))
        root.addWidget(divider())
        self._fft_widget = FFTWidget()
        root.addWidget(self._fft_widget)

        root.addStretch()

    # ── Update (called by main timer) ─────────────────────────────────────────
    def update_stats(self) -> None:
        try:
            buf   = self._engine.buffer
            total = buf.total_written
            if total < self._BP_WINDOW:
                return

            # ── Signal quality (last 1 s, all channels) ──────────────────────
            q_data = buf.read_from(total - self._RMS_WINDOW, self._RMS_WINDOW)
            if q_data is not None:
                rms_all = np.sqrt(np.mean(q_data ** 2, axis=0))   # shape (8,)
                # Normalise to ±150 µV expected full-scale
                for i in range(N_CHANNELS):
                    pct = min(100, int((rms_all[i] / 150.0) * 100))
                    self._q_bars[i].setValue(pct)
                    self._q_vals[i].setText(f"{rms_all[i]:.1f}")
                    # Recolour bar: green → warn → danger by saturation proxy
                    if pct < 5:
                        accent = C.DANGER   # flat signal / disconnected
                    elif pct > 80:
                        accent = C.WARN     # saturated
                    else:
                        accent = CH_COLORS[i]
                    self._q_bars[i].setStyleSheet(f"""
                        QProgressBar {{
                            background:{C.BG_CARD}; border:none; border-radius:4px;
                        }}
                        QProgressBar::chunk {{
                            background:{accent}; border-radius:4px;
                        }}
                    """)

            # ── Band power (last 2 s, Cz channel) ────────────────────────────
            bp_data = buf.read_from(total - self._BP_WINDOW, self._BP_WINDOW)
            if bp_data is not None:
                cz_sig = bp_data[:, 2].reshape(-1, 1)  # (n, 1)
                bp_rms = []
                for name, *_ in BANDS:
                    sos     = _BAND_SOS[name]
                    filtered = apply_filter_chain(cz_sig, [sos])
                    bp_rms.append(float(np.sqrt(np.mean(filtered ** 2))))
                max_bp = max(bp_rms) or 1.0
                for i, rms in enumerate(bp_rms):
                    pct = min(100, int((rms / max_bp) * 100))
                    self._bp_bars[i].setValue(pct)
                    self._bp_vals[i].setText(f"{rms:.2f}")

            # ── Live stats (last 1 s, all channels for peak; Cz for RMS/SNR) ─
            if q_data is not None:
                peak = float(np.max(np.abs(q_data)))
                cz   = q_data[:, 2]
                rms_cz = float(np.sqrt(np.mean(cz ** 2)))
                noise  = float(np.std(np.diff(cz))) / np.sqrt(2)
                snr    = 20.0 * np.log10(rms_cz / (noise + 1e-9))

                self._stat_labels["Peak"].setText(f"{peak:.1f}")
                self._stat_labels["RMS"].setText(f"{rms_cz:.1f}")
                self._stat_labels["SNR"].setText(f"{snr:.1f}")
                self._stat_labels["Samples"].setText(f"{total:,}")

            # ── FFT ───────────────────────────────────────────────────────────
            if q_data is not None:
                self._fft_widget.update_data(q_data)

        except Exception as exc:
            logger.debug("SidebarWidget.update_stats: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Toolbar
# ─────────────────────────────────────────────────────────────────────────────
class ToolbarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setStyleSheet(
            f"background:{C.BG_PANEL}; border-bottom:1px solid {C.BORDER};"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(12)

        # Logo
        logo = mono_label("🧠  UNICORN EEG VISUALIZER", 12, C.ACCENT)
        layout.addWidget(logo)

        # Device label
        self.device_lbl = mono_label("DEVICE: —", 10, C.TEXT_LO)
        layout.addWidget(self.device_lbl)

        layout.addStretch()

        # Scale selector
        layout.addWidget(mono_label("Scale:", 10, C.TEXT_LO))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["50 µV", "100 µV", "150 µV", "300 µV", "AUTO"])
        self.scale_combo.setCurrentIndex(2)   # 150 µV default
        self.scale_combo.setFixedWidth(90)
        self.scale_combo.setStyleSheet(f"""
            QComboBox {{
                background:{C.BG_CARD}; border:1px solid {C.BORDER};
                border-radius:4px; color:{C.TEXT_HI};
                font-family:'Share Tech Mono','Courier New',monospace;
                font-size:11px; padding:2px 6px;
            }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{
                background:{C.BG_CARD}; color:{C.TEXT_HI};
                selection-background-color:{C.ACCENT}33;
            }}
        """)
        layout.addWidget(self.scale_combo)

        # Window selector
        layout.addWidget(mono_label("Window:", 10, C.TEXT_LO))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["2 s", "4 s", "8 s"])
        self.window_combo.setCurrentIndex(1)   # 4 s default
        self.window_combo.setFixedWidth(70)
        self.window_combo.setStyleSheet(self.scale_combo.styleSheet())
        layout.addWidget(self.window_combo)

        # Pause button
        self.pause_btn = glow_button("⏸  PAUSE", C.ACCENT2)
        self.pause_btn.setFixedWidth(110)
        layout.addWidget(self.pause_btn)

        # Sample-rate badge
        self.rate_lbl = mono_label(f"{SAMPLE_RATE} Hz  ·  {N_CHANNELS} ch", 10, C.TEXT_LO)
        layout.addWidget(self.rate_lbl)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────
class EEGVisualizerWindow(QMainWindow):
    """
    Main application window.

    Refresh timers:
      • 50 ms  (20 fps)  — waveform & FFT
      • 200 ms (5 fps)   — signal quality, band power, stats
      •   1 s            — clock / sample counter in status bar
    """

    def __init__(self, forced_serial: str = ""):
        super().__init__()
        self.setWindowTitle("Unicorn EEG Visualizer")
        self.setMinimumSize(1100, 640)
        self.setStyleSheet(f"background:{C.BG_DEEP};")

        # ── Detect device ─────────────────────────────────────────────────────
        device, label, is_real = detect_device(forced_serial)
        self._is_real = is_real
        self._start_time = time.monotonic()

        self._engine = BrainEngine(device=device)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._toolbar = ToolbarWidget(self)
        device_color = C.SUCCESS if is_real else C.WARN
        self._toolbar.device_lbl.setText(f"DEVICE: {label}")
        self._toolbar.device_lbl.setStyleSheet(
            f"color:{device_color}; font-family:'Share Tech Mono','Courier New',monospace;"
            f"font-size:10px; background:transparent; border:none;"
        )

        self._waveform = WaveformWidget(self._engine)
        self._sidebar  = SidebarWidget(self._engine)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._waveform, stretch=1)
        body_layout.addWidget(self._sidebar)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._toolbar)
        root_layout.addWidget(body, stretch=1)
        self.setCentralWidget(root)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(
            f"background:{C.BG_PANEL}; color:{C.TEXT_LO};"
            f"font-family:'Share Tech Mono','Courier New',monospace; font-size:10px;"
            f"border-top:1px solid {C.BORDER};"
        )
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            f"{'Real device' if is_real else '⚠ MockUnicorn (no hardware/SDK found)'}"
            f"  —  {SAMPLE_RATE} Hz  ·  ±150 µV  ·  {N_CHANNELS} channels"
        )

        # ── Wire controls ─────────────────────────────────────────────────────
        self._paused = False
        self._toolbar.pause_btn.clicked.connect(self._toggle_pause)
        self._toolbar.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        self._toolbar.window_combo.currentTextChanged.connect(self._on_window_changed)

        # ── Timers ────────────────────────────────────────────────────────────
        self._wave_timer = QTimer(self)
        self._wave_timer.timeout.connect(self._tick_wave)
        self._wave_timer.start(50)   # 20 fps — waveform repaint

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._sidebar.update_stats)
        self._stats_timer.start(200)  # 5 fps — quality / band power / stats

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)

        # ── Start acquisition ─────────────────────────────────────────────────
        try:
            self._engine.start()
            logger.info("BrainEngine started (%s).", label)
        except Exception as exc:
            logger.exception("Failed to start BrainEngine: %s", exc)
            self._status_bar.showMessage(f"⚠ Engine start failed: {exc}")

    # ── Timer slots ───────────────────────────────────────────────────────────
    def _tick_wave(self) -> None:
        try:
            self._engine.check_health()
        except RuntimeError as exc:
            self._wave_timer.stop()
            self._stats_timer.stop()
            self._status_bar.showMessage(f"⚠ Acquisition error: {exc}")
            return
        self._waveform.fetch()

    def _tick_clock(self) -> None:
        elapsed = int(time.monotonic() - self._start_time)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        total   = self._engine.buffer.total_written
        self._toolbar.rate_lbl.setText(
            f"{SAMPLE_RATE} Hz  ·  {N_CHANNELS} ch  ·  {h:02d}:{m:02d}:{s:02d}  "
            f"·  {total:,} samples"
        )

    # ── Control handlers ──────────────────────────────────────────────────────
    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._waveform.set_paused(self._paused)
        self._toolbar.pause_btn.setText(
            "▶  RESUME" if self._paused else "⏸  PAUSE"
        )

    def _on_scale_changed(self, text: str) -> None:
        if text == "AUTO":
            self._waveform.set_autoscale(True)
        else:
            uv = float(text.split()[0])
            self._waveform.set_scale(uv)

    def _on_window_changed(self, text: str) -> None:
        seconds = int(text.split()[0])
        self._waveform.set_window(seconds)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        self._wave_timer.stop()
        self._stats_timer.stop()
        self._clock_timer.stop()
        try:
            self._engine.stop()
        except Exception as exc:
            logger.warning("Engine stop: %s", exc)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Windows: request 1 ms timer resolution for accurate acquisition timing
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    forced_serial = sys.argv[1] if len(sys.argv) > 1 else ""

    app = QApplication(sys.argv)
    app.setApplicationName("Unicorn EEG Visualizer")

    win = EEGVisualizerWindow(forced_serial=forced_serial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
