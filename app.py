"""
app.py — Neuro-Lock Main Application (PyQt6)
High-speed stimulus display, enrollment wizard, and vault screen.
Sub-team 3 (UX/UI) owns this file.
"""


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars set manually or via shell
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, pyqtSlot, QPropertyAnimation,
    QEasingCurve, QRectF, QPointF,
)
from PyQt6.QtGui import (
    QFont, QFontDatabase, QPixmap, QPainter, QColor, QPainterPath,
    QLinearGradient, QRadialGradient, QPen, QBrush, QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QFrame, QGraphicsDropShadowEffect,
    QGridLayout, QProgressBar, QMessageBox, QDialog, QSizePolicy,
)

from brain_engine import BrainEngine, MockUnicorn, SAMPLE_RATE, N_CHANNELS
from Fase1.signal_processing import AuthenticationPipeline, AuthResult, EPOCH_DURATION_S
from Fase1.stimulus_runner import StimulusRunner, ParadigmConfig
from data_logger import SessionLogger
from erp_viewer import ERPViewer

logger = logging.getLogger(__name__)

# ── Asset directory ────────────────────────────────────────────────────────────
IMAGES_DIR = Path(__file__).parent / "assets" / "images"
IMAGE_COUNT = 20   # IDs 0–19

UNICORN_SERIAL = os.environ.get("UNICORN_SERIAL", "")  # set to "" for mock

# ══════════════════════════════════════════════════════════════════════════════
#  STIMULUS CONFIGURATION  — edit this block to change images or password
#
#  DEFAULT_PASSWORD_IDS  : which image IDs the user must mentally focus on.
#                          Always pick exactly 3.
#
#  STIMULUS_CATALOG      : one entry per image (index = image ID).
#                          Each tuple: (bg_color_hex, big_symbol, label)
#                          • Password images  → vivid color + large symbol + word
#                          • Distractor images → dark grey, small ID number only
#
#  If  assets/images/<id:02d>.png  exists for a given ID, that file is shown
#  instead of the generated placeholder — so you can swap in real pictures
#  without touching any other code.
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PASSWORD_IDS = [0, 1, 2]   # ← IDs the user focuses on (must match catalog)

STIMULUS_CATALOG = [
    # ══════════════════════════════════════════════════════════════════════════
    # ALL 20 images share the SAME background (#2E2E2E) and font size so that
    # no low-level visual difference (luminance, colour) drives the ERP.
    # The ONLY discriminating feature between images is their unique symbol —
    # which only triggers a cognitive P300 in the user who chose that symbol.
    #
    # This eliminates Visual Evoked Potentials (VEP) that would grant access
    # to anyone regardless of which images they are mentally attending to.
    # ══════════════════════════════════════════════════════════════════════════
    # (bg_color,    symbol, label      )
    ("#2E2E2E",     "★",    "STAR"     ),  # 0
    ("#2E2E2E",     "♦",    "DIAMOND"  ),  # 1
    ("#2E2E2E",     "●",    "CIRCLE"   ),  # 2
    ("#2E2E2E",     "▲",    "TRIANGLE" ),  # 3
    ("#2E2E2E",     "♠",    "SPADE"    ),  # 4
    ("#2E2E2E",     "♥",    "HEART"    ),  # 5
    ("#2E2E2E",     "♣",    "CLUB"     ),  # 6
    ("#2E2E2E",     "⬟",    "PENTA"    ),  # 7
    ("#2E2E2E",     "✦",    "SPARK"    ),  # 8
    ("#2E2E2E",     "⬡",    "HEX"      ),  # 9
    ("#2E2E2E",     "⊕",    "TARGET"   ),  # 10
    ("#2E2E2E",     "⊗",    "CROSS"    ),  # 11
    ("#2E2E2E",     "⬤",    "DOT"      ),  # 12
    ("#2E2E2E",     "▼",    "DOWN"     ),  # 13
    ("#2E2E2E",     "◆",    "RHOMBUS"  ),  # 14
    ("#2E2E2E",     "■",    "SQUARE"   ),  # 15
    ("#2E2E2E",     "✿",    "FLOWER"   ),  # 16
    ("#2E2E2E",     "⬢",    "BLOCK"    ),  # 17
    ("#2E2E2E",     "☽",    "MOON"     ),  # 18
    ("#2E2E2E",     "⬠",    "PENT2"    ),  # 19
]


# ── Color Palette ──────────────────────────────────────────────────────────────
class Colors:
    BG_DEEP    = "#060a10"
    BG_PANEL   = "#0d1520"
    ACCENT     = "#00e5ff"
    ACCENT2    = "#7c4dff"
    SUCCESS    = "#00e676"
    DANGER     = "#ff1744"
    TEXT_HI    = "#e8f4fd"
    TEXT_LO    = "#4a6478"
    BORDER     = "#1a2e42"


# ── Worker Thread for Paradigm ─────────────────────────────────────────────────
class ParadigmWorker(QObject):
    """
    Runs the stimulus sequence off the main thread, emitting Qt signals for
    UI updates (image show, blank, completion).
    """
    sig_show_image  = pyqtSignal(int, bool)    # image_id, is_target
    sig_show_blank  = pyqtSignal()
    sig_completed   = pyqtSignal(object)       # AuthResult

    def __init__(self, engine: BrainEngine, password_ids: list[int]):
        super().__init__()
        self._engine       = engine
        self._password_ids = password_ids
        self._pipeline     = AuthenticationPipeline(engine)
        self._pipeline.set_targets(password_ids)

        cfg = ParadigmConfig(
            total_images=IMAGE_COUNT,
            n_targets=len(password_ids),
        )
        self._runner = StimulusRunner(engine, cfg)
        self._runner.set_password_ids(password_ids)
        self._runner.set_callbacks(
            on_show     = lambda img_id, is_tgt: self.sig_show_image.emit(img_id, is_tgt),
            on_blank    = lambda: self.sig_show_blank.emit(),
            on_complete = self._on_complete,
        )

    def run(self) -> None:
        self._runner.run_sync()

    def _on_complete(self, events) -> None:
        # Wait for the last epoch's post-stimulus data to arrive in the buffer
        import time
        time.sleep(EPOCH_DURATION_S + 0.1)
        result = self._pipeline.evaluate()
        # Write session log
        try:
            session = SessionLogger()
            markers = self._engine.get_markers()
            for m in markers:
                epoch = self._engine.get_epoch(m)
                session.log_marker(m, epoch)
            session.log_auth_result(result, self._pipeline.get_erp_data())
            session.flush()
            logger.info("Session saved to %s", session.session_dir)
        except Exception as exc:
            logger.warning("Session logging failed: %s", exc)
        self.sig_completed.emit(result)

    def erp_data(self) -> dict:
        return self._pipeline.get_erp_data()


# ── Custom Widgets ─────────────────────────────────────────────────────────────

class GlowButton(QPushButton):
    def __init__(self, text: str, accent: str = Colors.ACCENT, parent=None):
        super().__init__(text, parent)
        self._accent = accent
        self.setFixedHeight(52)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(accent))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1.5px solid {accent};
                border-radius: 8px;
                color: {accent};
                font-family: 'Share Tech Mono', 'Courier New', monospace;
                font-size: 14px;
                letter-spacing: 3px;
                padding: 0 28px;
            }}
            QPushButton:hover {{
                background: {accent}22;
                color: {Colors.TEXT_HI};
            }}
            QPushButton:pressed {{
                background: {accent}44;
            }}
            QPushButton:disabled {{
                border-color: {Colors.TEXT_LO};
                color: {Colors.TEXT_LO};
            }}
        """)


class ScanlineWidget(QWidget):
    """Background widget that draws an animated neural-net / scanline effect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)   # ~30 fps background

    def _tick(self):
        self._phase += 0.005
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Deep background
        painter.fillRect(0, 0, w, h, QColor(Colors.BG_DEEP))

        # Subtle scanlines
        pen = QPen(QColor(255, 255, 255, 6))
        pen.setWidth(1)
        painter.setPen(pen)
        for y in range(0, h, 4):
            painter.drawLine(0, y, w, y)

        # Radial glow
        grad = QRadialGradient(w * 0.5, h * 0.3, h * 0.6)
        grad.setColorAt(0, QColor(0, 80, 120, 40))
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, h, grad)
        painter.end()


class NeuralDivider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {Colors.BORDER};")


# ── Live Signal Visualizer ─────────────────────────────────────────────────────

class LiveSignalVisualizerWidget(QWidget):
    """
    Continuously paints the last WINDOW_S seconds of raw EEG directly from the
    ring buffer.  All 8 Unicorn channels are drawn as stacked lanes, each
    auto-scaled to ±3σ so spikes on one channel don't crush the others.

    Uses read_from() instead of snapshot() — reads only 1 000 samples (4 s ×
    250 Hz) per refresh instead of the full 30 000-sample ring buffer, so there
    is no performance impact on the acquisition thread.
    """

    CH_NAMES  = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    CH_COLORS = [
        "#00e5ff", "#7c4dff", "#00e676", "#ff5252",
        "#ffab40", "#ea80fc", "#40c4ff", "#b2ff59",
    ]
    WINDOW_S  = 4
    N_SAMPLES = SAMPLE_RATE * WINDOW_S   # 1 000 samples @ 250 Hz

    def __init__(self, engine: BrainEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._data:   Optional[np.ndarray] = None   # shape (N_SAMPLES, 8)

        self.setMinimumSize(480, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent;")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._fetch)
        self._timer.start(50)   # 20 fps

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch(self) -> None:
        try:
            buf   = self._engine.buffer
            total = buf.total_written
            n     = self.N_SAMPLES
            if total < n:
                return
            data = buf.read_from(total - n, n)
            if data is not None:
                self._data = data
                self.update()
        except Exception:
            pass

    def start(self) -> None:
        self._timer.start(50)

    def stop(self) -> None:
        self._timer.stop()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        data = self._data
        if data is None or len(data) == 0:
            # Draw empty panel while waiting for data
            painter = QPainter(self)
            painter.fillRect(0, 0, self.width(), self.height(), QColor(Colors.BG_PANEL))
            painter.setPen(QPen(QColor(Colors.TEXT_LO)))
            painter.setFont(QFont("Share Tech Mono", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for EEG data…")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        ml        = 38          # left margin for channel labels
        mr        = 8
        mt        = 6
        mb        = 22          # bottom margin for time axis
        plot_w    = w - ml - mr
        plot_h    = h - mt - mb
        ch_h      = plot_h / N_CHANNELS
        n_samples = len(data)

        # Panel background
        painter.fillRect(0, 0, w, h, QColor(Colors.BG_PANEL))

        # Outer border
        border_pen = QPen(QColor(Colors.BORDER))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(ml, mt, plot_w, plot_h)

        font_label = QFont("Share Tech Mono", 8)
        painter.setFont(font_label)

        for ch in range(N_CHANNELS):
            y_top    = mt + ch * ch_h
            y_center = y_top + ch_h / 2

            # Lane separator
            if ch > 0:
                sep_pen = QPen(QColor(Colors.BORDER))
                sep_pen.setWidth(1)
                painter.setPen(sep_pen)
                painter.drawLine(ml, int(y_top), ml + plot_w, int(y_top))

            # Channel label (left gutter)
            painter.setPen(QPen(QColor(self.CH_COLORS[ch])))
            painter.drawText(2, int(y_center) + 4, self.CH_NAMES[ch])

            # Baseline rule
            base_pen = QPen(QColor(30, 46, 66, 100))
            base_pen.setWidth(1)
            painter.setPen(base_pen)
            painter.drawLine(ml, int(y_center), ml + plot_w, int(y_center))

            # Signal trace
            signal = data[:, ch]
            std    = float(np.std(signal)) or 1.0
            scale  = (ch_h * 0.42) / (3.0 * std)   # 3-sigma fills ~42 % of lane

            sig_pen = QPen(QColor(self.CH_COLORS[ch]))
            sig_pen.setWidth(1)
            painter.setPen(sig_pen)

            for i in range(1, n_samples):
                x1 = ml + (i - 1) / n_samples * plot_w
                x2 = ml + i       / n_samples * plot_w
                y1 = y_center - signal[i - 1] * scale
                y2 = y_center - signal[i]     * scale
                # Clamp to this lane's bounds
                y1 = max(y_top + 1,        min(y_top + ch_h - 1, y1))
                y2 = max(y_top + 1,        min(y_top + ch_h - 1, y2))
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Time-axis ticks: -4s … now
        painter.setPen(QPen(QColor(Colors.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        for t_s in range(self.WINDOW_S + 1):
            x = ml + t_s / self.WINDOW_S * plot_w
            painter.drawLine(int(x), mt + plot_h, int(x), mt + plot_h + 4)
            lbl = f"-{self.WINDOW_S - t_s}s" if t_s < self.WINDOW_S else "now"
            painter.drawText(int(x) - 12, mt + plot_h + 16, lbl)

        painter.end()


# ── Signal Monitor Screen ──────────────────────────────────────────────────────

class SignalMonitorScreen(QWidget):
    """
    Full-screen panel that shows the live raw EEG visualizer.
    Sits at index 5 in the QStackedWidget.
    """
    sig_back = pyqtSignal()

    def __init__(self, engine: BrainEngine, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("⚡  LIVE EEG SIGNAL MONITOR", self)
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 16px; font-weight: 700; "
            f"font-family: 'Share Tech Mono', monospace; letter-spacing: 4px;"
        )
        shadow = QGraphicsDropShadowEffect(title)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(Colors.ACCENT))
        shadow.setOffset(0, 0)
        title.setGraphicsEffect(shadow)
        header.addWidget(title)
        header.addStretch()

        self._sample_lbl = QLabel(f"{SAMPLE_RATE} Hz  ·  {N_CHANNELS} ch  ·  4 s window", self)
        self._sample_lbl.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 11px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        header.addWidget(self._sample_lbl)
        layout.addLayout(header)
        layout.addWidget(NeuralDivider())

        # ── Visualizer ────────────────────────────────────────────────────────
        self.visualizer = LiveSignalVisualizerWidget(engine, self)
        layout.addWidget(self.visualizer, stretch=1)

        layout.addWidget(NeuralDivider())

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QHBoxLayout()
        hint = QLabel(
            "Each channel auto-scaled to ±3σ  •  Traces refresh at 20 Hz  •  "
            "No raw data is written to disk from this view",
            self,
        )
        hint.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 10px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        footer.addWidget(hint)
        footer.addStretch()

        btn_back = GlowButton("←  BACK", Colors.ACCENT2)
        btn_back.setFixedWidth(180)
        btn_back.clicked.connect(self.sig_back)
        footer.addWidget(btn_back)
        layout.addLayout(footer)


# ── Stimulus Flash Screen ──────────────────────────────────────────────────────

class StimulusScreen(QWidget):
    """
    Full-screen display optimised for zero-lag image rendering.
    Maintains a pixmap cache so QLabel.setPixmap() is instantaneous.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel(self)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(600, 600)
        self._layout.addWidget(self._img_label)

        self._progress = QProgressBar(self)
        self._progress.setFixedHeight(4)
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {Colors.BG_PANEL}; border: none; }}
            QProgressBar::chunk {{ background: {Colors.ACCENT}; }}
        """)
        self._layout.addWidget(self._progress)

        self._status = QLabel("Preparing…", self)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 13px; "
            f"font-family: 'Share Tech Mono', monospace; letter-spacing: 2px;"
        )
        self._layout.addWidget(self._status)

        # Pre-load image cache
        self._cache: dict[int, QPixmap] = {}
        self._preload_images()

    def _preload_images(self):
        target_size = 480
        for img_id in range(IMAGE_COUNT):
            path = IMAGES_DIR / f"{img_id:02d}.png"
            if path.exists():
                px = QPixmap(str(path)).scaled(
                    target_size, target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._cache[img_id] = px
                continue

            # ── Generated image from STIMULUS_CATALOG ────────────────────────
            # All 20 images: SAME background, SAME font size, SAME colour.
            # Only the symbol and label text differ — so the ERP difference
            # between target and non-target images is purely cognitive (P300),
            # not a low-level Visual Evoked Potential (VEP) response to
            # brightness or colour changes.
            bg_hex, symbol, label = STIMULUS_CATALOG[img_id]
            px = QPixmap(target_size, target_size)
            px.fill(QColor(bg_hex))
            painter = QPainter(px)

            # Large centred symbol — identical font size for every image
            font_sym = QFont("Segoe UI Symbol", 150, QFont.Weight.Bold)
            painter.setFont(font_sym)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                0, 20, target_size, 300,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                symbol,
            )
            # Label text below symbol — same font, same colour, same position
            font_lbl = QFont("Arial", 54, QFont.Weight.Bold)
            painter.setFont(font_lbl)
            painter.setPen(QColor(255, 255, 255, 210))
            painter.drawText(
                0, 320, target_size, 120,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            painter.end()
            self._cache[img_id] = px

    @pyqtSlot(int, bool)
    def show_image(self, image_id: int, is_target: bool):
        """Called from ParadigmWorker signal — renders image immediately."""
        px = self._cache.get(image_id)
        if px:
            self._img_label.setPixmap(px)
        self._status.setText(f"[ {image_id:02d} ]")
        # Force synchronous repaint — don't wait for the next event-loop cycle
        self._img_label.repaint()

    @pyqtSlot()
    def show_blank(self):
        self._img_label.clear()
        self._status.setText("")
        self._img_label.repaint()

    def set_progress(self, value: int):
        self._progress.setValue(value)

    def set_status(self, text: str, color: str = Colors.TEXT_LO):
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {color}; font-size: 13px; "
            f"font-family: 'Share Tech Mono', monospace; letter-spacing: 2px;"
        )


# ── Result Screen ──────────────────────────────────────────────────────────────

class ResultScreen(QWidget):
    sig_retry      = pyqtSignal()
    sig_back       = pyqtSignal()
    sig_open_vault = pyqtSignal()   # emitted only when granted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")

        # ── Main layout: verdict panel left, ERP plot right ────────────────────
        outer = QHBoxLayout(self)
        outer.setContentsMargins(40, 30, 40, 30)
        outer.setSpacing(32)

        # ── Left: verdict ──────────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        left.setSpacing(18)

        self._icon = QLabel("", self)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size: 96px;")
        left.addWidget(self._icon)

        self._title = QLabel("", self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            f"font-size: 32px; font-weight: 700; color: {Colors.TEXT_HI}; "
            f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
        )
        left.addWidget(self._title)

        self._detail = QLabel("", self)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setWordWrap(True)
        self._detail.setMaximumWidth(340)
        self._detail.setStyleSheet(
            f"font-size: 12px; color: {Colors.TEXT_LO}; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        left.addWidget(self._detail)

        left.addWidget(NeuralDivider())

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self._btn_retry      = GlowButton("⟳  RETRY", Colors.ACCENT)
        self._btn_back       = GlowButton("←  BACK", Colors.TEXT_LO)
        self._btn_open_vault = GlowButton("▶  OPEN VAULT", Colors.SUCCESS)
        self._btn_retry.clicked.connect(self._on_retry)
        self._btn_back.clicked.connect(self._on_back)
        self._btn_open_vault.clicked.connect(self._on_open_vault)
        btn_row.addWidget(self._btn_back)
        btn_row.addWidget(self._btn_retry)
        btn_row.addWidget(self._btn_open_vault)
        left.addLayout(btn_row)

        outer.addLayout(left, stretch=0)

        # ── Vertical divider ───────────────────────────────────────────────────
        vdiv = QFrame(self)
        vdiv.setFixedWidth(1)
        vdiv.setStyleSheet(f"background: {Colors.BORDER};")
        outer.addWidget(vdiv)

        # ── Right: live ERP viewer ─────────────────────────────────────────────
        self._erp_viewer = ERPViewer(parent=self)
        outer.addWidget(self._erp_viewer, stretch=1)

    # ── Internal slot wrappers — stop ERP refresh before emitting nav signals ──

    def _on_retry(self):
        self._erp_viewer.stop_live()
        self.sig_retry.emit()

    def _on_back(self):
        self._erp_viewer.stop_live()
        self.sig_back.emit()

    def _on_open_vault(self):
        self._erp_viewer.stop_live()
        self.sig_open_vault.emit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def show_result(self, result: AuthResult, pipeline=None):
        """Populate the verdict panel and kick the ERP display."""
        if result.granted:
            self._icon.setText("🔓")
            self._title.setText("ACCESS GRANTED")
            self._title.setStyleSheet(
                f"font-size: 32px; font-weight: 700; color: {Colors.SUCCESS}; "
                f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
            )
            self._btn_open_vault.setVisible(True)
            self._btn_retry.setVisible(False)
        else:
            self._icon.setText("🔒")
            self._title.setText("ACCESS DENIED")
            self._title.setStyleSheet(
                f"font-size: 32px; font-weight: 700; color: {Colors.DANGER}; "
                f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
            )
            self._btn_open_vault.setVisible(False)
            self._btn_retry.setVisible(True)

        self._detail.setText(
            f"{result.message}\n\n"
            f"Target: {result.target_peak_uv:.2f} µV  |  "
            f"Non-target: {result.nontarget_peak_uv:.2f} µV\n"
            f"SNR: {result.snr_db:.1f} dB"
        )

        # Wire up and start the live ERP plot (refreshes every 400 ms so the
        # waveform appears to "settle" cinematically after the scan ends)
        if pipeline is not None:
            self._erp_viewer.set_pipeline(pipeline)
            self._erp_viewer.start_live(interval_ms=400)


# ── Vault Screen ───────────────────────────────────────────────────────────────

class VaultScreen(QWidget):
    """
    Displayed after a successful P300 authentication.
    Shows a grid of password cards; each card has an eye-button to reveal/hide
    the stored credential.
    """
    sig_lock = pyqtSignal()   # emitted when user locks vault and returns to home

    # ── Example password vault entries ─────────────────────────────────────────
    VAULT_ENTRIES = [
        ("🐙  GitHub",          "Gr4p3_P1ck3r!2024"),
        ("📧  Gmail",           "S0lar_Fl4re#99"),
        ("💼  LinkedIn",        "ProN3tw0rk@2024"),
        ("☁️  AWS Console",     "Cl0ud9!AWS_Root"),
        ("🏦  Online Banking",  "S3cur3B4nk#Vault"),
        ("🎬  Netflix",         "B!ng3W4tch_2024"),
        ("📡  Wi-Fi Router",    "H0m3N3t_Adm1n!"),
        ("₿  Crypto Wallet",   "B1tc0in$H0DL999"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 36, 60, 36)
        layout.setSpacing(14)

        # ── Header ─────────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        icon_lbl = QLabel("🔓", self)
        icon_lbl.setStyleSheet("font-size: 38px;")
        header_row.addWidget(icon_lbl)

        title_lbl = QLabel("VAULT UNLOCKED", self)
        title_lbl.setStyleSheet(
            f"color: {Colors.SUCCESS}; font-size: 30px; font-weight: 800; "
            f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 6px;"
        )
        shadow = QGraphicsDropShadowEffect(title_lbl)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(Colors.SUCCESS))
        shadow.setOffset(0, 0)
        title_lbl.setGraphicsEffect(shadow)
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        sub_lbl = QLabel(
            "Neural identity confirmed  —  your credentials are displayed below.", self
        )
        sub_lbl.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 12px; letter-spacing: 2px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(sub_lbl)
        layout.addWidget(NeuralDivider())
        layout.addSpacing(4)

        # ── Password card grid (2 columns) ────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(10)
        for idx, (service, password) in enumerate(self.VAULT_ENTRIES):
            card = self._make_card(service, password)
            grid.addWidget(card, idx // 2, idx % 2)
        layout.addLayout(grid)

        layout.addStretch()
        layout.addWidget(NeuralDivider())

        # ── Footer ─────────────────────────────────────────────────────────────
        footer_row = QHBoxLayout()
        hint = QLabel("Click 👁 to reveal  •  Vault auto-locks on close", self)
        hint.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 11px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        footer_row.addWidget(hint)
        footer_row.addStretch()

        btn_lock = GlowButton("🔒  LOCK VAULT", Colors.DANGER)
        btn_lock.setFixedWidth(220)
        btn_lock.clicked.connect(self.sig_lock)
        footer_row.addWidget(btn_lock)
        layout.addLayout(footer_row)

    # ── Card builder ───────────────────────────────────────────────────────────

    def _make_card(self, service: str, password: str) -> QFrame:
        card = QFrame(self)
        card.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_PANEL}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(6)

        svc_lbl = QLabel(service, card)
        svc_lbl.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 11px; letter-spacing: 2px; "
            f"font-family: 'Share Tech Mono', monospace; border: none; background: transparent;"
        )
        cl.addWidget(svc_lbl)

        pw_row = QHBoxLayout()
        pw_lbl = QLabel("••••••••••••••••", card)
        pw_lbl.setStyleSheet(
            f"color: {Colors.TEXT_HI}; font-size: 14px; font-weight: 600; "
            f"font-family: 'Share Tech Mono', monospace; letter-spacing: 2px; "
            f"border: none; background: transparent;"
        )
        pw_row.addWidget(pw_lbl, stretch=1)

        eye_btn = QPushButton("👁", card)
        eye_btn.setFixedSize(30, 30)
        eye_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 4px; color: {Colors.TEXT_LO}; font-size: 13px; }}"
            f"QPushButton:hover {{ border-color: {Colors.ACCENT}; color: {Colors.ACCENT}; }}"
        )
        # Closure-safe toggle
        revealed = [False]
        def _toggle(_, lbl=pw_lbl, pw=password, state=revealed):
            state[0] = not state[0]
            lbl.setText(pw if state[0] else "••••••••••••••••")
        eye_btn.clicked.connect(_toggle)
        pw_row.addWidget(eye_btn)

        cl.addLayout(pw_row)
        return card


# ── Enrollment Screen ──────────────────────────────────────────────────────────

class EnrollmentScreen(QWidget):
    """
    Shows all 20 stimulus images in a grid.  User clicks exactly 3 to set as
    their mental password, then clicks CONFIRM.
    """
    sig_confirmed = pyqtSignal(list)   # emits list of 3 selected image IDs

    _THUMB = 110    # thumbnail size (px)
    _COLS  = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")
        self._selected: list[int] = []
        self._buttons: dict[int, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(14)

        title = QLabel("ENROLLMENT — Choose Your 3 Password Images", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 18px; font-weight: 700; "
            f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
        )
        layout.addWidget(title)

        self._hint = QLabel("Click 3 images to select your password  (0 / 3 selected)", self)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 12px; font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(self._hint)
        layout.addWidget(NeuralDivider())

        # ── Image grid ────────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(8)
        for img_id in range(IMAGE_COUNT):
            btn = QPushButton(self)
            btn.setFixedSize(self._THUMB, self._THUMB)
            btn.setCheckable(True)
            px = self._make_thumb(img_id)
            btn.setIcon(px)
            from PyQt6.QtCore import QSize
            btn.setIconSize(QSize(self._THUMB - 4, self._THUMB - 4))
            btn.setStyleSheet(self._btn_style(False))
            btn.clicked.connect(lambda checked, i=img_id: self._toggle(i))
            self._buttons[img_id] = btn
            grid.addWidget(btn, img_id // self._COLS, img_id % self._COLS)
        layout.addLayout(grid)

        layout.addWidget(NeuralDivider())

        row = QHBoxLayout()
        self._btn_confirm = GlowButton("✓  CONFIRM PASSWORD", Colors.SUCCESS)
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self._confirm)
        row.addStretch()
        row.addWidget(self._btn_confirm)
        layout.addLayout(row)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_thumb(self, img_id: int) -> "QIcon":
        from PyQt6.QtGui import QIcon
        path = IMAGES_DIR / f"{img_id:02d}.png"
        if path.exists():
            return QIcon(str(path))
        bg_hex, symbol, label = STIMULUS_CATALOG[img_id]
        px = QPixmap(self._THUMB, self._THUMB)
        px.fill(QColor(bg_hex))
        painter = QPainter(px)
        # All catalog entries have symbols — same font/colour for every thumbnail
        painter.setFont(QFont("Segoe UI Symbol", 36, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        painter.end()
        return QIcon(px)

    @staticmethod
    def _btn_style(selected: bool) -> str:
        if selected:
            return (f"QPushButton {{ border: 3px solid {Colors.SUCCESS}; "
                    f"border-radius: 6px; background: {Colors.SUCCESS}22; }}")
        return (f"QPushButton {{ border: 2px solid {Colors.BORDER}; "
                f"border-radius: 6px; background: {Colors.BG_PANEL}; }}"
                f"QPushButton:hover {{ border-color: {Colors.ACCENT}; }}")

    def _toggle(self, img_id: int):
        if img_id in self._selected:
            self._selected.remove(img_id)
            self._buttons[img_id].setStyleSheet(self._btn_style(False))
        elif len(self._selected) < 3:
            self._selected.append(img_id)
            self._buttons[img_id].setStyleSheet(self._btn_style(True))
        n = len(self._selected)
        self._hint.setText(
            f"Click 3 images to select your password  ({n} / 3 selected)"
        )
        self._btn_confirm.setEnabled(n == 3)

    def _confirm(self):
        self.sig_confirmed.emit(list(self._selected))

    def reset(self):
        """Clear selection (call before showing screen again)."""
        for img_id in list(self._selected):
            self._buttons[img_id].setStyleSheet(self._btn_style(False))
        self._selected.clear()
        self._hint.setText("Click 3 images to select your password  (0 / 3 selected)")
        self._btn_confirm.setEnabled(False)


# ── Home Screen ────────────────────────────────────────────────────────────────

class HomeScreen(QWidget):
    sig_start_auth = pyqtSignal()
    sig_enroll     = pyqtSignal()
    sig_monitor    = pyqtSignal()   # open live signal monitor

    _CH_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    def __init__(self, parent=None):
        super().__init__(parent)
        bg = ScanlineWidget(self)
        bg.setGeometry(0, 0, 9999, 9999)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)
        layout.setContentsMargins(80, 60, 80, 60)

        logo = QLabel("🧠", self)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("font-size: 72px;")
        layout.addWidget(logo)

        title = QLabel("NEURO-LOCK", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 48px; font-weight: 800; "
            f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 8px;"
        )
        shadow = QGraphicsDropShadowEffect(title)
        shadow.setBlurRadius(32)
        shadow.setColor(QColor(Colors.ACCENT))
        shadow.setOffset(0, 0)
        title.setGraphicsEffect(shadow)
        layout.addWidget(title)

        sub = QLabel("BRAINWAVE PASSWORD MANAGER", self)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 13px; letter-spacing: 4px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(sub)

        layout.addSpacing(16)
        layout.addWidget(NeuralDivider())
        layout.addSpacing(10)

        # ── Device status ─────────────────────────────────────────────────────
        self._status_lbl = QLabel("DEVICE: CONNECTING…", self)
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(
            f"color: {Colors.ACCENT2}; font-size: 12px; letter-spacing: 3px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(self._status_lbl)

        # ── Signal quality bar (8 channels) ──────────────────────────────────
        self._sq_frame = QFrame(self)
        self._sq_frame.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_PANEL}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; }}"
        )
        sq_layout = QHBoxLayout(self._sq_frame)
        sq_layout.setContentsMargins(12, 8, 12, 8)
        sq_layout.setSpacing(10)
        sq_lbl = QLabel("SIGNAL:", self._sq_frame)
        sq_lbl.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 10px; letter-spacing: 2px; "
            f"font-family: 'Share Tech Mono', monospace; border: none; background: transparent;"
        )
        sq_layout.addWidget(sq_lbl)
        self._ch_dots: list[QLabel] = []
        for ch in self._CH_NAMES:
            dot = QLabel(ch, self._sq_frame)
            dot.setFixedSize(38, 22)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(
                f"background: {Colors.TEXT_LO}; border-radius: 4px; "
                f"color: #000; font-size: 9px; font-weight: bold; border: none;"
            )
            sq_layout.addWidget(dot)
            self._ch_dots.append(dot)
        sq_layout.addStretch()
        layout.addWidget(self._sq_frame)

        # ── Password preview ──────────────────────────────────────────────────
        self._pw_lbl = QLabel("PASSWORD: ★ STAR  ♦ DIAMOND  ● CIRCLE", self)
        self._pw_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pw_lbl.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 11px; letter-spacing: 2px; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(self._pw_lbl)

        layout.addSpacing(8)

        self._btn_auth = GlowButton("▶  BEGIN NEURAL SCAN", Colors.ACCENT)
        self._btn_auth.setFixedWidth(360)
        self._btn_auth.clicked.connect(self.sig_start_auth)
        self._btn_auth.setEnabled(False)
        layout.addWidget(self._btn_auth, alignment=Qt.AlignmentFlag.AlignCenter)

        self._btn_enroll = GlowButton("⚙  SET PASSWORD IMAGES", Colors.ACCENT2)
        self._btn_enroll.setFixedWidth(360)
        self._btn_enroll.clicked.connect(self.sig_enroll)
        layout.addWidget(self._btn_enroll, alignment=Qt.AlignmentFlag.AlignCenter)

        self._btn_monitor = GlowButton("📊  SIGNAL MONITOR", Colors.TEXT_LO)
        self._btn_monitor.setFixedWidth(360)
        self._btn_monitor.clicked.connect(self.sig_monitor)
        layout.addWidget(self._btn_monitor, alignment=Qt.AlignmentFlag.AlignCenter)

        hint = QLabel(
            "Focus on the images you chose as your password.\n"
            "Keep still. Blink minimally during the scan.", self
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {Colors.TEXT_LO}; font-size: 11px; font-style: italic;"
        )
        layout.addWidget(hint)

    def set_device_ready(self, ready: bool, label: str = ""):
        if ready:
            self._status_lbl.setText(f"DEVICE: READY  —  {label}")
            self._status_lbl.setStyleSheet(
                f"color: {Colors.SUCCESS}; font-size: 12px; letter-spacing: 3px; "
                f"font-family: 'Share Tech Mono', monospace;"
            )
            self._btn_auth.setEnabled(True)
        else:
            self._status_lbl.setText(f"DEVICE: {label.upper() or 'CONNECTING…'}")
            self._btn_auth.setEnabled(False)

    def set_password_preview(self, password_ids: list[int]):
        parts = []
        for pid in password_ids:
            _, sym, lbl = STIMULUS_CATALOG[pid]
            parts.append(f"{sym} {lbl}" if sym else str(pid))
        self._pw_lbl.setText("PASSWORD:  " + "   ".join(parts))

    def update_signal_quality(self, channel_statuses: list[dict]):
        color_map = {"OK": Colors.SUCCESS, "POOR": Colors.DANGER, "SATURATED": "#ffaa00"}
        for i, ch in enumerate(channel_statuses[:len(self._ch_dots)]):
            c = color_map.get(ch["status"], Colors.TEXT_LO)
            self._ch_dots[i].setStyleSheet(
                f"background: {c}; border-radius: 4px; "
                f"color: #000; font-size: 9px; font-weight: bold; border: none;"
            )


# ── Main Window ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Lock")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet(f"background: {Colors.BG_DEEP};")

        if UNICORN_SERIAL:
            self._engine = BrainEngine(serial=UNICORN_SERIAL)
        else:
            logger.info("No UNICORN_SERIAL set — using MockUnicorn.")
            self._engine = BrainEngine(device=MockUnicorn())

        self._password_ids = list(DEFAULT_PASSWORD_IDS)

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        self._home_screen          = HomeScreen(self)
        self._stimulus_screen      = StimulusScreen(self)
        self._result_screen        = ResultScreen(self)
        self._vault_screen         = VaultScreen(self)
        self._enrollment_screen    = EnrollmentScreen(self)
        self._signal_monitor_screen = SignalMonitorScreen(self._engine, self)

        self._stack.addWidget(self._home_screen)            # 0
        self._stack.addWidget(self._stimulus_screen)        # 1
        self._stack.addWidget(self._result_screen)          # 2
        self._stack.addWidget(self._vault_screen)           # 3
        self._stack.addWidget(self._enrollment_screen)      # 4
        self._stack.addWidget(self._signal_monitor_screen)  # 5

        self._home_screen.sig_start_auth.connect(self._start_authentication)
        self._home_screen.sig_enroll.connect(self._go_enrollment)
        self._home_screen.sig_monitor.connect(self._go_signal_monitor)
        self._result_screen.sig_retry.connect(self._start_authentication)
        self._result_screen.sig_back.connect(self._go_home)
        self._result_screen.sig_open_vault.connect(self._open_vault)
        self._vault_screen.sig_lock.connect(self._go_home)
        self._enrollment_screen.sig_confirmed.connect(self._on_enrollment_done)
        self._signal_monitor_screen.sig_back.connect(self._go_home)

        self._sq_timer = QTimer(self)
        self._sq_timer.timeout.connect(self._update_signal_quality)
        self._sq_timer.start(1500)

        self._paradigm_thread: Optional[QThread] = None
        self._worker: Optional[ParadigmWorker]   = None

        QTimer.singleShot(200, self._init_engine)

    def _init_engine(self):
        try:
            self._engine.start()
            label = (UNICORN_SERIAL if UNICORN_SERIAL else "SIMULATOR")
            self._home_screen.set_device_ready(True, label)
        except Exception as exc:
            logger.exception("Failed to start BrainEngine: %s", exc)
            self._home_screen.set_device_ready(False, str(exc)[:40])
            QMessageBox.critical(
                self, "Device Error",
                f"Could not connect to Unicorn device:\n{exc}\n\nFalling back to simulator.",
            )
            self._engine = BrainEngine(device=MockUnicorn())
            self._engine.start()
            self._home_screen.set_device_ready(True, "SIMULATOR (fallback)")

    def _go_home(self):
        # Pause the signal visualizer if it was running to save CPU
        self._signal_monitor_screen.visualizer.stop()
        self._stack.setCurrentIndex(0)

    def _go_signal_monitor(self):
        self._signal_monitor_screen.visualizer.start()
        self._stack.setCurrentIndex(5)

    def _open_vault(self):
        self._stack.setCurrentIndex(3)

    def _go_enrollment(self):
        self._enrollment_screen.reset()
        self._stack.setCurrentIndex(4)

    def _on_enrollment_done(self, selected_ids: list):
        self._password_ids = selected_ids
        self._home_screen.set_password_preview(selected_ids)
        logger.info("Password enrolled: image IDs %s", selected_ids)
        self._stack.setCurrentIndex(0)

    def _update_signal_quality(self):
        try:
            from data_logger import ImpedanceChecker
            snap = self._engine.buffer.snapshot()
            report = ImpedanceChecker().check(snap)
            self._home_screen.update_signal_quality(report)
        except Exception:
            pass

    @pyqtSlot()
    def _start_authentication(self):
        try:
            self._engine.check_health()
        except RuntimeError as exc:
            QMessageBox.critical(self, "Device Error", f"EEG device error:\n{exc}")
            return

        # Stop signal-quality timer — its 1.9 MB snapshot every 1.5 s on the main
        # thread competes with stimulus rendering and causes timing jitter.
        self._sq_timer.stop()

        self._engine.clear_markers()
        self._stimulus_screen.set_status("PREPARING NEURAL SCAN…", Colors.TEXT_LO)
        self._stack.setCurrentIndex(1)

        self._worker = ParadigmWorker(self._engine, self._password_ids)
        self._paradigm_thread = QThread()
        self._worker.moveToThread(self._paradigm_thread)

        self._worker.sig_show_image.connect(self._stimulus_screen.show_image)
        self._worker.sig_show_blank.connect(self._stimulus_screen.show_blank)
        self._worker.sig_completed.connect(self._on_paradigm_done)

        self._paradigm_thread.started.connect(self._worker.run)
        self._paradigm_thread.start()
        # setPriority must be called AFTER start() — thread must be running
        self._paradigm_thread.setPriority(QThread.Priority.TimeCriticalPriority)

        self._stimulus_screen.set_status("SCANNING…  FOCUS ON YOUR KEY IMAGES", Colors.ACCENT)

    @pyqtSlot(object)
    def _on_paradigm_done(self, result: AuthResult):
        if self._paradigm_thread:
            self._paradigm_thread.quit()
            self._paradigm_thread.wait()
        # Restart signal-quality polling now that the paradigm is done
        self._sq_timer.start(1500)
        # Pass the pipeline so ResultScreen can drive the live ERP plot
        pipeline = self._worker._pipeline if self._worker else None
        self._result_screen.show_result(result, pipeline=pipeline)
        self._stack.setCurrentIndex(2)

    def closeEvent(self, event):
        self._sq_timer.stop()
        if self._paradigm_thread and self._paradigm_thread.isRunning():
            self._paradigm_thread.quit()
            self._paradigm_thread.wait(2000)
        self._engine.stop()
        super().closeEvent(event)


# ── Entry Point ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # ── Windows: set timer resolution to 1 ms ─────────────────────────────────
    # Default Windows timer granularity is 15.6 ms, which causes large scheduling
    # jitter in the stimulus timing thread.  timeBeginPeriod(1) requests 1 ms
    # resolution for the lifetime of this process.
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        logger.info("Windows timer resolution set to 1 ms.")
    except Exception:
        pass   # Non-Windows or call not available — acceptable

    app = QApplication(sys.argv)
    app.setApplicationName("Neuro-Lock")
    try:
        for font_path in (Path(__file__).parent / "assets" / "fonts").glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font_path))
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # ── Windows: set timer resolution to 1 ms ────────────────────────────────
    # Default Windows timer granularity is 15.6 ms, which causes large scheduling
    # jitter in the stimulus timing thread.  timeBeginPeriod(1) requests 1 ms
    # resolution for the lifetime of this process.
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        logger.info("Windows timer resolution set to 1 ms.")
    except Exception:
        pass   # Non-Windows or call not available — acceptable

    app = QApplication(sys.argv)
    app.setApplicationName("Neuro-Lock")
    try:
        for font_path in (Path(__file__).parent / "assets" / "fonts").glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font_path))
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
