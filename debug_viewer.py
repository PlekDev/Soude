"""
debug_viewer.py — Soude Session Replay Debugger
Loads any saved log session and shows:
  • Scrollable stimulus timeline with ISI analysis
  • Per-epoch inspector (click any event to inspect its 800 ms epoch)
  • ERP grand averages (target vs non-target)
  • Marker table with all event metadata
  • Timing stats / ISI histogram

Usage:
    python debug_viewer.py                     # opens with session picker
    python debug_viewer.py logs/20260425_122001  # load specific session
"""

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSlot
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView, QAbstractItemView,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
class C:
    BG_DEEP   = "#060a10"
    BG_PANEL  = "#0d1520"
    BG_CARD   = "#101828"
    ACCENT    = "#00e5ff"
    SUCCESS   = "#00e676"
    DANGER    = "#ff1744"
    WARN      = "#ffab40"
    TEXT_HI   = "#e8f4fd"
    TEXT_LO   = "#4a6478"
    BORDER    = "#1a2e42"
    TARGET    = "#00e676"
    NONTARGET = "#ff5252"
    P300_ZONE = "#00e5ff"

CH_NAMES   = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
P300_CH    = [2, 4, 6]   # Cz, Pz, Oz
CH_COLORS  = ["#00e5ff","#7c4dff","#00e676","#ff5252",
              "#ffab40","#ea80fc","#40c4ff","#b2ff59"]
SAMPLE_RATE = 250
EPOCH_SAMPLES = 200          # 800 ms at 250 Hz
P300_ONSET_MS  = 250.0
P300_OFFSET_MS = 500.0
P300_ONSET  = int(P300_ONSET_MS  / 1000 * SAMPLE_RATE)   # 62
P300_OFFSET = int(P300_OFFSET_MS / 1000 * SAMPLE_RATE)   # 125

# Symbol catalog (mirrors app.py STIMULUS_CATALOG)
CATALOG = [
    ("★","STAR"),("♦","DIAMOND"),("●","CIRCLE"),("▲","TRIANGLE"),
    ("♠","SPADE"),("♥","HEART"),("♣","CLUB"),("⬟","PENTA"),
    ("✦","SPARK"),("⬡","HEX"),("⊕","TARGET"),("⊗","CROSS"),
    ("⬤","DOT"),("▼","DOWN"),("◆","RHOMBUS"),("■","SQUARE"),
    ("✿","FLOWER"),("⬢","BLOCK"),("☽","MOON"),("⬠","PENT2"),
]

LOGS_DIR = Path(__file__).parent / "logs"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MarkerRow:
    index:        int
    image_id:     int
    is_target:    bool
    buffer_index: int
    timestamp:    float
    isi_s:        float = 0.0      # inter-stimulus interval from previous event
    epoch:        Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def symbol(self) -> str:
        return CATALOG[self.image_id][0] if self.image_id < len(CATALOG) else "?"

    @property
    def label(self) -> str:
        return CATALOG[self.image_id][1] if self.image_id < len(CATALOG) else str(self.image_id)


@dataclass
class SessionData:
    session_id:   str
    markers:      list[MarkerRow]
    auth_result:  dict
    erp_data:     Optional[dict]
    n_epochs_saved: int

    @property
    def duration_s(self) -> float:
        if len(self.markers) < 2:
            return 0.0
        return self.markers[-1].timestamp - self.markers[0].timestamp

    @property
    def granted(self) -> bool:
        return bool(self.auth_result.get("granted", False))

    @property
    def targets(self) -> list[MarkerRow]:
        return [m for m in self.markers if m.is_target]

    @property
    def nontargets(self) -> list[MarkerRow]:
        return [m for m in self.markers if not m.is_target]


def load_session(path: Path) -> SessionData:
    """Load a session from a logs/<id>/ directory."""
    markers: list[MarkerRow] = []

    # markers.csv
    csv_path = path / "markers.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                markers.append(MarkerRow(
                    index=int(row["index"]),
                    image_id=int(row["image_id"]),
                    is_target=bool(int(row["is_target"])),
                    buffer_index=int(row["buffer_index"]),
                    timestamp=float(row["timestamp"]),
                ))

    # ISI computation
    for i in range(1, len(markers)):
        markers[i].isi_s = markers[i].timestamp - markers[i - 1].timestamp

    # epochs.npz
    npz_path = path / "epochs.npz"
    n_saved = 0
    if npz_path.exists():
        data = np.load(str(npz_path))
        n_saved = len(data.files)
        for m in markers:
            key = f"epoch_{m.index:04d}"
            if key in data:
                m.epoch = data[key]   # shape (200, 8) raw µV with DC offset

    # auth_result.json
    auth_result = {}
    erp_data = None
    json_path = path / "auth_result.json"
    if json_path.exists():
        with open(json_path) as f:
            auth_result = json.load(f)
        erp_data = auth_result.get("erp_data")

    return SessionData(
        session_id=path.name,
        markers=markers,
        auth_result=auth_result,
        erp_data=erp_data,
        n_epochs_saved=n_saved,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def mono_label(text: str, size: int = 10, color: str = C.TEXT_LO) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{color}; font-family:'Share Tech Mono','Courier New',monospace;"
        f"font-size:{size}px; background:transparent; border:none;"
    )
    return lbl


def divider() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{C.BORDER};")
    return f


def section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{C.ACCENT}; font-family:'Share Tech Mono','Courier New',monospace;"
        f"font-size:10px; letter-spacing:3px; background:transparent; border:none;"
        f"padding:4px 0;"
    )
    return lbl


# ── Timeline Widget ───────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    """
    Horizontal scrollable timeline showing all stimulus events.
    Each event is a vertical coloured tick.  Click a tick to select its epoch.
    """

    TICK_H     = 60
    LABEL_H    = 20
    TOTAL_H    = TICK_H + LABEL_H + 40    # ticks + labels + ISI row
    MARGIN_L   = 40
    MARGIN_R   = 20
    PX_PER_S   = 10.0          # pixels per second (will auto-fit to width)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: Optional[SessionData] = None
        self._selected: int = -1
        self._callbacks: list = []
        self.setMinimumHeight(self.TOTAL_H)
        self.setMinimumWidth(800)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background:{C.BG_PANEL};")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_session(self, session: SessionData) -> None:
        self._session = session
        self._selected = -1
        self._update_scale()
        self.update()

    def on_select(self, cb) -> None:
        self._callbacks.append(cb)

    def _update_scale(self) -> None:
        if self._session and self._session.markers:
            duration = self._session.duration_s + 2.0
            available = self.width() - self.MARGIN_L - self.MARGIN_R
            self.PX_PER_S = max(6.0, available / duration)

    def _t_to_x(self, t: float) -> float:
        if not self._session or not self._session.markers:
            return self.MARGIN_L
        t0 = self._session.markers[0].timestamp
        return self.MARGIN_L + (t - t0) * self.PX_PER_S

    def resizeEvent(self, event):
        self._update_scale()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if not self._session:
            return
        mx = event.position().x()
        best_dist = 12.0
        best_idx  = -1
        for m in self._session.markers:
            x = self._t_to_x(m.timestamp)
            if abs(mx - x) < best_dist:
                best_dist = abs(mx - x)
                best_idx  = m.index
        if best_idx != self._selected:
            self._selected = best_idx
            self.update()
            for cb in self._callbacks:
                cb(best_idx)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(C.BG_PANEL))

        if not self._session or not self._session.markers:
            painter.setPen(QPen(QColor(C.TEXT_LO)))
            painter.setFont(QFont("Share Tech Mono", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No session loaded")
            painter.end()
            return

        t0     = self._session.markers[0].timestamp
        tick_y = 30
        isi_y  = tick_y + self.TICK_H + 4

        # Background ISI row
        painter.fillRect(0, isi_y - 4, w, 20, QColor(C.BG_CARD))

        # Expected ISI line
        SOA_S, BLANK_S = 0.500, 0.075
        expected_isi = SOA_S + BLANK_S   # 575 ms
        painter.setFont(QFont("Share Tech Mono", 7))
        painter.setPen(QPen(QColor(C.ACCENT), 1, Qt.PenStyle.DashLine))
        # (no horizontal line — too cluttered; just label the expected value)

        # Draw each event
        for m in self._session.markers:
            x = self._t_to_x(m.timestamp)
            if x < 0 or x > w:
                continue

            color = QColor(C.TARGET if m.is_target else C.NONTARGET)
            is_sel = (m.index == self._selected)

            # Selection highlight
            if is_sel:
                hl = QColor(color)
                hl.setAlpha(30)
                painter.fillRect(int(x) - 6, tick_y - 4,
                                 12, self.TICK_H + 28, hl)

            # Tick line
            pen = QPen(color, 3 if is_sel else 1.5)
            painter.setPen(pen)
            painter.drawLine(int(x), tick_y, int(x), tick_y + self.TICK_H)

            # Symbol label (only for every 5th or selected to avoid clutter)
            if is_sel or m.index % 5 == 0:
                painter.setFont(QFont("Segoe UI Symbol", 9 if is_sel else 7))
                painter.setPen(QPen(color))
                painter.drawText(int(x) - 8, tick_y - 2, m.symbol)

            # ISI badge below tick (red if > 650 ms jitter)
            if m.isi_s > 0:
                isi_ms = m.isi_s * 1000
                isi_color = QColor(C.DANGER) if abs(isi_ms - expected_isi * 1000) > 75 else QColor(C.TEXT_LO)
                painter.setPen(QPen(isi_color))
                painter.setFont(QFont("Share Tech Mono", 6))
                painter.drawText(int(x) - 10, isi_y + 13, f"{isi_ms:.0f}")

        # Target/nontarget legend
        painter.setFont(QFont("Share Tech Mono", 8))
        painter.setPen(QPen(QColor(C.TARGET)))
        painter.drawText(w - 140, tick_y + self.TICK_H // 2, "▌ Target")
        painter.setPen(QPen(QColor(C.NONTARGET)))
        painter.drawText(w - 80,  tick_y + self.TICK_H // 2, "▌ Non-target")

        # Time axis
        duration = self._session.duration_s
        painter.setPen(QPen(QColor(C.BORDER)))
        painter.drawLine(self.MARGIN_L, tick_y + self.TICK_H,
                         int(self._t_to_x(t0 + duration + 1)),
                         tick_y + self.TICK_H)
        painter.setFont(QFont("Share Tech Mono", 7))
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        for t_sec in range(0, int(duration) + 2, 5):
            x = self._t_to_x(t0 + t_sec)
            painter.drawLine(int(x), tick_y + self.TICK_H,
                             int(x), tick_y + self.TICK_H + 4)
            painter.drawText(int(x) - 10, tick_y + self.TICK_H + 16, f"{t_sec}s")

        # ISI label
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        painter.drawText(2, isi_y + 13, "ISI ms↓")

        painter.end()


# ── Epoch Inspector Widget ────────────────────────────────────────────────────

class EpochWidget(QWidget):
    """
    Shows a single epoch (200 samples × 8 channels).
    Channels are stacked; P300 channels highlighted; P300 window in cyan.
    The grand-average ERP for the same class can be overlaid.
    """

    ML, MR, MT, MB = 38, 12, 14, 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self._marker: Optional[MarkerRow] = None
        self._erp_avg: Optional[np.ndarray] = None   # (200, 8) grand average
        self._t_ms = np.linspace(0, 800, EPOCH_SAMPLES)
        self.setMinimumSize(420, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:{C.BG_PANEL};")

    def set_marker(self, marker: MarkerRow,
                   erp_avg: Optional[np.ndarray] = None) -> None:
        self._marker  = marker
        self._erp_avg = erp_avg
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ML, MR, MT, MB = self.ML, self.MR, self.MT, self.MB

        painter.fillRect(0, 0, w, h, QColor(C.BG_PANEL))

        if self._marker is None:
            painter.setPen(QPen(QColor(C.TEXT_LO)))
            painter.setFont(QFont("Share Tech Mono", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Click a stimulus event\nto inspect its epoch")
            painter.end()
            return

        m = self._marker
        epoch = m.epoch   # (200, 8) or None

        plot_w = w - ML - MR
        plot_h = h - MT - MB
        ch_h   = plot_h / 8

        # P300 zone highlight
        x_p300_on  = ML + (P300_ONSET_MS  / 800.0) * plot_w
        x_p300_off = ML + (P300_OFFSET_MS / 800.0) * plot_w
        p300_fill  = QColor(0, 229, 255, 15)
        painter.fillRect(int(x_p300_on), MT, int(x_p300_off - x_p300_on), plot_h,
                         p300_fill)
        pen_p300 = QPen(QColor(0, 229, 255, 50), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen_p300)
        painter.drawLine(int(x_p300_on),  MT, int(x_p300_on),  MT + plot_h)
        painter.drawLine(int(x_p300_off), MT, int(x_p300_off), MT + plot_h)

        # Border
        painter.setPen(QPen(QColor(C.BORDER)))
        painter.drawRect(ML, MT, plot_w, plot_h)

        for ch in range(8):
            y_top    = MT + ch * ch_h
            y_center = y_top + ch_h / 2.0

            # Lane separator
            if ch > 0:
                painter.setPen(QPen(QColor(C.BORDER)))
                painter.drawLine(ML, int(y_top), ML + plot_w, int(y_top))

            is_p300 = ch in P300_CH
            ch_color = QColor(CH_COLORS[ch])
            if is_p300:
                ch_color.setAlpha(255)
            else:
                ch_color.setAlpha(160)

            # Channel label
            painter.setPen(QPen(ch_color))
            painter.setFont(QFont("Share Tech Mono", 7))
            label = CH_NAMES[ch] + ("*" if is_p300 else "")
            painter.drawText(2, int(y_center) + 4, label)

            # Zero line
            painter.setPen(QPen(QColor(26, 46, 66, 60)))
            painter.drawLine(ML, int(y_center), ML + plot_w, int(y_center))

            # ── Draw epoch signal ───────────────────────────────────────────
            if epoch is not None:
                signal = epoch[:, ch]
                # Remove DC offset (hardware outputs ~200,000 µV baseline)
                signal = signal - signal.mean()
                scale  = (ch_h * 0.42) / max(float(np.std(signal)) * 3.0, 5.0)

                path = QPainterPath()
                for i in range(EPOCH_SAMPLES):
                    x = ML + (i / (EPOCH_SAMPLES - 1)) * plot_w
                    y = y_center - signal[i] * scale
                    y = max(y_top + 1.0, min(y_top + ch_h - 1.0, y))
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)

                pen = QPen(ch_color, 1.5 if is_p300 else 1.0)
                painter.setPen(pen)
                painter.drawPath(path)

            # ── Grand average ERP overlay (dashed) ─────────────────────────
            if self._erp_avg is not None:
                avg_sig = self._erp_avg[:, ch]
                avg_sig = avg_sig - avg_sig.mean()
                avg_scale = (ch_h * 0.42) / max(float(np.std(avg_sig)) * 3.0, 1.0)

                avg_path = QPainterPath()
                for i in range(EPOCH_SAMPLES):
                    x = ML + (i / (EPOCH_SAMPLES - 1)) * plot_w
                    y = y_center - avg_sig[i] * avg_scale
                    y = max(y_top + 1.0, min(y_top + ch_h - 1.0, y))
                    if i == 0:
                        avg_path.moveTo(x, y)
                    else:
                        avg_path.lineTo(x, y)

                avg_color = QColor(ch_color)
                avg_color.setAlpha(80)
                avg_pen = QPen(avg_color, 1.0, Qt.PenStyle.DashLine)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)

        # ── Header label ───────────────────────────────────────────────────
        color_str = C.TARGET if m.is_target else C.NONTARGET
        epoch_status = "✓ EPOCH ACCEPTED" if m.epoch is not None else "✗ NO EPOCH"
        painter.setPen(QPen(QColor(color_str)))
        painter.setFont(QFont("Share Tech Mono", 9))
        painter.drawText(ML + 4, MT - 2,
                         f"#{m.index:03d}  {m.symbol} {m.label}"
                         f"  {'TARGET' if m.is_target else 'NON-TARGET'}"
                         f"  |  t={m.timestamp:.3f}s  |  {epoch_status}")

        # Time axis
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        for t_ms in [0, 100, 200, 250, 300, 400, 500, 600, 700, 800]:
            x = ML + (t_ms / 800.0) * plot_w
            painter.drawLine(int(x), MT + plot_h, int(x), MT + plot_h + 3)
            painter.drawText(int(x) - 12, MT + plot_h + 13, f"{t_ms}")
        painter.drawText(ML + plot_w // 2 - 20, h - 1, "Time (ms)")

        painter.end()


# ── ERP Panel ─────────────────────────────────────────────────────────────────

class ERPPanel(QWidget):
    """Static ERP display fed from auth_result.json erp_data."""

    ML, MR, MT, MB = 44, 16, 20, 32

    def __init__(self, parent=None):
        super().__init__(parent)
        self._erp: Optional[dict] = None
        self._y_min = -8.0
        self._y_max = 12.0
        self.setMinimumSize(340, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:{C.BG_CARD};")

    def set_erp(self, erp_data: Optional[dict]) -> None:
        self._erp = erp_data
        if erp_data:
            all_vals = (erp_data.get("target", []) +
                        erp_data.get("nontarget", []))
            if all_vals:
                lo = min(all_vals) - 1
                hi = max(all_vals) + 1
                self._y_min = min(lo, -3.0)
                self._y_max = max(hi,  5.0)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ML, MR, MT, MB = self.ML, self.MR, self.MT, self.MB

        painter.fillRect(0, 0, w, h, QColor(C.BG_CARD))

        if not self._erp:
            painter.setPen(QPen(QColor(C.TEXT_LO)))
            painter.setFont(QFont("Share Tech Mono", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No ERP data")
            painter.end()
            return

        plot_w = w - ML - MR
        plot_h = h - MT - MB
        t_ms  = self._erp.get("t_ms", [])
        target = self._erp.get("target", [])
        nont   = self._erp.get("nontarget", [])
        sem    = self._erp.get("target_sem", [])
        p_on   = self._erp.get("p300_onset_ms", 250.0)
        p_off  = self._erp.get("p300_offset_ms", 500.0)

        if not t_ms:
            painter.end()
            return

        def to_px(t, amp):
            x = ML + (t - t_ms[0]) / (t_ms[-1] - t_ms[0]) * plot_w
            y = MT + (1 - (amp - self._y_min) / (self._y_max - self._y_min)) * plot_h
            return x, y

        # P300 zone
        x1, _ = to_px(p_on,  0)
        x2, _ = to_px(p_off, 0)
        painter.fillRect(int(x1), MT, int(x2 - x1), plot_h, QColor(0, 229, 255, 15))
        dpen = QPen(QColor(0, 229, 255, 60), 1, Qt.PenStyle.DashLine)
        painter.setPen(dpen)
        painter.drawLine(int(x1), MT, int(x1), MT + plot_h)
        painter.drawLine(int(x2), MT, int(x2), MT + plot_h)

        # Grid
        painter.setPen(QPen(QColor(C.BORDER)))
        for amp in np.arange(np.ceil(self._y_min), np.floor(self._y_max) + 1, 2):
            _, y = to_px(0, amp)
            painter.drawLine(ML, int(y), ML + plot_w, int(y))

        # Zero line
        painter.setPen(QPen(QColor("#2a4060"), 1))
        _, y0 = to_px(0, 0.0)
        painter.drawLine(ML, int(y0), ML + plot_w, int(y0))

        # Axes
        painter.setPen(QPen(QColor(C.BORDER)))
        painter.drawLine(ML, MT, ML, MT + plot_h)
        painter.drawLine(ML, MT + plot_h, ML + plot_w, MT + plot_h)

        # Y labels
        painter.setFont(QFont("Share Tech Mono", 7))
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        for amp in np.arange(np.ceil(self._y_min), np.floor(self._y_max) + 1, 2):
            _, y = to_px(0, amp)
            painter.drawText(2, int(y) + 4, f"{amp:.0f}")

        # SEM ribbon
        if sem and len(sem) == len(t_ms) and target:
            from PyQt6.QtCore import QPointF
            from PyQt6.QtGui import QPolygonF
            poly = []
            for t, a, s in zip(t_ms, target, sem):
                x, y = to_px(t, a + s)
                poly.append(QPointF(x, y))
            for t, a, s in zip(reversed(t_ms), reversed(target), reversed(sem)):
                x, y = to_px(t, a - s)
                poly.append(QPointF(x, y))
            painter.setBrush(QBrush(QColor(0, 230, 118, 35)))
            painter.setPen(Qt.PenStyle.NoPen)
            from PyQt6.QtGui import QPolygonF
            painter.drawPolygon(QPolygonF(poly))

        # Non-target waveform
        if nont:
            path = QPainterPath()
            for i, (t, a) in enumerate(zip(t_ms, nont)):
                x, y = to_px(t, a)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor("#ff5252"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        # Target waveform
        if target:
            path = QPainterPath()
            for i, (t, a) in enumerate(zip(t_ms, target)):
                x, y = to_px(t, a)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(C.TARGET), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        # Legend
        lx, ly = ML + 8, MT + 12
        painter.setPen(QPen(QColor(C.TARGET), 2))
        painter.drawLine(lx, ly, lx + 18, ly)
        painter.setFont(QFont("Share Tech Mono", 8))
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.drawText(lx + 22, ly + 4, "Target (P300 chs avg)")
        painter.setPen(QPen(QColor("#ff5252"), 2))
        painter.drawLine(lx + 150, ly, lx + 168, ly)
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.drawText(lx + 172, ly + 4, "Non-target")

        # X labels
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.setFont(QFont("Share Tech Mono", 7))
        for t_tick in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
            if t_tick <= t_ms[-1]:
                x, _ = to_px(t_tick, 0)
                painter.drawText(int(x) - 12, MT + plot_h + 16, str(t_tick))
        painter.drawText(ML + plot_w // 2 - 20, h - 2, "Time (ms)")

        # Y unit
        painter.save()
        painter.translate(12, MT + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-12, 0, "µV")
        painter.restore()

        painter.end()


# ── ISI Histogram ─────────────────────────────────────────────────────────────

class ISIWidget(QWidget):
    """Mini histogram of inter-stimulus intervals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._isis: list[float] = []
        self.setMinimumSize(200, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background:{C.BG_CARD};")

    def set_isis(self, isis_ms: list[float]) -> None:
        self._isis = isis_ms
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(C.BG_CARD))

        if not self._isis:
            painter.end()
            return

        ML, MR, MT, MB = 10, 10, 16, 22
        plot_w = w - ML - MR
        plot_h = h - MT - MB

        # Histogram bins: 400–800 ms range
        bins    = np.arange(400, 810, 25)
        counts, edges = np.histogram(self._isis, bins=bins)
        max_c   = max(counts) or 1
        n_bins  = len(counts)
        bar_w   = plot_w / n_bins
        EXPECTED_MS = 575.0

        for i, c in enumerate(counts):
            x       = ML + i * bar_w
            bar_h   = (c / max_c) * plot_h
            # Color: green if within ±25ms of expected, warn if >75ms off
            center_ms = (edges[i] + edges[i + 1]) / 2
            dist    = abs(center_ms - EXPECTED_MS)
            col     = QColor(C.SUCCESS if dist < 25 else C.WARN if dist < 75 else C.DANGER)
            col.setAlpha(200)
            painter.fillRect(int(x) + 1, MT + plot_h - int(bar_h),
                             max(1, int(bar_w) - 2), int(bar_h), col)

        # Expected line
        ex = ML + (EXPECTED_MS - 400) / (800 - 400) * plot_w
        painter.setPen(QPen(QColor(C.ACCENT), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(ex), MT, int(ex), MT + plot_h)

        # Stats text
        arr = np.array(self._isis)
        painter.setFont(QFont("Share Tech Mono", 7))
        painter.setPen(QPen(QColor(C.TEXT_LO)))
        painter.drawText(ML + 2, MT - 2,
                         f"ISI  μ={arr.mean():.0f}ms  σ={arr.std():.0f}ms  "
                         f"min={arr.min():.0f}  max={arr.max():.0f}")

        # X axis labels
        for t_ms in [400, 500, 575, 700, 800]:
            x = ML + (t_ms - 400) / 400 * plot_w
            painter.drawLine(int(x), MT + plot_h, int(x), MT + plot_h + 3)
            painter.drawText(int(x) - 12, MT + plot_h + 13, str(t_ms))

        painter.end()


# ── Marker Table ──────────────────────────────────────────────────────────────

class MarkerTable(QTableWidget):
    COLS = ["#", "Symbol", "Label", "Target", "Time (s)", "ISI (ms)", "Buffer", "Epoch"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setStyleSheet(f"""
            QTableWidget {{
                background:{C.BG_PANEL}; color:{C.TEXT_HI};
                font-family:'Share Tech Mono','Courier New',monospace;
                font-size:10px; gridline-color:{C.BORDER};
                border:none;
            }}
            QHeaderView::section {{
                background:{C.BG_CARD}; color:{C.ACCENT};
                font-family:'Share Tech Mono'; font-size:9px;
                border:none; padding:3px;
                border-bottom:1px solid {C.BORDER};
            }}
            QTableWidget::item:selected {{
                background:{C.ACCENT}33;
            }}
        """)
        self._callbacks: list = []

    def on_select(self, cb) -> None:
        self._callbacks.append(cb)
        self.itemSelectionChanged.connect(self._on_row_changed)

    def _on_row_changed(self) -> None:
        rows = self.selectedItems()
        if rows:
            idx = int(self.item(rows[0].row(), 0).text())
            for cb in self._callbacks:
                cb(idx)

    def load_session(self, session: SessionData) -> None:
        self.setRowCount(0)
        for m in session.markers:
            row = self.rowCount()
            self.insertRow(row)
            items = [
                str(m.index),
                m.symbol,
                m.label,
                "TARGET" if m.is_target else "—",
                f"{m.timestamp:.3f}",
                f"{m.isi_s * 1000:.1f}" if m.isi_s > 0 else "—",
                str(m.buffer_index),
                "✓" if m.epoch is not None else "✗",
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if m.is_target:
                    item.setForeground(QColor(C.TARGET))
                # ISI warning colour
                if col == 5 and m.isi_s > 0:
                    isi_ms = m.isi_s * 1000
                    if abs(isi_ms - 575) > 75:
                        item.setForeground(QColor(C.DANGER))
                self.setItem(row, col, item)

    def select_index(self, idx: int) -> None:
        for row in range(self.rowCount()):
            if self.item(row, 0) and int(self.item(row, 0).text()) == idx:
                self.setCurrentCell(row, 0)
                self.scrollToItem(self.item(row, 0))
                return


# ── Summary panel ─────────────────────────────────────────────────────────────

class SummaryPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{C.BG_CARD}; border:1px solid {C.BORDER};"
                           f"border-radius:6px;")
        self.setFixedHeight(56)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(24)

        def stat(label, default="—"):
            col = QVBoxLayout()
            col.setSpacing(1)
            lbl = mono_label(label, 8, C.TEXT_LO)
            val = mono_label(default, 13, C.TEXT_HI)
            col.addWidget(lbl)
            col.addWidget(val)
            layout.addLayout(col)
            return val

        self._result_badge = stat("RESULT")
        self._n_total      = stat("TOTAL EVENTS")
        self._n_target     = stat("TARGET")
        self._n_nontarget  = stat("NON-TARGET")
        self._duration     = stat("DURATION")
        self._p300         = stat("ΔP300 MEAN")
        self._pre_sigma    = stat("PRE σ")
        self._snr          = stat("SNR")
        layout.addStretch()

    def update_session(self, session: SessionData) -> None:
        ar = session.auth_result
        granted = ar.get("granted", False)
        self._result_badge.setText("GRANTED ✓" if granted else "DENIED ✗")
        self._result_badge.setStyleSheet(
            f"color:{'#00e676' if granted else '#ff1744'};"
            f"font-family:'Share Tech Mono'; font-size:13px;"
        )
        self._n_total.setText(str(len(session.markers)))
        self._n_target.setText(str(len(session.targets)))
        self._n_nontarget.setText(str(len(session.nontargets)))
        self._duration.setText(f"{session.duration_s:.1f}s")
        msg = ar.get("message", "")
        # Parse µV and SNR from message if present
        tgt = ar.get("target_peak_uv", 0.0)
        nt  = ar.get("nontarget_peak_uv", 0.0)
        snr = ar.get("snr_db", 0.0)
        self._p300.setText(f"{abs(tgt - nt):.2f} µV")
        # Extract pre_σ from message string if new format
        import re
        m = re.search(r"pre_σ=([\d.]+)", msg)
        self._pre_sigma.setText(f"{m.group(1)} µV" if m else "—")
        self._snr.setText(f"{snr:.1f} dB")


# ── Main Window ───────────────────────────────────────────────────────────────

class DebugViewer(QMainWindow):

    def __init__(self, initial_session: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("🧠  Soude — Session Debug Viewer")
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(f"background:{C.BG_DEEP};")

        self._session: Optional[SessionData] = None
        self._selected_idx: int = -1

        # ── Build UI ──────────────────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"background:{C.BG_PANEL}; border-bottom:1px solid {C.BORDER};"
        )
        hdr_layout = QHBoxLayout(header)
        hdr_layout.setContentsMargins(14, 4, 14, 4)
        hdr_layout.setSpacing(12)
        hdr_layout.addWidget(mono_label("🧠  SOUDE  DEBUG  VIEWER", 13, C.ACCENT))
        hdr_layout.addSpacing(16)
        hdr_layout.addWidget(mono_label("Session:", 10, C.TEXT_LO))

        self._session_combo = QComboBox()
        self._session_combo.setFixedWidth(240)
        self._session_combo.setStyleSheet(f"""
            QComboBox {{
                background:{C.BG_CARD}; border:1px solid {C.BORDER};
                border-radius:4px; color:{C.TEXT_HI};
                font-family:'Share Tech Mono'; font-size:11px; padding:2px 8px;
            }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{
                background:{C.BG_CARD}; color:{C.TEXT_HI};
                selection-background-color:{C.ACCENT}33;
            }}
        """)
        hdr_layout.addWidget(self._session_combo)

        load_btn = QPushButton("LOAD")
        load_btn.setFixedSize(70, 30)
        load_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; border:1px solid {C.ACCENT};
                border-radius:4px; color:{C.ACCENT};
                font-family:'Share Tech Mono'; font-size:11px; letter-spacing:2px;
            }}
            QPushButton:hover {{ background:{C.ACCENT}22; }}
            QPushButton:pressed {{ background:{C.ACCENT}44; }}
        """)
        load_btn.clicked.connect(self._load_selected)
        hdr_layout.addWidget(load_btn)

        hdr_layout.addStretch()
        hdr_layout.addWidget(mono_label("* = P300 channels  |  Green = target  |  Red = non-target  |  Cyan zone = 250–500 ms",
                                         9, C.TEXT_LO))
        root_layout.addWidget(header)

        # ── Summary panel ────────────────────────────────────────────────────
        self._summary = SummaryPanel()
        wrap = QWidget()
        wrap.setStyleSheet(f"background:{C.BG_DEEP};")
        wrap_l = QHBoxLayout(wrap)
        wrap_l.setContentsMargins(10, 6, 10, 4)
        wrap_l.addWidget(self._summary)
        root_layout.addWidget(wrap)

        # ── Timeline (scrollable) ────────────────────────────────────────────
        tl_section = QWidget()
        tl_section.setStyleSheet(f"background:{C.BG_DEEP};")
        tl_layout = QVBoxLayout(tl_section)
        tl_layout.setContentsMargins(10, 2, 10, 2)
        tl_layout.setSpacing(2)
        tl_layout.addWidget(section_header("STIMULUS  TIMELINE  (click event to inspect)"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(TimelineWidget.TOTAL_H + 4)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{C.BG_PANEL}; border:none;")

        self._timeline = TimelineWidget()
        self._timeline.on_select(self._on_event_selected)
        scroll.setWidget(self._timeline)
        tl_layout.addWidget(scroll)
        root_layout.addWidget(tl_section)

        # ── Main body: splitter ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background:#1a2e42; width:2px; }")

        # ── Left: marker table ────────────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{C.BG_DEEP};")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(10, 4, 4, 10)
        left_l.setSpacing(2)
        left_l.addWidget(section_header("MARKER  LOG"))
        self._table = MarkerTable()
        self._table.on_select(self._on_event_selected)
        left_l.addWidget(self._table, stretch=1)
        splitter.addWidget(left)

        # ── Centre: epoch inspector ───────────────────────────────────────────
        centre = QWidget()
        centre.setStyleSheet(f"background:{C.BG_DEEP};")
        centre_l = QVBoxLayout(centre)
        centre_l.setContentsMargins(4, 4, 4, 10)
        centre_l.setSpacing(2)
        centre_l.addWidget(section_header(
            "EPOCH  INSPECTOR   (solid=selected epoch  |  dashed=grand average overlay)"))
        self._epoch_widget = EpochWidget()
        centre_l.addWidget(self._epoch_widget, stretch=1)
        splitter.addWidget(centre)

        # ── Right: ERP + ISI ─────────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{C.BG_DEEP};")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(4, 4, 10, 10)
        right_l.setSpacing(2)
        right_l.addWidget(section_header("ERP  GRAND  AVERAGES  (Cz·Pz·Oz  mean)"))
        self._erp_panel = ERPPanel()
        right_l.addWidget(self._erp_panel, stretch=3)
        right_l.addSpacing(4)
        right_l.addWidget(section_header("ISI  DISTRIBUTION   (expected 575 ms)"))
        self._isi_widget = ISIWidget()
        right_l.addWidget(self._isi_widget, stretch=1)
        splitter.addWidget(right)

        splitter.setSizes([260, 540, 360])
        body = QWidget()
        body.setStyleSheet(f"background:{C.BG_DEEP};")
        body_l = QVBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.addWidget(splitter, stretch=1)
        root_layout.addWidget(body, stretch=1)

        # ── Populate session list ─────────────────────────────────────────────
        self._populate_sessions()

        # Load initial session if specified
        if initial_session and initial_session.exists():
            self._load_path(initial_session)
        elif self._session_combo.count() > 0:
            self._load_selected()

    # ── Session management ────────────────────────────────────────────────────

    def _populate_sessions(self) -> None:
        self._session_combo.clear()
        if not LOGS_DIR.exists():
            return
        sessions = sorted(
            [d for d in LOGS_DIR.iterdir() if d.is_dir() and (d / "markers.csv").exists()],
            reverse=True,
        )
        for s in sessions:
            # Quick peek at result
            ar_path = s / "auth_result.json"
            badge = ""
            if ar_path.exists():
                try:
                    with open(ar_path) as f:
                        ar = json.load(f)
                    badge = " ✓" if ar.get("granted") else " ✗"
                except Exception:
                    pass
            self._session_combo.addItem(s.name + badge, userData=s)

    def _load_selected(self) -> None:
        path = self._session_combo.currentData()
        if path:
            self._load_path(path)

    def _load_path(self, path: Path) -> None:
        try:
            session = load_session(path)
            self._session = session
            self._summary.update_session(session)
            self._timeline.set_session(session)
            self._table.load_session(session)
            self._erp_panel.set_erp(session.erp_data)
            isis_ms = [m.isi_s * 1000 for m in session.markers if m.isi_s > 0]
            self._isi_widget.set_isis(isis_ms)
            # Auto-select first target event (fall back to first event for
            # older sessions where is_target was not logged correctly)
            targets = session.targets
            first = targets[0] if targets else (session.markers[0] if session.markers else None)
            if first:
                self._on_event_selected(first.index)
            logger.info("Loaded session: %s (%d markers)", path.name, len(session.markers))
        except Exception as exc:
            logger.exception("Failed to load session %s: %s", path, exc)

    # ── Event selection ───────────────────────────────────────────────────────

    def _on_event_selected(self, idx: int) -> None:
        if self._session is None or idx == self._selected_idx:
            return
        self._selected_idx = idx
        marker = next((m for m in self._session.markers if m.index == idx), None)
        if marker is None:
            return

        # Build grand-average ERP array for the same class (for overlay)
        erp_avg = self._build_class_average(marker.is_target)
        self._epoch_widget.set_marker(marker, erp_avg)

        # Sync table and timeline without triggering recursive callbacks
        self._table.blockSignals(True)
        self._table.select_index(idx)
        self._table.blockSignals(False)
        self._timeline._selected = idx
        self._timeline.update()

    def _build_class_average(self, is_target: bool) -> Optional[np.ndarray]:
        """Build a grand-average epoch (200, 8) from all same-class epochs."""
        if not self._session:
            return None
        epochs = [
            m.epoch for m in self._session.markers
            if m.is_target == is_target and m.epoch is not None
        ]
        if not epochs:
            return None
        stacked = np.stack(epochs, axis=0)   # (N, 200, 8)
        return stacked.mean(axis=0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    initial: Optional[Path] = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            initial = p
        else:
            initial = LOGS_DIR / sys.argv[1]

    app = QApplication(sys.argv)
    app.setApplicationName("Soude Debug Viewer")
    win = DebugViewer(initial_session=initial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() LOGS_DIR / sys.argv[1]

    app = QApplication(sys.argv)
    app.setApplicationName("Soude Debug Viewer")
    win = DebugViewer(initial_session=initial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
 LOGS_DIR / sys.argv[1]

    app = QApplication(sys.argv)
    app.setApplicationName("Soude Debug Viewer")
    win = DebugViewer(initial_session=initial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
