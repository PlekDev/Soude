"""
erp_viewer.py — Neuro-Lock Live ERP Viewer
Standalone PyQt6 widget that plots target vs non-target grand averages in
real time.  Sub-team 2 uses this during signal validation; Sub-team 4 uses
it in the demo.  Can be embedded or run standalone.
"""

import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy,
)

# ── Colors ─────────────────────────────────────────────────────────────────────
C_BG        = QColor("#060a10")
C_GRID      = QColor("#1a2e42")
C_TARGET    = QColor("#00e676")   # green
C_NONTARGET = QColor("#ff5252")   # red
C_SHADE     = QColor(0, 230, 118, 30)
C_P300_ZONE = QColor(0, 229, 255, 18)
C_TEXT      = QColor("#4a6478")
C_AXIS      = QColor("#1a2e42")


class ERPCanvas(QWidget):
    """
    Custom painting widget: draws target/non-target ERP waveforms with
    a shaded P300 detection window and SEM error ribbon.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent;")

        self._t_ms:         Optional[list] = None
        self._target:       Optional[list] = None
        self._nontarget:    Optional[list] = None
        self._target_sem:   Optional[list] = None
        self._p300_onset:   float = 250.0
        self._p300_offset:  float = 500.0

        # Y-axis range (µV) — auto-scaled after first data
        self._y_min = -5.0
        self._y_max = 15.0

    def update_data(self, erp_data: dict) -> None:
        self._t_ms        = erp_data.get("t_ms")
        self._target      = erp_data.get("target")
        self._nontarget   = erp_data.get("nontarget")
        self._target_sem  = erp_data.get("target_sem")
        self._p300_onset  = erp_data.get("p300_onset_ms", 250.0)
        self._p300_offset = erp_data.get("p300_offset_ms", 500.0)

        # Auto-scale Y
        all_vals = (
            (self._target or []) +
            (self._nontarget or []) +
            (self._target_sem or [])
        )
        if all_vals:
            lo = min(all_vals) - 2
            hi = max(all_vals) + 2
            self._y_min = min(lo, -2.0)
            self._y_max = max(hi,  5.0)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Margins
        ml, mr, mt, mb = 52, 20, 20, 36
        plot_w = w - ml - mr
        plot_h = h - mt - mb

        def to_px(t_ms: float, amp: float):
            t_range = max((self._t_ms[-1] - self._t_ms[0]) if self._t_ms else 800.0, 1e-9)
            x = ml + (t_ms - (self._t_ms[0] if self._t_ms else 0)) / t_range * plot_w
            y_range = max(self._y_max - self._y_min, 1e-9)
            y = mt + (1 - (amp - self._y_min) / y_range) * plot_h
            return x, y

        # Background
        painter.fillRect(0, 0, w, h, C_BG)

        # P300 detection zone
        if self._t_ms:
            x1, _ = to_px(self._p300_onset, 0)
            x2, _ = to_px(self._p300_offset, 0)
            painter.fillRect(int(x1), mt, int(x2 - x1), plot_h, C_P300_ZONE)
            pen = QPen(QColor(0, 229, 255, 60))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(int(x1), mt, int(x1), mt + plot_h)
            painter.drawLine(int(x2), mt, int(x2), mt + plot_h)

        # Grid lines
        grid_pen = QPen(C_GRID)
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for amp in np.arange(np.ceil(self._y_min), np.floor(self._y_max) + 1, 2):
            _, y = to_px(0, amp)
            painter.drawLine(ml, int(y), ml + plot_w, int(y))

        # Zero line
        zero_pen = QPen(QColor("#2a4060"))
        zero_pen.setWidth(1)
        painter.setPen(zero_pen)
        _, y0 = to_px(0, 0.0)
        painter.drawLine(ml, int(y0), ml + plot_w, int(y0))

        # Axes
        ax_pen = QPen(C_AXIS)
        ax_pen.setWidth(1)
        painter.setPen(ax_pen)
        painter.drawLine(ml, mt, ml, mt + plot_h)
        painter.drawLine(ml, mt + plot_h, ml + plot_w, mt + plot_h)

        # Axis labels
        font = QFont("Share Tech Mono", 9)
        painter.setFont(font)
        painter.setPen(QPen(C_TEXT))
        for amp in np.arange(np.ceil(self._y_min), np.floor(self._y_max) + 1, 2):
            _, y = to_px(0, amp)
            painter.drawText(2, int(y) + 4, f"{amp:.0f}")

        if self._t_ms:
            for t_tick in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
                if t_tick <= self._t_ms[-1]:
                    x, _ = to_px(t_tick, 0)
                    painter.drawText(int(x) - 14, mt + plot_h + 16, f"{t_tick}")

        # SEM ribbon for target
        if self._t_ms and self._target and self._target_sem:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            ribbon_color = QColor(0, 230, 118, 40)
            poly = []
            for i, (t, a, s) in enumerate(zip(self._t_ms, self._target, self._target_sem)):
                x, y = to_px(t, a + s)
                poly.append(QPointF(x, y))
            for i in range(len(self._t_ms) - 1, -1, -1):
                t, a, s = self._t_ms[i], self._target[i], self._target_sem[i]
                x, y = to_px(t, a - s)
                poly.append(QPointF(x, y))
            painter.setBrush(ribbon_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF(poly))

        # Non-target waveform
        if self._t_ms and self._nontarget:
            nt_pen = QPen(C_NONTARGET)
            nt_pen.setWidth(2)
            painter.setPen(nt_pen)
            for i in range(1, len(self._t_ms)):
                x1, y1 = to_px(self._t_ms[i - 1], self._nontarget[i - 1])
                x2, y2 = to_px(self._t_ms[i], self._nontarget[i])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Target waveform
        if self._t_ms and self._target:
            t_pen = QPen(C_TARGET)
            t_pen.setWidth(2)
            painter.setPen(t_pen)
            for i in range(1, len(self._t_ms)):
                x1, y1 = to_px(self._t_ms[i - 1], self._target[i - 1])
                x2, y2 = to_px(self._t_ms[i], self._target[i])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Legend
        lx, ly = ml + 10, mt + 14
        painter.setPen(QPen(C_TARGET))
        painter.drawLine(lx, ly, lx + 20, ly)
        painter.setPen(QPen(C_TEXT))
        painter.drawText(lx + 24, ly + 4, "Target")
        painter.setPen(QPen(C_NONTARGET))
        painter.drawLine(lx + 90, ly, lx + 110, ly)
        painter.setPen(QPen(C_TEXT))
        painter.drawText(lx + 114, ly + 4, "Non-target")

        # Axis units
        painter.setPen(QPen(C_TEXT))
        painter.save()
        painter.translate(12, mt + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-20, 0, "µV")
        painter.restore()
        painter.drawText(ml + plot_w // 2 - 16, h - 4, "Time (ms)")

        painter.end()


class ERPViewer(QWidget):
    """
    Full panel: canvas + refresh controls.
    Can be instantiated standalone or embedded in the main app.
    """

    def __init__(self, pipeline=None, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self.setStyleSheet("background: #060a10;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QHBoxLayout()
        title = QLabel("ERP MONITOR — Pz / Cz / Oz Average")
        title.setStyleSheet(
            "color: #00e5ff; font-size: 12px; letter-spacing: 3px; "
            "font-family: 'Share Tech Mono', monospace;"
        )
        header.addWidget(title)
        header.addStretch()

        self._n_label = QLabel("T: 0 | NT: 0")
        self._n_label.setStyleSheet(
            "color: #4a6478; font-size: 11px; font-family: 'Share Tech Mono', monospace;"
        )
        header.addWidget(self._n_label)
        layout.addLayout(header)

        self._canvas = ERPCanvas(self)
        layout.addWidget(self._canvas, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def set_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    def start_live(self, interval_ms: int = 500) -> None:
        self._timer.start(interval_ms)

    def stop_live(self) -> None:
        self._timer.stop()

    @pyqtSlot()
    def _refresh(self) -> None:
        if self._pipeline is None:
            return
        try:
            erp = self._pipeline.get_erp_data()
            self._canvas.update_data(erp)
            avg = self._pipeline._averager
            self._n_label.setText(f"T: {avg.n_target} | NT: {avg.n_nontarget}")
        except Exception as exc:
            pass  # Don't crash the UI on display errors


# ── Standalone demo ────────────────────────────────────────────────────────────

def _demo():
    """
    Run standalone ERP viewer with synthetic data to validate the widget.
    """
    import math

    app = QApplication(sys.argv)
    viewer = ERPViewer()
    viewer.resize(700, 340)
    viewer.setWindowTitle("Neuro-Lock — ERP Monitor (Demo)")

    n = 200
    t = [i * 4.0 for i in range(n)]

    def p300_shape(t_ms, peak_t=320, amp=8.0, width=60):
        return amp * math.exp(-0.5 * ((t_ms - peak_t) / width) ** 2)

    target    = [p300_shape(ti) + 0.3 * math.sin(ti * 0.05) for ti in t]
    nontarget = [0.5 * math.sin(ti * 0.04) - 0.2 for ti in t]
    sem       = [0.4 for _ in t]

    viewer._canvas.update_data({
        "t_ms": t,
        "target": target,
        "nontarget": nontarget,
        "target_sem": sem,
        "p300_onset_ms": 250,
        "p300_offset_ms": 500,
    })
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()
