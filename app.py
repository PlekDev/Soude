"""
app.py — Neuro-Lock Main Application (PyQt6)
High-speed stimulus display, enrollment wizard, and vault screen.
Sub-team 3 (UX/UI) owns this file.
"""


from dotenv import load_dotenv
load_dotenv()
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

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
    QGridLayout, QProgressBar, QMessageBox, QDialog,
)

from brain_engine import BrainEngine, MockUnicorn, SAMPLE_RATE
from Fase1.signal_processing import AuthenticationPipeline, AuthResult, EPOCH_DURATION_S
from Fase1.stimulus_runner import StimulusRunner, ParadigmConfig
from data_logger import SessionLogger

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
    # ── PASSWORD IMAGES (IDs 0, 1, 2) — make these unmistakable ──────────────
    # (bg_color,    symbol, label   )
    ("#AA0000",     "★",    "STAR"    ),  # 0  ← PASSWORD  — red background
    ("#005500",     "♦",    "DIAMOND" ),  # 1  ← PASSWORD  — green background
    ("#0033AA",     "●",    "CIRCLE"  ),  # 2  ← PASSWORD  — blue background

    # ── DISTRACTOR IMAGES (IDs 3–19) — all identical dark grey ───────────────
    ("#1C1C1C",     "",     "04"  ),  # 3
    ("#1C1C1C",     "",     "05"  ),  # 4
    ("#1C1C1C",     "",     "06"  ),  # 5
    ("#1C1C1C",     "",     "07"  ),  # 6
    ("#1C1C1C",     "",     "08"  ),  # 7
    ("#1C1C1C",     "",     "09"  ),  # 8
    ("#1C1C1C",     "",     "10"  ),  # 9
    ("#1C1C1C",     "",     "11"  ),  # 10
    ("#1C1C1C",     "",     "12"  ),  # 11
    ("#1C1C1C",     "",     "13"  ),  # 12
    ("#1C1C1C",     "",     "14"  ),  # 13
    ("#1C1C1C",     "",     "15"  ),  # 14
    ("#1C1C1C",     "",     "16"  ),  # 15
    ("#1C1C1C",     "",     "17"  ),  # 16
    ("#1C1C1C",     "",     "18"  ),  # 17
    ("#1C1C1C",     "",     "19"  ),  # 18
    ("#1C1C1C",     "",     "20"  ),  # 19
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

            # ── Generated placeholder from STIMULUS_CATALOG ───────────────────
            bg_hex, symbol, label = STIMULUS_CATALOG[img_id]
            px = QPixmap(target_size, target_size)
            px.fill(QColor(bg_hex))
            painter = QPainter(px)

            if symbol:
                # PASSWORD image: large symbol + word label centred
                font_sym = QFont("Segoe UI Symbol", 150, QFont.Weight.Bold)
                painter.setFont(font_sym)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    0, 20, target_size, 300,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    symbol,
                )
                font_lbl = QFont("Arial", 54, QFont.Weight.Bold)
                painter.setFont(font_lbl)
                painter.setPen(QColor(255, 255, 255, 210))
                painter.drawText(
                    0, 320, target_size, 120,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
            else:
                # DISTRACTOR image: tiny ID number in bottom-right corner only
                font_id = QFont("Arial", 18)
                painter.setFont(font_id)
                painter.setPen(QColor(70, 70, 70))
                painter.drawText(
                    target_size - 44, target_size - 10,
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

    @pyqtSlot()
    def show_blank(self):
        self._img_label.clear()
        self._status.setText("")

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

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(24)
        layout.setContentsMargins(60, 60, 60, 60)

        self._icon = QLabel("", self)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size: 96px;")
        layout.addWidget(self._icon)

        self._title = QLabel("", self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            f"font-size: 36px; font-weight: 700; color: {Colors.TEXT_HI}; "
            f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
        )
        layout.addWidget(self._title)

        self._detail = QLabel("", self)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(
            f"font-size: 15px; color: {Colors.TEXT_LO}; "
            f"font-family: 'Share Tech Mono', monospace;"
        )
        layout.addWidget(self._detail)

        layout.addWidget(NeuralDivider())

        btn_row = QHBoxLayout()
        self._btn_retry      = GlowButton("⟳  RETRY SCAN", Colors.ACCENT)
        self._btn_back       = GlowButton("←  BACK", Colors.TEXT_LO)
        self._btn_open_vault = GlowButton("▶  OPEN VAULT", Colors.SUCCESS)
        self._btn_retry.clicked.connect(self.sig_retry)
        self._btn_back.clicked.connect(self.sig_back)
        self._btn_open_vault.clicked.connect(self.sig_open_vault)
        btn_row.addWidget(self._btn_back)
        btn_row.addWidget(self._btn_retry)
        btn_row.addWidget(self._btn_open_vault)
        layout.addLayout(btn_row)

    def show_result(self, result: AuthResult):
        if result.granted:
            self._icon.setText("🔓")
            self._title.setText("ACCESS GRANTED")
            self._title.setStyleSheet(
                f"font-size: 36px; font-weight: 700; color: {Colors.SUCCESS}; "
                f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
            )
            # Show vault button, hide retry
            self._btn_open_vault.setVisible(True)
            self._btn_retry.setVisible(False)
        else:
            self._icon.setText("🔒")
            self._title.setText("ACCESS DENIED")
            self._title.setStyleSheet(
                f"font-size: 36px; font-weight: 700; color: {Colors.DANGER}; "
                f"font-family: 'Exo 2', 'Segoe UI', sans-serif; letter-spacing: 4px;"
            )
            # Show retry button, hide vault
            self._btn_open_vault.setVisible(False)
            self._btn_retry.setVisible(True)
        self._detail.setText(
            f"{result.message}\n\n"
            f"Target P300: {result.target_peak_uv:.2f} µV  |  "
            f"Non-target: {result.nontarget_peak_uv:.2f} µV  |  "
            f"SNR: {result.snr_db:.1f} dB"
        )


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
        if symbol:
            painter.setFont(QFont("Segoe UI Symbol", 36, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        else:
            painter.setFont(QFont("Arial", 14))
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, str(img_id))
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

        self._home_screen       = HomeScreen(self)
        self._stimulus_screen   = StimulusScreen(self)
        self._result_screen     = ResultScreen(self)
        self._vault_screen      = VaultScreen(self)
        self._enrollment_screen = EnrollmentScreen(self)

        self._stack.addWidget(self._home_screen)       # 0
        self._stack.addWidget(self._stimulus_screen)   # 1
        self._stack.addWidget(self._result_screen)     # 2
        self._stack.addWidget(self._vault_screen)      # 3
        self._stack.addWidget(self._enrollment_screen) # 4

        self._home_screen.sig_start_auth.connect(self._start_authentication)
        self._home_screen.sig_enroll.connect(self._go_enrollment)
        self._result_screen.sig_retry.connect(self._start_authentication)
        self._result_screen.sig_back.connect(self._go_home)
        self._result_screen.sig_open_vault.connect(self._open_vault)
        self._vault_screen.sig_lock.connect(self._go_home)
        self._enrollment_screen.sig_confirmed.connect(self._on_enrollment_done)

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
        self._stack.setCurrentIndex(0)

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

        self._stimulus_screen.set_status("SCANNING…  FOCUS ON YOUR KEY IMAGES", Colors.ACCENT)

    @pyqtSlot(object)
    def _on_paradigm_done(self, result: AuthResult):
        if self._paradigm_thread:
            self._paradigm_thread.quit()
            self._paradigm_thread.wait()
        self._result_screen.show_result(result)
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
