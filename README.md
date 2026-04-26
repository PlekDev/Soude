# 🧠 Neuro-Lock — Brainwave Password Manager

> **Your password is in your head. Literally.**

Neuro-Lock is a **P300 ERP-based password manager** that authenticates users with their own brainwaves using the [g.tec Unicorn Hybrid Black](https://www.unicorn-bi.com/) EEG headset. There is no keyboard, no mouse, no passphrase to type — just a brief brain scan that unlocks your credential vault.

Built for the **Br41n.IO Hackathon** by team **SOUDE**.

---

## What Is P300 Authentication?

The **P300** is an involuntary electrical brain response that occurs approximately 300 ms after a person mentally recognises a rare, meaningful stimulus (the "oddball"). It is:

- **Involuntary** — you cannot suppress your P300 response to your own password images.
- **Individual** — peak latency and amplitude vary between people, making the signal person-specific.
- **Covert** — there is no stored hash, no key material, nothing to steal. The secret exists only in the brain.

### How it works

1. The user privately selects 3 images from a pool of 20 as their "mental password" during enrollment.
2. At login, all 20 images flash at 400 ms intervals (the **oddball paradigm**).
3. Each time a target (password) image appears, the brain produces a characteristic P300 deflection on centroparietal channels (Cz, Pz, Oz).
4. Neuro-Lock averages 5 repetitions per image, computes the mean P300 amplitude in the 250–500 ms window, and compares target vs. non-target averages.
5. If the target P300 exceeds the non-target by ≥ 1.5 µV, the vault unlocks.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NEURO-LOCK                                   │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Unicorn HB  │────▶│           brain_engine.py                │  │
│  │  (UnicornPy) │     │  ┌─────────────┐  ┌──────────────────┐  │  │
│  │  8-ch EEG    │     │  │ RealUnicorn │  │  MockUnicorn     │  │  │
│  │  250 Hz      │     │  │ (hardware)  │  │  (simulator)     │  │  │
│  └──────────────┘     │  └──────┬──────┘  └────────┬─────────┘  │  │
│                        │         └──────────┬────────┘            │  │
│                        │               ┌────▼──────┐              │  │
│                        │               │ RingBuffer│              │  │
│                        │               │  5 s×8 ch │              │  │
│                        │               └────┬──────┘              │  │
│                        └────────────────────┼────────────────────┘  │
│                                             │ mark_stimulus()        │
│  ┌──────────────────────────────────────────▼──────────────────────┐ │
│  │                  signal_processing.py                           │ │
│  │  1–10 Hz bandpass + 60 Hz notch  →  EpochExtractor             │ │
│  │  baseline_correct  →  artifact_reject  →  SignalAverager        │ │
│  │  P300 peak (250–500 ms, Cz+Pz+Oz)  →  AuthenticationPipeline  │ │
│  │  ΔP300 ≥ 1.5 µV  →  GRANT                                     │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                              │ AuthResult                             │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                     app.py  (PyQt6 UI)                          │ │
│  │  HomeScreen → StimulusScreen → ResultScreen → VaultScreen       │ │
│  │  ParadigmWorker (QThread) + StimulusRunner (400 ms SOA)         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│              data_logger.py          erp_viewer.py                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

| File | Role |
|---|---|
| `app.py` | PyQt6 UI — HomeScreen, StimulusScreen, ResultScreen, VaultScreen |
| `brain_engine.py` | UnicornPy wrapper, thread-safe RingBuffer, stimulus marker management |
| `signal_processing.py` | Bandpass/notch filters, epoch extraction, P300 averaging, auth decision |
| `stimulus_runner.py` | Oddball paradigm scheduler with sub-millisecond timing |
| `erp_viewer.py` | Live ERP waveform plot widget (standalone or embedded) |
| `data_logger.py` | Session CSV/NPZ logger, impedance quality checker |
| `test_pipeline.py` | End-to-end validation suite (no headset required) |
| `ARCHITECTURE.md` | Internal team playbook and tuning guide |

---

## Prerequisites

### Software

```bash
pip install numpy scipy PyQt6
```

### Unicorn SDK (for real hardware)

1. Install the **g.tec Unicorn Suite** (Windows only, [download here](https://www.unicorn-bi.com/)).
2. Add UnicornPy to your Python path:

```bat
set PYTHONPATH=C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python
```

### Python version

Python 3.10+ recommended (uses `list[int]` type hints throughout).

---

## Installation

```bash
git clone https://github.com/PlekDev/Soude.git
cd neuro-lock
pip install numpy scipy PyQt6
```

---

## Usage Flow

### Step 1 — Run without hardware (simulator mode)

Leave `UNICORN_SERIAL` unset (or empty). The app automatically falls back to `MockUnicorn`, which generates synthetic pink-noise EEG and injects realistic P300 pulses for target images.

```bash
python app.py
```

### Step 2 — Run with the real Unicorn Hybrid Black

```bat
set UNICORN_SERIAL=UN-2023.10.01   :: replace with your device serial
set PYTHONPATH=C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python
python app.py
```

Or on PowerShell:

```powershell
$env:UNICORN_SERIAL = "UN-2023.10.01"
$env:PYTHONPATH = "C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python"
python app.py
```

### Step 3 — Enrollment (first time)

Open `app.py` and set the three password image IDs:

```python
DEFAULT_PASSWORD_IDS = [3, 11, 17]   # change to your chosen image indices (0–19)
```

The 20 images live in `assets/images/00.png` through `19.png`. Use the image ID numbers to pick your mental password.

### Step 4 — Authentication loop

```
Home Screen
    │
    ▼  [▶ BEGIN NEURAL SCAN]
Stimulus Screen  (20 images flash × 5 repetitions ≈ 14 seconds)
    │
    ▼  (P300 evaluated)
Result Screen
    ├── GRANTED → [▶ OPEN VAULT]  →  Vault Screen (passwords visible)
    │                                      │
    │                               [🔒 LOCK VAULT]  →  Home Screen
    │
    └── DENIED  → [⟳ RETRY SCAN]  →  Stimulus Screen
                  [← BACK]        →  Home Screen
```

### Step 5 — ERP monitor (signal validation)

Run the standalone ERP viewer to inspect live waveforms:

```bash
python erp_viewer.py
```

### Step 6 — Pipeline self-test (no headset)

```bash
python test_pipeline.py
```

All 5 stages should pass in ~20 seconds.

---

## Configuration Parameters

All key parameters are documented in-file. The most important ones to tune when using real hardware:

### `Fase1/signal_processing.py`

| Parameter | Default | Effect |
|---|---|---|
| `AUTH_THRESHOLD_UV` | `0.8 µV` | Minimum ΔP300 for GRANT. Increase if false accepts occur. |
| `MIN_EPOCHS` | `2` | Min epochs per class required for a valid decision. |
| `EPOCH_DURATION_S` | `0.800 s` | Post-stimulus window. Keep at 800 ms — captures full P300. |
| `P300_ONSET_S` | `0.250 s` | Start of P300 detection window. |
| `P300_OFFSET_S` | `0.500 s` | End of P300 detection window. |

### `Fase1/stimulus_runner.py`

| Parameter | Default | Effect |
|---|---|---|
| `TARGET_REPEATS` | `5` | Repetitions per password image. More = better SNR, longer scan. |
| `NONTARGET_REPEATS` | `5` | Repetitions per non-target. Must equal TARGET_REPEATS to keep target rate at 15%. |
| `SOA_S` | `0.500 s` | Stimulus Onset Asynchrony. Do not lower below 400 ms. |
| `BLANK_S` | `0.075 s` | Inter-stimulus blank interval. |

> **Total scan duration:** 100 events × 0.575 s ≈ **57 seconds**

### `brain_engine.py`

| Parameter | Default | Effect |
|---|---|---|
| `SAMPLE_RATE` | `250 Hz` | Unicorn Hybrid Black native rate. Do not change. |
| `N_CHANNELS` | `8` | EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8). |
| `BUFFER_SECONDS` | `120 s` | Ring buffer — must exceed paradigm + enrollment time (~1.9 MB, negligible). |
| `GETDATA_BLOCK` | `4 samples` | Pull size (~16 ms latency). Reduce to 1 for minimum latency. |

---

## UI Customization (PyQt6)

The interface uses a cyberpunk/neuro aesthetic built entirely with PyQt6. Below are the key techniques used to achieve the look and feel shown in the app.

### Window Icon & Title

```python
self.setWindowTitle("NEURO-LOCK")
self.setWindowIcon(QIcon("assets/brain_icon.png"))
```

Place a 256×256 PNG in `assets/` for best results. On Windows, an `.ico` file is preferred:

```python
from PIL import Image
img = Image.open("assets/brain_icon.png")
img.save("assets/brain_icon.ico", format="ICO", sizes=[(256,256),(64,64),(32,32)])
```

### Custom Fonts

Download **Orbitron** and **Share Tech Mono** from [Google Fonts](https://fonts.google.com/) and place them in `assets/fonts/`. Load them at startup before any window is created:

```python
def load_custom_fonts():
    fonts_dir = Path("assets/fonts")
    for font_file in fonts_dir.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(font_file))

# In main(), before QMainWindow instantiation
load_custom_fonts()

# Usage
label.setFont(QFont("Orbitron", 28, QFont.Weight.Bold))
```

### Global Stylesheet

Apply a single stylesheet to `QApplication` so all widgets inherit the theme:

```python
NEURO_STYLESHEET = """
QMainWindow, QWidget#centralWidget {
    background-color: #050505;
}
QWidget {
    font-family: 'Share Tech Mono', monospace;
    color: #b4ff00;
}
QPushButton {
    background-color: transparent;
    border: 1px solid #b4ff00;
    border-radius: 4px;
    color: #b4ff00;
    font-family: 'Share Tech Mono';
    font-size: 13px;
    letter-spacing: 3px;
    padding: 14px 24px;
}
QPushButton:hover {
    background-color: rgba(180, 255, 0, 0.08);
    border-color: #ccff33;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: rgba(180, 255, 0, 0.2);
}
QPushButton:disabled {
    border-color: #333;
    color: #333;
}
QLabel#titleLabel {
    color: #b4ff00;
    font-family: 'Orbitron';
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 8px;
}
QFrame#panel {
    background-color: #0d0d0d;
    border: 1px solid rgba(180, 255, 0, 0.25);
    border-radius: 6px;
}
QProgressBar {
    background-color: #111;
    border: 1px solid rgba(180, 255, 0, 0.3);
    border-radius: 3px;
    height: 6px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4dff00, stop:1 #b4ff00);
    border-radius: 3px;
}
"""

app = QApplication(sys.argv)
app.setStyleSheet(NEURO_STYLESHEET)
```

### Glow Effect on Key Widgets

```python
def add_glow(widget: QWidget, color: str = "#b4ff00", radius: int = 20) -> None:
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(radius)
    effect.setColor(QColor(color))
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)

# Usage
add_glow(self.title_label, radius=30)
add_glow(self.scan_button, radius=15)
```

> **Note:** PyQt6 allows only one `QGraphicsEffect` per widget. Adding a second replaces the first.

### Frameless Window with Drag Support

Remove the native OS title bar and implement custom dragging:

```python
class NeurolockWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._drag_pos:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
```

### Signal Badge Pulse Animation

The EEG channel badges (Fz, C3, Cz…) animate when a signal is active:

```python
# In your QSS, add a named class or use objectName
# In Python, toggle a property and call style().unpolish/polish to re-apply
badge.setProperty("active", True)
badge.style().unpolish(badge)
badge.style().polish(badge)
```

```css
/* QSS for active badge */
QPushButton[active="true"] {
    background-color: #b4ff00;
    color: #050505;
    border-color: #b4ff00;
}
```

### Design Token Reference

Centralise all theme values to avoid magic numbers scattered throughout the code:

```python
THEME = {
    "accent":       "#b4ff00",
    "accent_dim":   "rgba(180, 255, 0, 0.25)",
    "bg_primary":   "#050505",
    "bg_secondary": "#0d0d0d",
    "text_primary": "#b4ff00",
    "text_muted":   "#555555",
    "font_display": "Orbitron",
    "font_mono":    "Share Tech Mono",
    "radius":       "4px",
    "glow_radius":  20,
}
```

### Asset Folder Structure

```
neuro-lock/
├── assets/
│   ├── fonts/
│   │   ├── Orbitron-Bold.ttf
│   │   └── ShareTechMono-Regular.ttf
│   ├── images/
│   │   └── 00.png … 19.png
│   ├── brain_icon.png    ← 256×256, used on Linux/macOS
│   └── brain_icon.ico    ← multi-size, used on Windows
```

### Packaging with PyInstaller

Bundle the app into a single executable with the custom icon included:

```bash
pyinstaller --onefile --windowed --icon=assets/brain_icon.ico app.py
```

Add `--add-data "assets;assets"` (Windows) or `--add-data "assets:assets"` (macOS/Linux) to bundle the fonts and images.

---

## Connecting the Real Unicorn

1. **Charge** the headset and confirm the battery LED is green.
2. **Apply gel** to all 8 electrodes. Run Unicorn Recorder's impedance check — all channels should be below 10 kΩ (green).
3. **Power on** the headset. It appears as a Bluetooth COM device. Unicorn Suite must be installed.
4. Find your serial number: run `UnicornPy.GetAvailableDevices(True)` in a Python REPL or check the label on the headset.
5. Set `UNICORN_SERIAL` as shown above and launch `app.py`.
6. The Home Screen status indicator will turn green and read **DEVICE: READY — UN-XXXX.XX.XX**.

### Error codes

| Code | Meaning | Fix |
|---|---|---|
| 1 | Device not found | Check USB dongle/Bluetooth, verify serial |
| 7 | Acquisition already running | Call `StopAcquisition()` or reboot headset |
| Flat signal | Electrode contact lost | Re-gel and re-seat electrodes |

---

## Hardware Setup — Electrode Placement

The Unicorn Hybrid Black uses the standard 10-20 system with 8 electrodes:

```
         Fz (ch 0)
    C3 (ch 1)  C4 (ch 3)
         Cz (ch 2)  ←─── P300 channels
    PO7 (ch 5)  PO8 (ch 7)
         Pz (ch 4)  ←─── P300 channels
         Oz (ch 6)  ←─── P300 channels
```

P300 detection focuses on **Cz, Pz, Oz** (centroparietal/occipital), where the P300 amplitude is maximal.

---

## Image Guidelines

Place 20 images in `assets/images/` named `00.png` through `19.png`.

- **Size:** 480×480 px minimum
- **Format:** PNG (transparent background OK)
- **Content:** Distinct semantic categories — faces, animals, objects, symbols, scenes
- **Avoid:** Images that look visually similar (reduces P300 discrimination)
- **Tip:** The 3 password images should feel personally meaningful — the P300 is driven by cognitive significance

---

## Session Logs

Every authentication run is saved to `logs/<timestamp>/`:

```
logs/20260425_022022/
    markers.csv       — stimulus event log (image_id, timestamp, is_target)
    epochs.npz        — raw epoch array per marker (numpy compressed)
    auth_result.json  — final auth decision + ERP data
    summary.txt       — human-readable summary
```

---

## Demo Script (~5 minutes)

| Time | Action |
|---|---|
| 0:00 | Put on EEG headset; explain P300 in one sentence |
| 0:30 | Show enrollment: user secretly picks 3 images as password (★ STAR, ♦ DIAMOND, ● CIRCLE) |
| 1:00 | Hit **BEGIN NEURAL SCAN** — images flash for ~57 s (100 events, 5 repeats each) |
| 2:00 | While scanning: explain the oddball paradigm and P300 latency |
| 2:10 | **ACCESS GRANTED** → **OPEN VAULT** → passwords appear |
| 2:30 | Explain: no keyboard, no hash, brain response cannot be faked |
| 3:00 | False attempt: different user, same password images → **ACCESS DENIED** |
| 4:00 | Q&A — mention SNR, threshold tuning, anti-spoofing properties |

---

## Contingency Plans

| Risk | Mitigation |
|---|---|
| Headset dropout during demo | Auto-fallback to MockUnicorn; present as "simulator mode" |
| Poor SNR / no P300 detected | Re-gel electrodes; increase `TARGET_REPEATS` to 8 |
| User can't focus | Have them re-enroll with more personally meaningful images |
| PyQt6 not available | Run `python erp_viewer.py` demo only (matplotlib-free) |
| Windows thread priority error | Non-fatal; ctypes block in `brain_engine.py` is silently skipped |

---

## Security Properties

| Property | Detail |
|---|---|
| **No stored secret** | The "password" is a set of image IDs; the key is in the user's neural response |
| **Anti-spoofing** | Passive playback of EEG cannot be used — the paradigm requires live, time-locked acquisition |
| **Signal averaging** | 5 repetitions × 3 targets = 15 target epochs; noise cancels as 1/√N |
| **Artifact rejection** | Epochs with peak-to-peak amplitude > 100 µV are automatically discarded |
| **No eye-tracking required** | The user only needs to mentally acknowledge target images, not fixate |

---

## Tech Stack

- **EEG hardware:** g.tec Unicorn Hybrid Black (8 ch, 250 Hz)
- **Python 3.10+**
- **NumPy / SciPy** — signal processing, filtering, epoch averaging
- **PyQt6** — desktop UI, threading, custom paint widgets
- **UnicornPy** — g.tec SDK for hardware acquisition

---

## Team

Built at the **Br41n.IO Hackathon** by team **SOUDE**:

| Role | Sub-team |
|---|---|
| Hardware / API integration | Sub-team 1 |
| Signal processing & math | Sub-team 2 |
| UX / UI / stimulus design | Sub-team 3 |
| Integration, logging & demo | Sub-team 4 |

---

## License

MIT — see `LICENSE` file.