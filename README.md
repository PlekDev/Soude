# 🧠 Neuro-Lock — Brainwave Password Manager

> **Your password is in your head. Literally.**

Neuro-Lock is a **P300 ERP-based password manager** that authenticates users with their own brainwaves using the [g.tec Unicorn Hybrid Black](https://www.unicorn-bi.com/) EEG headset. There is no keyboard, no mouse, no passphrase to type — just a brief 14-second brain scan that unlocks your credential vault.

Built in 36 hours for the **Br41n.IO Hackathon**.

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
git clone https://github.com/<your-team>/neuro-lock.git
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

### `signal_processing.py`

| Parameter | Default | Effect |
|---|---|---|
| `AUTH_THRESHOLD_UV` | `1.5 µV` | Minimum ΔP300 for GRANT. Increase if false accepts occur. |
| `MIN_EPOCHS` | `3` | Min epochs per class. Lower to `2` if artifact rejection is aggressive. |
| `EPOCH_DURATION_S` | `0.800 s` | Post-stimulus window. Keep at 800 ms — captures full P300. |
| `P300_ONSET_S` | `0.250 s` | Start of P300 detection window. |
| `P300_OFFSET_S` | `0.500 s` | End of P300 detection window. |

### `stimulus_runner.py`

| Parameter | Default | Effect |
|---|---|---|
| `TARGET_REPEATS` | `5` | Repetitions per password image. More = better SNR, longer scan. |
| `SOA_S` | `0.400 s` | Stimulus Onset Asynchrony. Do not lower below 350 ms. |
| `NONTARGET_REPEATS` | `1` | Repetitions for non-password images. |

### `brain_engine.py`

| Parameter | Default | Effect |
|---|---|---|
| `SAMPLE_RATE` | `250 Hz` | Unicorn Hybrid Black native rate. Do not change. |
| `N_CHANNELS` | `8` | EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8). |
| `BUFFER_SECONDS` | `5 s` | Ring buffer duration. Covers one full paradigm run. |
| `GETDATA_BLOCK` | `4 samples` | Pull size (~16 ms latency). Reduce to 1 for minimum latency. |

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

## Demo Script (3 minutes)

| Time | Action |
|---|---|
| 0:00 | Put on EEG headset; explain P300 in one sentence |
| 0:30 | Show enrollment: user secretly picks 3 images as password |
| 1:00 | Hit **BEGIN NEURAL SCAN** — images flash for 14 s |
| 1:15 | ERP monitor shows green waveform peaking at ~300 ms |
| 1:30 | **ACCESS GRANTED** → **OPEN VAULT** → passwords appear |
| 1:45 | Explain: no keyboard, no hash, brain response cannot be faked |
| 2:00 | False attempt: different user, same password images → **ACCESS DENIED** |
| 2:30 | Q&A — mention SNR, threshold tuning, anti-spoofing properties |

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

Built at the **Br41n.IO Hackathon** by team **SOUBE**:

| Role | Sub-team |
|---|---|
| Hardware / API integration | Sub-team 1 |
| Signal processing & math | Sub-team 2 |
| UX / UI / stimulus design | Sub-team 3 |
| Integration, logging & demo | Sub-team 4 |

---

## License

MIT — see `LICENSE` file.
