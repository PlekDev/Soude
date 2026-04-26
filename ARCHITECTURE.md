# Soude — Architecture & Team Playbook
## Br41n.IO Hackathon — 36 Hours Remaining

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           SOUDE MVP                                 │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Unicorn HB  │────▶│         brain_engine.py                  │  │
│  │  (UnicornPy) │     │  ┌────────────┐   ┌──────────────────┐  │  │
│  │  8-ch EEG    │     │  │ RealUnicorn│   │  MockUnicorn     │  │  │
│  │  250 Hz      │     │  │ (hardware) │   │  (simulator)     │  │  │
│  └──────────────┘     │  └─────┬──────┘   └────────┬─────────┘  │  │
│                        │        └──────────┬─────────┘            │  │
│  ┌──────────────┐     │              ┌─────▼──────┐               │  │
│  │  MockUnicorn │     │              │ RingBuffer  │               │  │
│  │  (dev/test)  │     │              │ 5s × 8ch   │               │  │
│  └──────────────┘     │              └─────┬──────┘               │  │
│                        │                    │  mark_stimulus()     │  │
│                        └────────────────────┼─────────────────────┘  │
│                                             │                        │
│  ┌──────────────────────────────────────────▼─────────────────────┐  │
│  │                  signal_processing.py                          │  │
│  │                                                                │  │
│  │  OnlineFilter (1-10 Hz BP + 60 Hz Notch)                      │  │
│  │      │                                                         │  │
│  │  EpochExtractor ──▶ filter_epoch ──▶ baseline_correct          │  │
│  │      │                   │                │                    │  │
│  │  artifact_reject         │           SignalAverager             │  │
│  │                          │         target/non-target            │  │
│  │                          ▼                │                    │  │
│  │                   P300 peak (250-500ms)   │                    │  │
│  │                   Cz + Pz + Oz avg        │                    │  │
│  │                          │                │                    │  │
│  │                   AuthenticationPipeline ◀┘                    │  │
│  │                   ΔP300 ≥ 1.5 µV → GRANT                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              │ AuthResult                            │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                       app.py (PyQt6)                         │    │
│  │                                                              │    │
│  │  HomeScreen ──▶ StimulusScreen ──▶ ResultScreen              │    │
│  │                      │                                       │    │
│  │              ParadigmWorker (QThread)                        │    │
│  │                      │                                       │    │
│  │              StimulusRunner ──────────────────────────────▶  │    │
│  │              (stimulus_runner.py)   mark_stimulus()          │    │
│  │              400ms SOA oddball                               │    │
│  │              precise busy-wait                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                        data_logger.py                                │
│                        erp_viewer.py                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Map

| File | Owner | Purpose |
|---|---|---|
| `brain_engine.py` | Sub-team 1 | UnicornPy wrapper, RingBuffer, stimulus marking |
| `signal_processing.py` | Sub-team 2 | Filters, epoching, averaging, auth decision |
| `stimulus_runner.py` | Sub-team 3 | Oddball paradigm, precise timing |
| `app.py` | Sub-team 3 | PyQt6 UI — Home, Stimulus, Result screens |
| `erp_viewer.py` | Sub-team 2/4 | Live ERP plotting widget |
| `data_logger.py` | Sub-team 4 | Session CSV/NPZ logging, impedance check |
| `test_pipeline.py` | Sub-team 4 | End-to-end validation without hardware |

---

## Sub-Team Responsibilities

### Sub-team 1 — Hardware/API (2 people)
**Files:** `brain_engine.py`

**Priority tasks (next 12 h):**
1. Connect Unicorn HB, verify `UnicornPy.GetAvailableDevices()` returns the serial.
2. Run `BrainEngine(serial="UN-XXXX.XX.XX").start()` — confirm no exceptions.
3. Check impedance via Unicorn Recorder software first. All channels should be green (<10 kΩ).
4. Verify ring buffer fills correctly: `engine.buffer.total_written` should increment at ~250/s.
5. Test `mark_stimulus()` latency: measure `time.perf_counter()` before/after and confirm <0.5 ms overhead.

**Error codes to handle:**
- `UnicornPy.DeviceException` code 1 = no device found → check USB + driver
- Code 7 = acquisition already running → call `StopAcquisition()` first
- Flat signal on all channels = impedance check failed → re-gel electrodes

---

### Sub-team 2 — Signal/Math (3 people)
**Files:** `signal_processing.py`, `erp_viewer.py`

**Priority tasks (next 12 h):**
1. Run `erp_viewer.py` standalone to validate the plot widget.
2. Collect 5 minutes of resting EEG, run through `filter_epoch()`, confirm 1-10 Hz bandpass shapes the signal correctly (should see alpha ~8-12 Hz attenuated above 10 Hz).
3. Run a manual P300 session: show the user the password images **outside** the paradigm first, then run the paradigm. Collect 10 runs and plot the grand averages in `erp_viewer.py`.
4. Tune `AUTH_THRESHOLD_UV` in `signal_processing.py` based on real SNR measurements.
5. If SNR is low: increase `TARGET_REPEATS` in `stimulus_runner.py` from 5 to 8.

**Key parameters to tune:**
```python
# signal_processing.py
AUTH_THRESHOLD_UV = 1.5   # Start here; increase if false accepts occur
MIN_EPOCHS = 3             # Lower to 2 if artifact rejection is aggressive
EPOCH_DURATION_S = 0.800  # Keep at 800ms — captures full P300 window

# stimulus_runner.py
TARGET_REPEATS = 5         # More repetitions = better SNR but longer scan
SOA_S = 0.400              # Minimum reliable SOA for P300; don't go lower
```

---

### Sub-team 3 — UX/UI (3 people)
**Files:** `app.py`, `stimulus_runner.py`, `assets/`

**Priority tasks (next 12 h):**
1. Set `UNICORN_SERIAL = ""` in `app.py` to run in simulator mode and build full UI flow.
2. Create 20 image assets in `assets/images/00.png` through `19.png`. Use distinctive, memorable images (faces, animals, symbols work best for P300).
3. Embed `ERPViewer` widget into the result screen for a real-time display during demos.
4. Add an enrollment screen: allow user to click 3 "password" images before the scan.

**Image guidelines:**
- Size: 480×480 px minimum
- Format: PNG, transparent background OK
- Content: Distinct categories (face, animal, object, scene, symbol)
- Avoid: Similar images that could cause visual confusion
- The 3 password images should feel personally meaningful to the user

---

### Sub-team 4 — Integration/Demo (2 people)
**Files:** `test_pipeline.py`, `data_logger.py`, pitch deck

**Priority tasks (next 12 h):**
1. Run `python test_pipeline.py` after every significant code change to catch regressions.
2. Integrate `SessionLogger` into `app.py`'s `_on_paradigm_done()` handler:
   ```python
   from data_logger import SessionLogger
   session = SessionLogger()
   session.log_auth_result(result, pipeline.get_erp_data())
   session.flush()
   ```
3. Build the pitch demo flow: enroll → scan → show ERP + auth result.
4. Prepare fallback: if headset fails during demo, launch with `MockUnicorn` and explain P300 science with the synthetic ERP.

**Demo Script (3 minutes):**
- 0:00 — Show EEG headset being put on; explain P300 protocol
- 0:30 — Enrollment: user secretly picks 3 images as "password"
- 1:00 — Run paradigm (14 seconds); live waveform visible
- 1:15 — ERP plot appears; target waveform peaks at 300ms
- 1:30 — "ACCESS GRANTED" screen
- 1:45 — Explain security: no keyboard, no eye-tracking, brain-unique response
- 2:00 — Show false attempt: different user, same images → DENIED

---

## Quickstart

```bash
# 1. Install Python deps
pip install numpy scipy PyQt6

# 2. Add UnicornPy to Python path (Windows, after SDK install)
set PYTHONPATH=C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python

# 3. Set your device serial (or leave blank for simulator)
set UNICORN_SERIAL=UN-2023.10.01

# 4. Run tests (no headset needed)
python test_pipeline.py

# 5. Launch app
python app.py

# 6. Live ERP monitor (standalone, for Sub-team 2 validation)
python erp_viewer.py
```

---

## P300 Science — Quick Reference for the Pitch

The **P300** is an event-related potential: a positive voltage deflection peaking ~300ms after a **rare, meaningful stimulus** (the "oddball"). It is generated in posterior parietal cortex and reflects **cognitive recognition** — not just seeing, but *knowing*.

**Why it's secure:**
- The response is **involuntary** — you cannot suppress the P300 to your own password.
- It is **individual** — peak latency varies ±30ms between people.
- It is **covert** — there is nothing to steal: no stored hash, no key material.

**Our protocol:**
- 20 images flashed at 400ms intervals
- 3 are the user's "mental password"
- Signal averaging across 5 repetitions cancels noise (improves SNR by √5 ≈ 2.2×)
- Authentication compares the 250–500ms window of target vs. non-target averages

---

## Contingency Plans

| Risk | Mitigation |
|---|---|
| Headset dropout during demo | Auto-fallback to MockUnicorn; present as "simulator mode" |
| High noise / poor SNR | Re-gel electrodes; increase `TARGET_REPEATS` to 8 |
| No P300 detected | User re-enrollment; verify they focused on target images |
| PyQt6 not available | Run `erp_viewer.py` demo only via matplotlib backend |
| Windows permission error on thread priority | Non-fatal; remove the ctypes block in `brain_engine.py` |
