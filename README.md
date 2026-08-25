# 🧠 Soude — Brainwave Password Manager

> **Your password is in your head. Literally.**

Soude is a **P300 ERP-based authentication system** that verifies users with their own brainwaves using the [g.tec Unicorn Hybrid Black](https://www.unicorn-bi.com/) EEG headset (8 channels, 250 Hz). No keyboard, no passphrase — a brief brain scan against a set of mentally-selected images unlocks the vault.

Born at the **Br41n.IO Hackathon** (team SOUDE); now under active development toward congress-grade results.

---

## How it works

1. During enrollment the user privately selects **3 of 20 images** as their "mental password".
2. At login all 20 images flash in random order (**oddball paradigm**, 500 ms SOA + 75 ms blank, ~57.5 s, target rate 15%).
3. Each password image evokes a **P300** — an involuntary positive voltage deflection ~300 ms after a recognized, rare stimulus — strongest on **Cz, Pz, Oz**.
4. Epochs (800 ms post-stimulus) are band-passed 1–10 Hz + 60 Hz notch, baseline-corrected, artifact-rejected (>100 µV p-p) and averaged per class.
5. Access is **granted** only if the mean target-vs-non-target difference in the 250–500 ms window is **positive** and exceeds both the amplitude threshold and 1.5× the pre-stimulus noise floor.

---

## Project structure

```text
Soude/
├── run.py                     # entry point:  python run.py
├── requirements.txt
├── .env.example               # plantilla de configuración (copiar a .env)
├── conftest.py
├── neurolock/                 # paquete núcleo
│   ├── brain_engine.py        # RealUnicorn / MockUnicorn / LSLUnicorn + RingBuffer
│   ├── filters.py             # filtros SOS (P300 1–10 Hz, mu/beta 8–30 Hz, notch 60 Hz)
│   ├── signal_processing.py   # épocas, promediado, decisión de autenticación
│   ├── stimulus_runner.py     # paradigma oddball con timing sub-ms
│   ├── data_logger.py         # sesiones a logs/<timestamp>/
│   ├── signal_quality.py      # heurístico de impedancia por varianza
│   └── ui/                    # app PyQt6 (app.py) + visor ERP (erp_viewer.py)
├── ml/                        # Fase 2: clasificador Rest/Passthought (SVM)
│   ├── create_database.py     # recolección → data/dataset_phase2.csv
│   ├── train_model.py         # entrenamiento con validación honesta
│   └── test_model.py          # inferencia en vivo (espejo del entrenamiento)
├── tools/                     # utilerías standalone
│   ├── eeg_visualizer.py      # osciloscopio EEG en vivo
│   ├── debug_viewer.py        # inspector post-mortem de sesiones grabadas
│   ├── lsl_sender.py          # emisor LSL (modo dos máquinas)
│   └── lsl_monitor.py         # monitor de consola (--source mock|lsl|real)
├── tests/                     # pytest
├── assets/                    # fuentes, íconos (imágenes 00–19.png opcionales)
├── docs/                      # ARCHITECTURE.md (playbook del hackathon), guiones
├── data/                      # datasets y modelos generados (fuera de git)
└── logs/                      # sesiones grabadas (fuera de git)
```

---

## Installation

```bash
git clone https://github.com/PlekDev/Soude.git
cd Soude
pip install -r requirements.txt
```

Python 3.10+. For real hardware, install the **g.tec Unicorn Suite** (Windows) — the `UnicornPy` SDK is not pip-installable; point `UNICORN_SDK_PATH` at it (see below).

---

## Running

### Simulator mode (no hardware)

```bash
python run.py
```

Without `UNICORN_SERIAL` the app uses `MockUnicorn` (synthetic pink-noise EEG with realistic injected P300s). Everything — enrollment, scan, vault, logging — works.

### Real headset

Copy `.env.example` to `.env` and fill in:

```ini
UNICORN_SERIAL=UN-XXXX.XX.XX
UNICORN_SDK_PATH=C:\Program Files\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib
```

Gel all 8 electrodes, verify impedance in Unicorn Recorder (<10 kΩ), then `python run.py`.

### Two-machine mode (LSL over LAN)

The machine with the licensed dongle runs `python tools/lsl_sender.py` (or the main app — the engine always re-broadcasts as LSL stream `Unicorn_EEG`). The receiving machine sets `UNICORN_SERIAL=LSL` in its `.env` and runs `python run.py` normally.

### Environment switches

| Variable | Effect |
|---|---|
| `UNICORN_SERIAL` | empty = simulator · serial = real headset · `LSL` = network receiver |
| `UNICORN_SDK_PATH` | folder containing `UnicornPy` (real hardware only) |
| `SOUDE_DEBUG=1` | faulthandler + Qt debug logging |
| `SOUDE_SESSION_TYPE=impostor` | marks the session log as a controlled impostor attempt (for FAR analysis); default `genuine` |

---

## Tests

```bash
pytest
```

Five stages run without hardware in ~25 s: ring buffer, filters (including real 60 Hz attenuation asserts), mock P300 injection, the full authentication pipeline (must GRANT on injected P300s), and the session logger (structured metrics verified).

---

## Fase 2 — Rest/Passthought classifier (ML)

```bash
python ml/create_database.py        # graba ventanas etiquetadas (casco real)
python ml/train_model.py            # accuracy honesta: split 70/30 + CV 5-fold
python ml/test_model.py             # inferencia en vivo, espejo exacto del entrenamiento
```

All three accept `--mock` to exercise the flow without hardware (mock data trains at chance level, by design). Datasets and the pickled `{model, scaler}` live in `data/` (git-ignored).

---

## Session logs

Every authentication run writes `logs/<timestamp>/`:

- `markers.csv` — stimulus event log (image_id, timestamp, is_target)
- `epochs.npz` — raw epoch array per marker
- `auth_result.json` — decision **plus structured metrics** (`delta_uv`, `pre_sigma_uv`, `n_target`, `n_nontarget`, `snr_db`, `session_type`) so FAR/FRR/EER/ROC can be recomputed offline with different thresholds
- `summary.txt` — human-readable summary

Inspect any session with `python tools/debug_viewer.py logs/<timestamp>`.

---

## Key parameters

| Where | Parameter | Default | Effect |
|---|---|---|---|
| `neurolock/signal_processing.py` | `AUTH_THRESHOLD_UV` | 1.5 µV | Minimum positive ΔP300 for GRANT |
| `neurolock/signal_processing.py` | `MIN_EPOCHS` | 5 | Min clean epochs per class |
| `neurolock/signal_processing.py` | `EPOCH_DURATION_S` | 0.800 s | Post-stimulus window |
| `neurolock/stimulus_runner.py` | `TARGET_REPEATS` | 5 | Repetitions per password image (SNR ∝ √N) |
| `neurolock/stimulus_runner.py` | `SOA_S` | 0.500 s | Stimulus onset asynchrony |
| `neurolock/brain_engine.py` | `SAMPLE_RATE` | 250 Hz | Unicorn native rate — do not change |
| `neurolock/brain_engine.py` | `BUFFER_SECONDS` | 120 s | Ring buffer span |

---

## Electrode placement (10-20 system)

```text
         Fz (ch 0)
    C3 (ch 1)  C4 (ch 3)
         Cz (ch 2)  ←─ P300
    PO7 (ch 5)  PO8 (ch 7)
         Pz (ch 4)  ←─ P300
         Oz (ch 6)  ←─ P300
```

Fase 2 (motor imagery) uses **C3/C4**; P300 detection uses **Cz/Pz/Oz**.

| Error | Meaning | Fix |
|---|---|---|
| Code 1 | Device not found | Check Bluetooth/dongle and serial |
| Code 7 | Acquisition already running | `StopAcquisition()` or reboot headset |
| Flat signal | Electrode contact lost | Re-gel and re-seat |

---

## Team

Built at the **Br41n.IO Hackathon** by team **SOUDE**:

- [Aaron Emmanuel Hernández Rodriguez](https://github.com/AaronHero03)
- [Angel Landin Lopez](https://github.com/AngelLandin)
- [Victor Velázquez](https://github.com/Victor-123321)
- [Andrés Guzmán](https://github.com/andyys27)
- [Emiliano Montalvo](https://github.com/EmiMon-456)
- [Diego Huitron](https://github.com/Huitr0n)
- [Osvaldo Franco](https://github.com/Osva-Franco)
- [Emiliano Galván](https://github.com/Soy3miliano)
- [DiegoLizarraga](https://github.com/DiegoLizarraga)
- Kimberly Chihiro Camarillo Paredes

---

## License

Pendiente de definir por el equipo (aún no hay archivo `LICENSE` en el repo).
