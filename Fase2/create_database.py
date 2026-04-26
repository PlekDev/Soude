import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from brain_engine import BrainEngine, CH_C3, CH_C4, SAMPLE_RATE
from filters import build_mu_beta_chain, apply_filter_chain

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")
if not SERIAL_REAL:
    print("❌ ERROR: UNICORN_SERIAL not found in .env file")
    sys.exit(1)

WINDOW_SECONDS      = 2
SAMPLES_PER_WINDOW  = int(WINDOW_SECONDS * SAMPLE_RATE)
NUM_WINDOWS         = 15
STATES              = ['Rest', 'Passthought']
ARTIFACT_THRESHOLD  = 150.0   # µV pico a pico — ajusta si rechaza demasiado

filter_chain = build_mu_beta_chain()

# ── Recolección ───────────────────────────────────────────────────────────────
def collect_data():
    engine  = BrainEngine(serial=SERIAL_REAL)
    dataset = []

    try:
        print(f"🔌 Connecting to Unicorn: {SERIAL_REAL}")
        engine.start()
        print("✅ Connection established. Warming up 1 second...")
        time.sleep(1.0)

        last_read_index = engine.buffer.total_written

        for state in STATES:
            print(f"\n{'='*40}")
            print(f" CURRENT PHASE: {state.upper()}")
            print(f"{'='*40}")
            print("Get ready... recording starts in 3 seconds.")
            time.sleep(3)

            # Descartar muestras del periodo de transición
            last_read_index = engine.buffer.total_written

            print(f"🔴 Recording {NUM_WINDOWS * WINDOW_SECONDS}s of {state}...")

            collected = 0   # contador de samples VÁLIDOS por estado
            i         = 0   # intentos totales (incluyendo rechazados)

            while collected < NUM_WINDOWS:
                target_index = last_read_index + SAMPLES_PER_WINDOW

                while engine.buffer.total_written < target_index:
                    time.sleep(0.005)

                raw = engine.buffer.read_from(last_read_index, SAMPLES_PER_WINDOW)
                last_read_index += SAMPLES_PER_WINDOW
                i += 1

                if raw is None:
                    print(f"   [!] Attempt {i:02d} — buffer overrun, skipped.")
                    continue

                filtered    = apply_filter_chain(raw, filter_chain)
                peak_to_peak = filtered.max(axis=0) - filtered.min(axis=0)

                if peak_to_peak.max() > ARTIFACT_THRESHOLD:
                    print(f"   [!] Attempt {i:02d} — artifact rejected "
                          f"({peak_to_peak.max():.1f} µV p-p > {ARTIFACT_THRESHOLD} µV)")
                    continue   # no guardar, pero sí seguir grabando

                e3 = np.var(filtered[:, CH_C3])
                e4 = np.var(filtered[:, CH_C4])
                dataset.append([e3, e4, state])
                collected += 1
                print(f"   [OK] {collected:02d}/{NUM_WINDOWS} | "
                      f"Var C3: {e3:7.2f} | Var C4: {e4:7.2f}")

    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print("\nStopping acquisition...")
        engine.stop()

    if dataset:
        df = pd.DataFrame(dataset, columns=['Variance_C3', 'Variance_C4', 'Class'])
        os.makedirs('Fase2', exist_ok=True)
        df.to_csv('dataset_phase2.csv', index=False)
        print(f"\n🏁 DONE! {len(df)} samples saved to 'Fase2/dataset_phase2.csv'")
        print("\nDistribución de clases:")
        print(df['Class'].value_counts().to_string())
    else:
        print("⚠️  No data collected.")

if __name__ == "__main__":
    collect_data()