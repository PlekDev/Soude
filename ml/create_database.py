"""
ml/create_database.py — recolección de datos para el clasificador Rest/Passthought.

Graba ventanas de 2 s con el Unicorn real (UNICORN_SERIAL en .env), filtra a la
banda mu/beta (8-30 Hz + notch 60 Hz), rechaza artefactos y guarda la varianza
de C3 y C4 como features etiquetadas en data/dataset_phase2.csv.

    python ml/create_database.py            # casco real
    python ml/create_database.py --mock     # simulador (solo para probar el flujo)
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from neurolock.brain_engine import BrainEngine, MockUnicorn, CH_C3, CH_C4, SAMPLE_RATE
from neurolock.filters import build_mu_beta_chain, apply_filter_chain

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
WINDOW_SECONDS      = 2
SAMPLES_PER_WINDOW  = int(WINDOW_SECONDS * SAMPLE_RATE)
NUM_WINDOWS         = 15
MAX_ATTEMPTS        = NUM_WINDOWS * 5   # tope: si todo es artefacto, abortar con claridad
STATES              = ["Rest", "Passthought"]
ARTIFACT_THRESHOLD  = 150.0   # µV pico a pico — ajusta si rechaza demasiado

DATA_DIR   = ROOT / "data"
OUTPUT_CSV = DATA_DIR / "dataset_phase2.csv"

filter_chain = build_mu_beta_chain()


def collect_data(use_mock: bool = False) -> None:
    if use_mock:
        print("[AVISO] Modo simulador: estos datos NO sirven para entrenar de verdad.")
        engine = BrainEngine(device=MockUnicorn())
    else:
        serial = os.environ.get("UNICORN_SERIAL", "").strip()
        if not serial:
            raise SystemExit("ERROR: define UNICORN_SERIAL en tu .env (o usa --mock).")
        print(f"Conectando al Unicorn: {serial}")
        engine = BrainEngine(serial=serial)

    dataset = []
    try:
        engine.start()
        print("Conexion establecida. Calentando 1 segundo...")
        time.sleep(1.0)

        for state in STATES:
            print(f"\n{'=' * 40}")
            print(f" FASE ACTUAL: {state.upper()}")
            print(f"{'=' * 40}")
            print("Preparate... la grabacion empieza en 3 segundos.")
            time.sleep(3)

            # Descartar las muestras del periodo de transición
            last_read_index = engine.buffer.total_written
            print(f"Grabando ~{NUM_WINDOWS * WINDOW_SECONDS}s de {state}...")

            collected = 0   # ventanas VALIDAS
            attempts  = 0   # intentos totales (incluyendo rechazadas)
            while collected < NUM_WINDOWS:
                if attempts >= MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"{attempts} intentos y solo {collected} ventanas limpias de "
                        f"'{state}': revisa el gel y la impedancia de los electrodos."
                    )

                target_index = last_read_index + SAMPLES_PER_WINDOW
                while engine.buffer.total_written < target_index:
                    time.sleep(0.005)

                raw = engine.buffer.read_from(last_read_index, SAMPLES_PER_WINDOW)
                last_read_index += SAMPLES_PER_WINDOW
                attempts += 1

                if raw is None:
                    print(f"   [!] Intento {attempts:02d} — buffer sobrepasado, se omite.")
                    continue

                filtered     = apply_filter_chain(raw, filter_chain)
                peak_to_peak = filtered.max(axis=0) - filtered.min(axis=0)

                if peak_to_peak.max() > ARTIFACT_THRESHOLD:
                    print(f"   [!] Intento {attempts:02d} — artefacto rechazado "
                          f"({peak_to_peak.max():.1f} uV p-p > {ARTIFACT_THRESHOLD} uV)")
                    continue

                e3 = float(np.var(filtered[:, CH_C3]))
                e4 = float(np.var(filtered[:, CH_C4]))
                dataset.append([e3, e4, state])
                collected += 1
                print(f"   [OK] {collected:02d}/{NUM_WINDOWS} | "
                      f"Var C3: {e3:7.2f} | Var C4: {e4:7.2f}")

    except Exception:
        print("ERROR critico durante la recoleccion:")
        traceback.print_exc()
    finally:
        print("\nDeteniendo adquisicion...")
        engine.stop()

    if dataset:
        DATA_DIR.mkdir(exist_ok=True)
        df = pd.DataFrame(dataset, columns=["Variance_C3", "Variance_C4", "Class"])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nLISTO: {len(df)} muestras guardadas en {OUTPUT_CSV}")
        print("\nDistribucion de clases:")
        print(df["Class"].value_counts().to_string())
    else:
        print("AVISO: no se recolectaron datos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolector de dataset Rest/Passthought.")
    parser.add_argument("--mock", action="store_true",
                        help="usar el simulador (solo para probar el flujo)")
    args = parser.parse_args()
    collect_data(use_mock=args.mock)
