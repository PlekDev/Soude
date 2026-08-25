"""
ml/test_model.py — inferencia en tiempo real del clasificador Rest/Passthought.

Espejo EXACTO del entrenamiento (antes no lo era y por eso nunca funcionó):
mismo filtro mu/beta (notch 60 Hz + bandpass 8-30 Hz), mismos canales C3/C4
de brain_engine, mismo tamaño de ventana, y aplica el scaler guardado junto
al modelo antes de predecir.

    python ml/test_model.py             # casco real (UNICORN_SERIAL en .env)
    python ml/test_model.py --mock      # simulador (para probar el flujo)
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from dotenv import load_dotenv

from neurolock.brain_engine import BrainEngine, MockUnicorn, CH_C3, CH_C4, SAMPLE_RATE
from neurolock.filters import build_mu_beta_chain, apply_filter_chain

load_dotenv()

WINDOW_SECONDS     = 2
SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * SAMPLE_RATE)
MODEL_PKL          = ROOT / "data" / "passthought_model.pkl"

filter_chain = build_mu_beta_chain()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia en vivo Rest/Passthought.")
    parser.add_argument("--mock", action="store_true", help="usar el simulador")
    args = parser.parse_args()

    if not MODEL_PKL.exists():
        raise SystemExit(f"No existe {MODEL_PKL}. Entrena primero con ml/train_model.py")
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]
    print("Modelo y scaler cargados.")

    if args.mock:
        engine = BrainEngine(device=MockUnicorn())
    else:
        serial = os.environ.get("UNICORN_SERIAL", "").strip()
        if not serial:
            raise SystemExit("Define UNICORN_SERIAL en tu .env (o usa --mock).")
        engine = BrainEngine(serial=serial)

    engine.start()
    print("Escuchando pensamientos... Ctrl+C para salir.\n")
    last_read_index = engine.buffer.total_written
    try:
        while True:
            target_index = last_read_index + SAMPLES_PER_WINDOW
            while engine.buffer.total_written < target_index:
                time.sleep(0.005)

            raw = engine.buffer.read_from(last_read_index, SAMPLES_PER_WINDOW)
            last_read_index += SAMPLES_PER_WINDOW
            if raw is None:
                # buffer sobrepasado: re-sincronizar al presente
                last_read_index = engine.buffer.total_written
                continue

            filtered = apply_filter_chain(raw, filter_chain)
            features = [[float(np.var(filtered[:, CH_C3])),
                         float(np.var(filtered[:, CH_C4]))]]
            decision = model.predict(scaler.transform(features))[0]

            if decision == "Passthought":
                print(f"[BOVEDA ABIERTA]  Passthought detectado | Var C3: {features[0][0]:8.2f}")
            else:
                print(f"[ACCESO DENEGADO] Usuario en reposo     | Var C3: {features[0][0]:8.2f}")
    except KeyboardInterrupt:
        print("\nPrueba finalizada.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
