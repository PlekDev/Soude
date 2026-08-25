"""
tools/lsl_sender.py — emisor LSL para el modo de dos máquinas.

Reemplaza al antiguo emisor.py.  BrainEngine ya re-emite todo lo que adquiere
como stream LSL 'Unicorn_EEG', así que este script solo arranca el engine
(casco real o simulador) y lo mantiene vivo; la otra máquina lo recibe con
UNICORN_SERIAL=LSL o con tools/lsl_monitor.py --source lsl.

    python tools/lsl_sender.py            # casco real (UNICORN_SERIAL en .env)
    python tools/lsl_sender.py --mock     # simulador (para probar la cadena de red)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from neurolock.brain_engine import BrainEngine, MockUnicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Emisor LSL del EEG ('Unicorn_EEG').")
    parser.add_argument("--mock", action="store_true", help="usar el simulador en vez del casco")
    args = parser.parse_args()

    if args.mock:
        engine = BrainEngine(device=MockUnicorn())
    else:
        serial = os.environ.get("UNICORN_SERIAL", "").strip()
        if not serial or serial.upper() == "LSL":
            raise SystemExit("Define UNICORN_SERIAL en tu .env (o usa --mock).")
        engine = BrainEngine(serial=serial)

    engine.start()
    print("Emitiendo stream LSL 'Unicorn_EEG' — Ctrl+C para detener.")
    try:
        while True:
            time.sleep(1.0)
            engine.check_health()
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
