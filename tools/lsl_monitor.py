"""
tools/lsl_monitor.py — monitor de consola de los 8 canales EEG.

Reemplaza a los antiguos receiver_LSL.py y print_data.py:

    python tools/lsl_monitor.py --source mock     # simulador local (default)
    python tools/lsl_monitor.py --source lsl      # recibe 'Unicorn_EEG' por red
    python tools/lsl_monitor.py --source real     # casco Unicorn (UNICORN_SERIAL en .env)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from brain_engine import BrainEngine, MockUnicorn, CHANNEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor de consola del EEG (8 canales).")
    parser.add_argument(
        "--source", choices=["mock", "lsl", "real"], default="mock",
        help="mock: simulador | lsl: stream 'Unicorn_EEG' de la red | real: casco (UNICORN_SERIAL)",
    )
    parser.add_argument("--interval", type=float, default=0.5, help="segundos entre lecturas")
    args = parser.parse_args()

    if args.source == "mock":
        engine = BrainEngine(device=MockUnicorn())
    elif args.source == "lsl":
        engine = BrainEngine(serial="LSL")
    else:
        serial = os.environ.get("UNICORN_SERIAL", "").strip()
        if not serial or serial.upper() == "LSL":
            raise SystemExit("Define UNICORN_SERIAL en tu .env para usar --source real.")
        engine = BrainEngine(serial=serial)

    engine.start()
    print(f"Fuente: {args.source} — Ctrl+C para salir.")
    try:
        while True:
            engine.check_health()
            snap = engine.buffer.snapshot()
            last = snap[-1]
            print(" | ".join(f"{name}: {v:8.2f}" for name, v in zip(CHANNEL_NAMES, last)))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
