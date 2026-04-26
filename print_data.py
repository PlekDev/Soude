import os
import time
import numpy as np
from brain_engine import BrainEngine
from dotenv import load_dotenv

load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")  # "LSL" -> network mode, "" -> mock

engine = BrainEngine(serial=SERIAL_REAL if SERIAL_REAL else None)
engine.start()

print("Iniciando monitor EEG... Presiona Ctrl+C para detener.")

try:
    while True:
        # 1. Tomamos una "foto" del estado actual del buffer completo
        datos_actuales = engine.buffer.snapshot()

        # 2. Tomamos solo la ULTIMA fila (la muestra mas reciente)
        # datos_actuales tiene forma (30000, 8)
        ultima_muestra = datos_actuales[-1]

        # 3. Formateamos la salida para que sea facil de leer (2 decimales)
        # Los canales son: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
        print(f"Fz: {ultima_muestra[0]:7.2f} uV | "
              f"C3: {ultima_muestra[1]:7.2f} uV | "
              f"Cz: {ultima_muestra[2]:7.2f} uV | "
              f"C4: {ultima_muestra[3]:7.2f} uV | "
              f"Pz: {ultima_muestra[4]:7.2f} uV | "
              f"PO7: {ultima_muestra[5]:7.2f} uV | "
              f"Oz: {ultima_muestra[6]:7.2f} uV | "
              f"PO8: {ultima_muestra[7]:7.2f} uV")

        # 4. Dormimos el hilo principal medio segundo
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nDeteniendo la adquisicion...")
    engine.stop()
