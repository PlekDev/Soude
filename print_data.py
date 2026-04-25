import os
import time
import numpy as np
from brain_engine import BrainEngine 
from dotenv import load_dotenv

load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")

engine = BrainEngine() # o usa MockUnicorn() si no tienes la diadema conectada
engine.start()

print("Iniciando monitor EEG... Presiona Ctrl+C para detener.")

try:
    while True:
        # 1. Tomamos una "foto" del estado actual del búfer completo
        datos_actuales = engine.buffer.snapshot()
        
        # 2. Tomamos solo la ÚLTIMA fila (la muestra más reciente)
        # Recuerda: datos_actuales tiene forma (30000, 8)
        ultima_muestra = datos_actuales[-1]
        
        # 3. Formateamos la salida para que sea fácil de leer (2 decimales)
        # Los canales son: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
        print(f"Fz: {ultima_muestra[0]:7.2f} µV | "
              f"C3: {ultima_muestra[1]:7.2f} µV | "
              f"Cz: {ultima_muestra[2]:7.2f} µV | "
              f"C4: {ultima_muestra[3]:7.2f} µV | "
              f"Pz: {ultima_muestra[4]:7.2f} µV | "
              f"P07: {ultima_muestra[5]:7.2f} µV | "
              f"0z: {ultima_muestra[6]:7.2f} µV  | "
              f"P08: {ultima_muestra[7]:7.2f} µV")
        
        # 4. Dormimos el hilo principal medio segundo (para no saturar la consola)
        time.sleep(0.5)

except KeyboardInterrupt:
    # Si presionas Ctrl+C en la terminal, cerramos el dispositivo limpiamente
    print("\nDeteniendo la adquisición...")
    engine.stop()