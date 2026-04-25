import os
import sys

# 1. Agregamos la ruta principal del proyecto
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Importaciones limpias utilizando tus constantes y funciones globales
from brain_engine import BrainEngine, CH_C3, CH_C4, SAMPLE_RATE
from filters import build_mu_beta_chain, apply_filter_chain

# 1. Load configuration from .env file
load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")

if not SERIAL_REAL:
    print("❌ ERROR: UNICORN_SERIAL not found in .env file")
    sys.exit(1)

# 2. Processing Configuration
WINDOW_SECONDS = 2
SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * SAMPLE_RATE)
NUM_WINDOWS = 15
STATES = ['Rest', 'Passthought']

# Usamos la cadena de filtros completa: Notch (60Hz) + Passthought (8-30 Hz)
filter_chain = build_mu_beta_chain()

def collect_data():
    engine = BrainEngine(serial=SERIAL_REAL)
    dataset = []

    try:
        print(f"🔌 Connecting to Unicorn: {SERIAL_REAL}")
        engine.start()
        print("✅ Connection established. RingBuffer is active.")

        for state in STATES:
            print(f"\n{'='*40}")
            print(f" CURRENT PHASE: {state.upper()}")
            print(f"{'='*40}")
            print("Get ready... recording starts in 3 seconds.")
            time.sleep(3)

            print(f"🔴 Recording {NUM_WINDOWS * WINDOW_SECONDS} seconds of {state}...")
            
            # Anclamos el puntero al índice actual del buffer
            last_read_index = engine.buffer.total_written
            
            for i in range(NUM_WINDOWS):
                # Esperamos de forma inteligente hasta que el buffer tenga los 500 samples exactos
                target_index = last_read_index + SAMPLES_PER_WINDOW
                while engine.buffer.total_written < target_index:
                    time.sleep(0.05) # Pausa corta para no saturar el CPU
                    
                # Leemos la ventana de datos exacta sin perder ni repetir frames
                raw = engine.buffer.read_from(last_read_index, SAMPLES_PER_WINDOW)
                last_read_index += SAMPLES_PER_WINDOW
                
                if raw is not None:
                    # Aplicar la cadena de filtros (Notch -> Mu/Beta)
                    filtered_data = apply_filter_chain(raw, filter_chain)
                    
                    # Extraer varianza usando las constantes legibles (CH_C3, CH_C4)
                    e3 = np.var(filtered_data[:, CH_C3])
                    e4 = np.var(filtered_data[:, CH_C4])
                    
                    dataset.append([e3, e4, state])
                    print(f"   [OK] Sample {i+1:02d}/{NUM_WINDOWS} | Var C3: {e3:7.2f} | Var C4: {e4:7.2f}")
                else:
                    print(f"   [!] Error: Buffer overrun/underrun on sample {i+1}.")

    except Exception as e:
        print(f"❌ Critical error: {e}")
    finally:
        print("\nStopping acquisition...")
        engine.stop()
        
        if dataset:
            df = pd.DataFrame(dataset, columns=['Variance_C3', 'Variance_C4', 'Class'])
            
            # Aseguramos que el directorio Fase2 exista por si acaso
            os.makedirs('Fase2', exist_ok=True)
            df.to_csv('Fase2/dataset_phase2.csv', index=False)
            print(f"\n🏁 DONE! 'dataset_phase2.csv' generated with {len(df)} samples.")

if __name__ == "__main__":
    collect_data()