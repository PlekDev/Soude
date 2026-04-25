from brain_engine import BrainEngine
import pandas as pd
import time
import numpy as np
from scipy import signal

# 1. Configuración del hardware (Usa el serial de tu Unicorn)
SERIAL_UNICORN = "UN-2023.05.27" # <--- CAMBIA ESTO
engine = BrainEngine(serial=SERIAL_UNICORN)
engine.start()

fs = 250.0
sos = signal.butter(4, [8.0, 30.0], btype='bandpass', fs=fs, output='sos')
dataset = []

try:
    print("\n--- INICIANDO CAPTURA REAL ---")
    for estado in ['Reposo', 'Passthought']:
        print(f"\n>>> PREPÁRATE PARA: {estado} <<<")
        print("Tienes 5 segundos para concentrarte...")
        time.sleep(5)
        
        print(f"🔴 GRABANDO {estado} (20 segundos)...")
        for i in range(10): # Capturaremos 10 bloques de 2 segundos
            time.sleep(2)
            # Pedimos los últimos 2 segundos al buffer de tu amigo
            data = engine.buffer.read_from(engine.buffer.total_written - 500, 500)
            
            if data is not None:
                # Filtramos la señal
                filtrados = signal.sosfilt(sos, data, axis=0)
                # IMPORTANTE: Usamos índices 1 (C3) y 3 (C4) del brain_engine
                e3 = np.var(filtrados[:, 1])
                e4 = np.var(filtrados[:, 3])
                
                dataset.append([e3, e4, estado])
                print(f" Muestra {i+1}/10 guardada (Energía C3: {e3:.1f})")

finally:
    engine.stop()
    # Guardamos el archivo CSV real
    df = pd.DataFrame(dataset, columns=['Energia_C3', 'Energia_C4', 'Clase'])
    df.to_csv('dataset_real.csv', index=False)
    print("\n✅ Dataset real guardado como 'dataset_real.csv'")