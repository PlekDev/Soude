import os
import time
import numpy as np
import pandas as pd
from scipy import signal
from dotenv import load_dotenv

# Importamos el motor que ya construyó tu equipo
from brain_engine import BrainEngine

# 1. Cargar configuración desde el archivo .env
load_dotenv()
SERIAL_REAL = os.environ.get("UNICORN_SERIAL")

if not SERIAL_REAL:
    print("❌ ERROR: No se encontró UNICORN_SERIAL en el archivo .env")
    exit()

# 2. Configuración de procesamiento
FS = 250.0
# Filtro Passthought (8-30 Hz) para detectar ritmo Mu/Beta
sos_pt = signal.butter(4, [8.0, 30.0], btype='bandpass', fs=FS, output='sos')

def recolectar():
    # Inicializamos el motor con el serial del .env
    engine = BrainEngine(serial=SERIAL_REAL)
    dataset = []

    try:
        print(f"🔌 Conectando al Unicorn: {SERIAL_REAL}")
        engine.start()
        print("✅ Conexión establecida. El RingBuffer está activo.")

        # Definimos las etiquetas de entrenamiento
        for estado in ['Reposo', 'Passthought']:
            print(f"\n{'='*40}")
            print(f" FASE ACTUAL: {estado.upper()}")
            print(f"{'='*40}")
            print("Prepárate... la grabación inicia en 3 segundos.")
            time.sleep(3)

            print(f"🔴 Grabando 30 segundos de {estado}...")
            # Capturamos 15 ventanas de 2 segundos cada una
            for i in range(15):
                time.sleep(2)
                
                # Leemos los últimos 2 segundos del buffer centralizado
                # total_written es el puntero global de muestras que maneja el motor
                raw = engine.buffer.read_from(engine.buffer.total_written - 500, 500)
                
                if raw is not None:
                    # Aplicamos el filtro de segundo orden (SOS)
                    filtrados = signal.sosfilt(sos_pt, raw, axis=0)
                    
                    # Extraemos varianza de C3 (1) y C4 (3) según el mapa de brain_engine
                    e3 = np.var(filtrados[:, 1])
                    e4 = np.var(filtrados[:, 3])
                    
                    dataset.append([e3, e4, estado])
                    print(f"   [OK] Muestra {i+1}/15 | Var C3: {e3:.2f} | Var C4: {e4:.2f}")
                else:
                    print("   [!] Error: El buffer no tiene suficientes datos aún.")

    except Exception as e:
        print(f"❌ Error crítico: {e}")
    finally:
        print("\nDeteniendo adquisición...")
        engine.stop()
        
        if dataset:
            df = pd.DataFrame(dataset, columns=['Energia_C3', 'Energia_C4', 'Clase'])
            df.to_csv('dataset_real_fase2.csv', index=False)
            print(f"\n🏁 ¡LISTO! Archivo 'dataset_real_fase2.csv' generado con {len(df)} muestras.")

if __name__ == "__main__":
    recolectar()