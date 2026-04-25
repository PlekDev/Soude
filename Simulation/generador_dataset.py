import pylsl
import numpy as np
from scipy import signal
import pandas as pd
import time

# 1. Configuración del Filtro y Variables
fs = 250.0
sos = signal.butter(4, [8.0, 30.0], btype='bandpass', fs=fs, output='sos')
window_size = int(fs * 2) # 2 segundos
dataset_features = [] # Aquí guardaremos "Los Puntos"

print("Buscando el Cerebro Sintético en la red...")
streams = pylsl.resolve_byprop('name', 'Unicorn')
inlet = pylsl.StreamInlet(streams[0])
print("✅ Conectado al simulador.")

# 2. Instrucciones para recolectar datos
print("\n--- INICIANDO RECOLECCIÓN DE DATOS ---")
print("El simulador alterna cada 5 segundos. Acumularemos 40 ejemplos en total.")

muestras_recolectadas = 0
buffer = []

while muestras_recolectadas < 40:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample)

    if len(buffer) == window_size:
        # Convertir a matriz y filtrar
        datos_crudos = np.array(buffer)
        datos_filtrados = signal.sosfilt(sos, datos_crudos, axis=0)

        # ==========================================
        # EL PASO MÁGICO: EXTRACCIÓN DE CARACTERÍSTICAS
        # ==========================================
        # Extraemos la energía (varianza) de C3 (índice 2) y C4 (índice 4)
        energia_c3 = np.var(datos_filtrados[:, 2])
        energia_c4 = np.var(datos_filtrados[:, 4])
        
        # Etiquetamos el dato automáticamente basado en la energía de C3 para este experimento
        # (En la vida real, tendrías un programa pidiéndole al usuario que imagine o descanse)
        etiqueta = "Passthought" if energia_c3 < 150 else "Reposo"

        # Guardamos "El Punto" (Energía C3, Energía C4, y a qué clase pertenece)
        dataset_features.append([energia_c3, energia_c4, etiqueta])
        
        muestras_recolectadas += 1
        print(f"Ejemplo {muestras_recolectadas}/40 guardado -> Clase: {etiqueta}")
        
        buffer = [] # Limpiar el buffer para los siguientes 2 segundos

# 3. Guardar el Dataset en un archivo CSV
df = pd.DataFrame(dataset_features, columns=['Energia_C3', 'Energia_C4', 'Clase'])
df.to_csv('dataset_entrenamiento.csv', index=False)
print("\n✅ ¡Dataset guardado como 'dataset_entrenamiento.csv'!")