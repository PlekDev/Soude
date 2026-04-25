import pickle
import pylsl
import numpy as np
from scipy import signal

# ==========================================
# 1. DESPERTAR A LA IA
# ==========================================
print("Cargando el cerebro artificial (.pkl)...")
with open('modelo_passthought.pkl', 'rb') as archivo:
    modelo_svm = pickle.load(archivo)
print("✅ IA lista para tomar decisiones.\n")

# Configuramos el filtro de siempre
fs = 250.0
sos = signal.butter(4, [8.0, 30.0], btype='bandpass', fs=fs, output='sos')

# ==========================================
# 2. CONECTAR AL CASCO SIMULADO
# ==========================================
print("Buscando transmisión cerebral en la red...")
streams = pylsl.resolve_byprop('name', 'Unicorn')
inlet = pylsl.StreamInlet(streams[0])
print("✅ Conectado. Escuchando pensamientos...\n")

# ==========================================
# 3. EL BUCLE DE VIGILANCIA
# ==========================================
buffer = []

try:
    while True:
        sample, timestamp = inlet.pull_sample()
        buffer.append(sample)

        # Cada 2 segundos (500 muestras), evaluamos
        if len(buffer) == int(fs * 2):
            datos_crudos = np.array(buffer)
            
            # Filtramos
            datos_filtrados = signal.sosfilt(sos, datos_crudos, axis=0)

            # Extraemos la energía de C3 y C4
            energia_c3 = np.var(datos_filtrados[:, 2])
            energia_c4 = np.var(datos_filtrados[:, 4])

            # 🧠 LA IA TOMA LA DECISIÓN AQUÍ
            # Le pasamos los dos números nuevos y nos devuelve su veredicto
            caracteristicas = [[energia_c3, energia_c4]]
            decision = modelo_svm.predict(caracteristicas)[0]

            # Mostramos el resultado en pantalla
            if decision == "Passthought":
                print(f"🔓 BÓVEDA ABIERTA   | IA detectó ERD     (Energía: {energia_c3:.1f})")
            else:
                print(f"🔒 ACCESO DENEGADO  | Usuario en Reposo  (Energía: {energia_c3:.1f})")

            # Vaciamos el buffer para los siguientes 2 segundos
            buffer = []
            
except KeyboardInterrupt:
    print("\nPrueba finalizada.")