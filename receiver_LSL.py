from pylsl import resolve_byprop, StreamInlet
import time

print("Buscando transmisión del Unicorn en la red local...")

# 1. Busca cualquier stream que tenga la propiedad 'name' igual a 'Unicorn_EEG'
streams = resolve_byprop('name', 'Unicorn_EEG')
inlet = StreamInlet(streams[0])

print("¡Conectado! Recibiendo datos desde la PC con licencia...")

try:
    while True:
        # 2. Recibe los datos a través del WiFi/LAN
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            for i, sample in enumerate(chunk):
                # Imprimir el canal Fz (índice 0) de la muestra recibida
                print(f"[{timestamps[i]:.4f}] Fz: {sample[0]:.2f} µV")
                
        time.sleep(0.01) # Pequeña pausa para no saturar el CPU

except KeyboardInterrupt:
    print("\nDesconectado del stream.")