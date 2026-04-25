from pylsl import resolve_stream, StreamInlet
import time

print("Buscando transmisión del Unicorn en la red local...")

# 1. Busca cualquier stream que se llame 'Unicorn_EEG'
streams = resolve_stream('name', 'Unicorn_EEG')
inlet = StreamInlet(streams[0])

print("¡Conectado! Recibiendo datos desde la PC con licencia...")

try:
    while True:
        # 2. Recibe los datos a través del WiFi
        # chunk será una lista de muestras, timestamp es el tiempo sincronizado
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            for i, sample in enumerate(chunk):
                # Imprimir el canal Fz de la muestra recibida
                print(f"[{timestamps[i]:.4f}] Fz: {sample[0]:.2f} µV")
                
        time.sleep(0.01) # Pequeña pausa para no saturar el CPU

except KeyboardInterrupt:
    print("\nDesconectado del stream.")