from pylsl import resolve_byprop, StreamInlet
import time

print("Buscando transmisión del Unicorn en la red local...")

# 1. Busca cualquier stream que tenga la propiedad 'name' igual a 'Unicorn_EEG'
streams = resolve_byprop('name', 'Unicorn_EEG')
inlet = StreamInlet(streams[0])

print("¡Conectado! Recibiendo los 8 canales desde la PC con licencia...")

try:
    while True:
        # 2. Recibe los datos a través del WiFi/LAN
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            for i, sample in enumerate(chunk):
                # 'sample' contiene los 8 valores en este orden exacto:
                fz, c3, cz, c4, pz, po7, oz, po8 = sample
                
                # Imprimir los 8 canales alineados
                # Usamos 7.2f para que ocupen un espacio fijo y la consola no brinque
                print(f"[{timestamps[i]:.3f}] "
                      f"Fz:{fz:7.2f} | C3:{c3:7.2f} | Cz:{cz:7.2f} | C4:{c4:7.2f} | "
                      f"Pz:{pz:7.2f} | PO7:{po7:7.2f} | Oz:{oz:7.2f} | PO8:{po8:7.2f}")
                
        time.sleep(0.05) # Pequeña pausa para no saturar el CPU

except KeyboardInterrupt:
    print("\nDesconectado del stream.")