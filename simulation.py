# SCRIPT PARA SIMULAR EL UNICORN (Para que los programadores avancen)
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# Crear info de stream falso (8 canales, 250Hz)
info = StreamInfo('Unicorn', 'EEG', 8, 250, 'float32', 'sim_id_123')
outlet = StreamOutlet(info)

print("Simulando Unicorn... Los programadores ya pueden conectarse al stream.")

while True:
    # Crear datos aleatorios que parecen EEG
    sample = np.random.uniform(-50, 50, 8).tolist()
    outlet.push_sample(sample)
    time.sleep(1/250.0)