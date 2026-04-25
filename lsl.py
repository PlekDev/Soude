import UnicornPy
import pylsl
import time

# 1. Configurar LSL
info = pylsl.StreamInfo('Unicorn_Raw', 'EEG', 8, 250, 'float32', 'un-123')
outlet = pylsl.StreamOutlet(info)

# 2. Conectar al Casco
device = UnicornPy.Unicorn("UN-2023.05.27") # <--- QUE PONGA SU SERIAL AQUÍ
device.StartAcquisition(False)
print("📡 Transmitiendo datos del Unicorn a la red LSL...")

try:
    while True:
        # Leer 1 muestra (8 canales)
        data = device.GetData(1) 
        # Enviar al Wi-Fi
        outlet.push_sample(data[0]) 
except KeyboardInterrupt:
    device.StopAcquisition()
    print("Detenido.")