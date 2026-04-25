"""
emisor_LSL.py — Transmisor continuo de Unicorn BCI a red local
Este script se conecta a la diadema Unicorn y empuja los datos 
directamente a un stream LSL de forma ininterrumpida.
"""

import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# --- 1. Configuración de la API de g.tec ---
# IMPORTANTE: Mantengo la ruta que usabas en tu archivo original
ruta_unicorn = r"C:\Users\joldo\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib"
sys.path.append(ruta_unicorn)

try:
    import UnicornPy # type: ignore
except ImportError:
    print("Error: No se pudo importar UnicornPy. Verifica la ruta.")
    sys.exit(1)

# --- 2. Constantes del Unicorn Hybrid Black ---
SAMPLE_RATE = 250
N_CHANNELS_EEG = 8
UNICORN_TOTAL_COLS = 17 # 8 EEG + Accel + Gyro + Batería + etc.
GETDATA_BLOCK = 4       # Muestras por lectura (baja latencia)

def main():
    # --- 3. Buscar y conectar la diadema ---
    dispositivos = UnicornPy.GetAvailableDevices(True)
    if not dispositivos:
        print("No se encontraron dispositivos Unicorn emparejados por Bluetooth.")
        return

    # Usamos el primer dispositivo que encuentre
    serial_device = dispositivos[0]
    print(f"Intentando conectar a: {serial_device}...")
    
    try:
        device = UnicornPy.Unicorn(serial_device)
        print("¡Conectado exitosamente!")
    except Exception as e:
        print(f"Error al conectar con la diadema: {e}")
        return

    # --- 4. Configurar LSL (El Megáfono) ---
    print("Configurando transmisión LSL...")
    info = StreamInfo(
        name='Unicorn_EEG', 
        type='EEG', 
        channel_count=N_CHANNELS_EEG, 
        nominal_srate=SAMPLE_RATE, 
        channel_format='float32', 
        source_id=serial_device
    )
    outlet = StreamOutlet(info)

    # Pre-asignar memoria para las lecturas (Optimización)
    # Tamaño = (4 muestras) * (17 columnas) * (4 bytes por float32)
    buffer_bytes = bytearray(GETDATA_BLOCK * UNICORN_TOTAL_COLS * 4)

    # --- 5. Bucle de Transmisión Continua ---
    print("\n" + "="*50)
    print(" TRANSMISIÓN INICIADA ".center(50, "="))
    print("="*50)
    print("Presiona Ctrl+C para detener de forma segura.")
    
    device.StartAcquisition(False)

    try:
        while True:
            # A. Extraer datos crudos de la diadema
            device.GetData(GETDATA_BLOCK, buffer_bytes, len(buffer_bytes))
            
            # B. Convertir los bytes a números (matriz NumPy)
            data_matrix = np.frombuffer(buffer_bytes, dtype=np.float32).reshape(GETDATA_BLOCK, UNICORN_TOTAL_COLS)
            
            # C. Recortar solo los primeros 8 canales (EEG puro)
            eeg_data = data_matrix[:, :N_CHANNELS_EEG]
            
            # D. Empujar por la red WiFi/LAN
            outlet.push_chunk(eeg_data.tolist())
            
    except KeyboardInterrupt:
        # Se ejecuta cuando presionas Ctrl+C
        print("\n\nDeteniendo transmisión por solicitud del usuario...")
    except Exception as e:
        print(f"\n\nError inesperado durante la lectura: {e}")
    finally:
        # --- 6. Apagado Seguro ---
        # Si no detienes la adquisición correctamente, el Bluetooth se queda 
        # "enganchado" y tendrás que reiniciar la diadema.
        print("Liberando dispositivo Unicorn...")
        device.StopAcquisition()
        del device
        print("Proceso finalizado correctamente. ¡Hasta luego!")

if __name__ == "__main__":
    main()