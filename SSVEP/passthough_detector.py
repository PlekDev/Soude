import sys
import time
import numpy as np
from pylsl import resolve_byprop, StreamInlet

# --- CONFIGURACIÓN DEL CLASIFICADOR ---
FS = 250                     # Frecuencia de muestreo del Unicorn
TIEMPO_ANALISIS = 3.0        # Segundos de datos a capturar (3s = 750 muestras)
MUESTRAS_REQUERIDAS = int(FS * TIEMPO_ANALISIS)
FREQ_OBJETIVO = 15.0         # La frecuencia "contraseña"
TOLERANCIA_HZ = 0.5          # Buscar el pico entre 14.5 Hz y 15.5 Hz

# En el Unicorn, el lóbulo occipital (visión) está en: PO7 (índice 5), Oz (índice 6), PO8 (índice 7)
CANALES_VISUALES = [5, 6, 7] 

def calcular_fft_y_decidir(datos_eeg):
    """
    Toma una matriz de datos (muestras, canales), aplica FFT
    y decide si hay un pico significativo en 15 Hz.
    """
    # 1. Promediar los canales visuales para reducir ruido y crear una super-señal
    # datos_eeg tiene forma (750, 8). Tomamos solo 3 canales y los promediamos.
    senal_visual = np.mean(datos_eeg[:, CANALES_VISUALES], axis=1)
    
    # 2. Limpiar la señal: Restar la media (quitar el offset de DC)
    senal_limpia = senal_visual - np.mean(senal_visual)
    
    # Aplicar ventana de Hanning para suavizar los bordes antes de la FFT
    ventana = np.hanning(len(senal_limpia))
    senal_limpia = senal_limpia * ventana
    
    # 3. Calcular la Transformada de Fourier
    espectro = np.fft.rfft(senal_limpia)
    frecuencias = np.fft.rfftfreq(len(senal_limpia), d=1.0/FS)
    
    # Obtener el poder (Amplitud real)
    amplitud = np.abs(espectro)
    
    # 4. Analizar la zona de interés (15 Hz)
    idx_15hz = np.where((frecuencias >= FREQ_OBJETIVO - TOLERANCIA_HZ) & 
                        (frecuencias <= FREQ_OBJETIVO + TOLERANCIA_HZ))[0]
    
    # Analizar el ruido de fondo (promedio de las frecuencias alrededor de 13-17 Hz, ignorando los 15Hz)
    idx_ruido = np.where(((frecuencias >= 13.0) & (frecuencias <= 17.0)) & 
                         ~((frecuencias >= 14.5) & (frecuencias <= 15.5)))[0]
    
    poder_objetivo = np.max(amplitud[idx_15hz])
    poder_ruido_fondo = np.mean(amplitud[idx_ruido])
    
    # Calcular el Signal-to-Noise Ratio (SNR)
    # ¿Es el pico de 15 Hz mucho más grande que el ruido que lo rodea?
    snr = poder_objetivo / poder_ruido_fondo
    
    print(f"\n--- ANÁLISIS COMPLETADO ---")
    print(f"Ruido de fondo: {poder_ruido_fondo:.2f}")
    print(f"Poder en 15 Hz: {poder_objetivo:.2f}")
    print(f"Fuerza de la señal (SNR): {snr:.2f}x")
    
    # 5. EL UMBRAL DE DECISIÓN (Tendrás que calibrar este número)
    # Si la señal en 15Hz es 3 veces más fuerte que el ruido de fondo, pasamos.
    UMBRAL = 2.0 
    
    if snr >= UMBRAL:
        print("\n🟢 ¡ACCESO CONCEDIDO! (Firma SSVEP de 15 Hz detectada) 🟢")
    else:
        print("\n🔴 ACCESO DENEGADO (Firma no reconocida) 🔴")

def main():
    print("Buscando streams en la red local...")
    
    # Conectar a ambos streams
    streams_eeg = resolve_byprop('name', 'Unicorn_EEG', timeout=5)
    streams_mrk = resolve_byprop('name', 'SSVEP_Marcadores', timeout=5)
    
    if not streams_eeg or not streams_mrk:
        print("Error: No se encontraron ambos streams (EEG y Marcadores).")
        sys.exit(1)
        
    inlet_eeg = StreamInlet(streams_eeg[0])
    inlet_mrk = StreamInlet(streams_mrk[0])
    
    print("Conectado exitosamente al Unicorn y al Generador de Estímulos.")
    
    while True:
        print("\nEsperando marcador de estímulo visual...")
        
        # 1. Bloquear hasta que llegue un marcador por la red
        marcador, timestamp_mrk = inlet_mrk.pull_sample()
        
        if marcador and marcador[0] == 'INICIO_15HZ':
            print("¡Estímulo detectado! Capturando ondas cerebrales...")
            
            # Limpiar cualquier dato viejo atascado en el búfer
            inlet_eeg.flush()
            
            datos_epoca = []
            muestras_recolectadas = 0
            
            # 2. Recolectar exactamente 3 segundos de datos (750 muestras)
            while muestras_recolectadas < MUESTRAS_REQUERIDAS:
                chunk, timestamps = inlet_eeg.pull_chunk()
                if chunk:
                    datos_epoca.extend(chunk)
                    muestras_recolectadas += len(chunk)
                time.sleep(0.01) # Pequeña pausa
                
            # Recortar exactamente a 750 muestras por si entraron de más
            matriz_datos = np.array(datos_epoca)[:MUESTRAS_REQUERIDAS]
            
            # 3. Analizar matemáticamente
            calcular_fft_y_decidir(matriz_datos)
            
            # Pequeño enfriamiento antes de buscar otra contraseña
            time.sleep(2)

if __name__ == '__main__':
    main()