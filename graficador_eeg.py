import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import resolve_byprop, StreamInlet

# --- Configuración ---
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
FS = 250                # Frecuencia de muestreo (Hz)
WINDOW_SEC = 4          # Segundos de datos a mostrar en pantalla
BUFFER_SIZE = FS * WINDOW_SEC

def main():
    print("Buscando transmisión del Unicorn (LSL)...")
    streams = resolve_byprop('name', 'Unicorn_EEG', timeout=5)
    if not streams:
        print("Error: No se encontró el stream LSL. Verifica que el emisor esté corriendo.")
        sys.exit(1)
    
    inlet = StreamInlet(streams[0])
    print("¡Conectado! Abriendo interfaz gráfica...")

    # --- Configuración de la Ventana (PyQtGraph) ---
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True) # Suavizado de líneas
    win = pg.GraphicsLayoutWidget(show=True, title="Unicorn EEG - Osciloscopio")
    win.resize(1200, 800)
    win.setBackground('#111111') # Fondo oscuro

    plots = []
    curves = []
    
    # Búfer circular para los datos de los 8 canales
    data_buffer = np.zeros((8, BUFFER_SIZE))
    
    # Eje X en milisegundos (ej. de -4000 ms a 0 ms)
    time_buffer = np.linspace(-WINDOW_SEC * 1000, 0, BUFFER_SIZE)

    # Crear los 8 rieles (gráficas)
    for i, ch_name in enumerate(CHANNELS):
        p = win.addPlot(row=i, col=0)
        p.setLabel('left', ch_name, units='µV')
        
        # Ocultar los números del eje X en todos excepto en el último (abajo)
        if i < 7:
            p.getAxis('bottom').setStyle(showValues=False)
        else:
            p.setLabel('bottom', 'Tiempo', units='ms')
            
        p.showGrid(x=True, y=True, alpha=0.3)
        
        # Limitar el eje Y inicialmente a +/- 150 µV (Rango normal EEG)
        p.setYRange(-150, 150)
        
        # Sincronizar el eje X para que al hacer zoom en una, se haga en todas
        if i > 0:
            p.setXLink(plots[0])
            
        plots.append(p)
        
        # Crear la línea de la gráfica (Color cian)
        c = p.plot(pen=pg.mkPen(color=(50, 200, 255), width=1.5))
        curves.append(c)

    # --- Función de Actualización (Bucle Principal) ---
    def update():
        nonlocal data_buffer
        
        # Jalar todos los datos disponibles desde la última actualización
        chunk, timestamps = inlet.pull_chunk()
        
        if chunk:
            n_new = len(chunk)
            new_data = np.array(chunk).T # Transponer a forma (8, N_muestras_nuevas)
            
            # Recorrer el búfer circular (desplazar datos viejos y meter los nuevos)
            if n_new >= BUFFER_SIZE:
                data_buffer = new_data[:, -BUFFER_SIZE:]
            else:
                data_buffer[:, :-n_new] = data_buffer[:, n_new:]
                data_buffer[:, -n_new:] = new_data
            
            # IMPORTANTE: Eliminación del Offset (Centrar en 0 µV)
            # Calculamos la media de los últimos 4 segundos y se la restamos a la señal
            # Esto elimina el voltaje de +200,000 µV de la piel y deja solo la onda cerebral
            means = np.mean(data_buffer, axis=1, keepdims=True)
            centered_data = data_buffer - means
            
            # Dibujar los datos actualizados en las 8 gráficas
            for i in range(8):
                curves[i].setData(time_buffer, centered_data[i])

    # --- Temporizador para animar a ~30 FPS ---
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33) # 33 ms = ~30 cuadros por segundo

    # Iniciar la aplicación
    sys.exit(app.exec())

if __name__ == '__main__':
    main()