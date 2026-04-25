"""
realtime_viewer.py — Visor de EEG en Tiempo Real
Conecta con el Unicorn Hybrid Black y grafica los 8 canales usando PyQtGraph.
"""

import os
import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
from dotenv import load_dotenv

# Importar el motor y los filtros del proyecto
from brain_engine import BrainEngine, SAMPLE_RATE, N_CHANNELS
from filters import build_p300_chain, apply_filter_chain

class EEGVisualizer(QMainWindow):
    def __init__(self, serial_device: str):
        super().__init__()
        self.setWindowTitle(f"Neuro-Lock: Visor EEG en Vivo ({serial_device})")
        self.resize(1000, 700)

        # 1. Configurar el Motor BCI
        self.engine = BrainEngine(serial=serial_device)
        self.filter_chain = build_p300_chain() # Notch 60Hz + Bandpass 1-10Hz para estabilizar la vista
        
        # Parámetros de visualización
        self.window_seconds = 4  # Cuántos segundos de historial mostrar en pantalla
        self.display_samples = int(self.window_seconds * SAMPLE_RATE)
        self.channel_offset = 150 # Espaciado vertical (µV) entre cada canal para que no se encimen

        # 2. Construir la Interfaz de Usuario
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Barra superior de controles
        top_bar = QHBoxLayout()
        self.lbl_status = QLabel("ESTADO: Conectando...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #ffaa00;")
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        
        self.btn_toggle_filter = QPushButton("Alternar Filtro (Activo)")
        self.btn_toggle_filter.setCheckable(True)
        self.btn_toggle_filter.setChecked(True)
        top_bar.addWidget(self.btn_toggle_filter)
        layout.addLayout(top_bar)

        # 3. Configurar PyQtGraph
        pg.setConfigOptions(antialias=True) # Gráficos más suaves
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#060a10')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Tiempo (Segundos)')
        self.plot_widget.setLabel('left', 'Amplitud de Canales')
        
        # Ocultar los números del eje Y porque usaremos canales apilados
        self.plot_widget.getAxis('left').setTicks([]) 
        layout.addWidget(self.plot_widget)

        # Crear una curva (línea) por cada canal
        self.curves = []
        channel_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        colors = ['#ff1744', '#00e5ff', '#00e676', '#ffea00', '#d500f9', '#ff9100', '#00b0ff', '#1de9b6']

        for i in range(N_CHANNELS):
            curve = self.plot_widget.plot(pen=pg.mkPen(color=colors[i], width=1.5), name=channel_names[i])
            self.curves.append(curve)
            
            # Etiqueta de texto para el nombre del canal
            text = pg.TextItem(channel_names[i], color=colors[i], anchor=(0, 0.5))
            text.setPos(0, i * self.channel_offset)
            self.plot_widget.addItem(text)

        # 4. Timer de Actualización (Render Loop)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Iniciar el dispositivo
        self.start_device()

    def start_device(self):
        try:
            self.engine.start()
            self.lbl_status.setText("ESTADO: ADQUIRIENDO DATOS")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676;")
            # Actualizar la pantalla a ~30 FPS (cada 33 ms)
            self.timer.start(33)
        except Exception as e:
            self.lbl_status.setText(f"ERROR: {e}")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #ff1744;")

    def update_plot(self):
        # 1. Obtener una copia de todo el buffer cronológico
        snapshot = self.engine.buffer.snapshot()
        
        # 2. Si aún no hay suficientes datos para llenar la pantalla, rellenamos con ceros
        if self.engine.buffer.total_written < self.display_samples:
            return # Esperar un poco a que se llene el inicio del buffer
            
        # 3. Recortar solo los últimos X segundos para mostrarlos
        view_data = snapshot[-self.display_samples:, :]
        
        # Eje X en segundos
        time_axis = np.linspace(0, self.window_seconds, self.display_samples)

        # 4. Filtrar (Opcional mediante el botón)
        if self.btn_toggle_filter.isChecked():
            view_data = apply_filter_chain(view_data, self.filter_chain)

        # 5. Dibujar cada canal
        for ch in range(N_CHANNELS):
            y_data = view_data[:, ch]
            
            # Centrar la señal en cero (quitar el offset de corriente continua)
            y_centered = y_data - np.mean(y_data)
            
            # Apilar los canales sumándoles un offset fijo (0, 150, 300, 450...)
            y_stacked = y_centered + (ch * self.channel_offset)
            
            # Actualizar la línea en el gráfico
            self.curves[ch].setData(time_axis, y_stacked)

    def closeEvent(self, event):
        """Asegurarse de apagar el Unicorn correctamente al cerrar la ventana."""
        self.timer.stop()
        self.engine.stop()
        event.accept()

if __name__ == '__main__':
    # Cargar serial desde el .env
    load_dotenv()
    SERIAL = os.environ.get("UNICORN_SERIAL")
    
    if not SERIAL:
        print("❌ ERROR: No se encontró UNICORN_SERIAL en el archivo .env")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = EEGVisualizer(SERIAL)
    window.show()
    sys.exit(app.exec())
    """
realtime_viewer.py — Visor de EEG en Tiempo Real
Conecta con el Unicorn Hybrid Black y grafica los 8 canales en carriles separados.
"""

import os
import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QHBoxLayout, QLabel, QSpinBox)
from PyQt6.QtCore import QTimer, Qt
from dotenv import load_dotenv

# Importar el motor y los filtros del proyecto
from brain_engine import BrainEngine, SAMPLE_RATE, N_CHANNELS
from filters import build_p300_chain, apply_filter_chain

class EEGVisualizer(QMainWindow):
    def __init__(self, serial_device: str):
        super().__init__()
        self.setWindowTitle(f"Neuro-Lock: Visor EEG en Vivo ({serial_device})")
        self.resize(1200, 800)

        # 1. Configurar el Motor BCI
        self.engine = BrainEngine(serial=serial_device)
        self.filter_chain = build_p300_chain()
        
        # Parámetros de visualización
        self.window_seconds = 4
        self.display_samples = int(self.window_seconds * SAMPLE_RATE)

        # 2. Construir la Interfaz de Usuario
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Barra superior de controles ---
        top_bar = QHBoxLayout()
        
        self.lbl_status = QLabel("ESTADO: Conectando...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #ffaa00; font-size: 14px;")
        top_bar.addWidget(self.lbl_status)
        
        top_bar.addStretch()

        # Control de Zoom Y (Amplitud)
        lbl_zoom = QLabel("Zoom Amplitud Y:")
        top_bar.addWidget(lbl_zoom)
        
        self.spin_zoom = QSpinBox()
        self.spin_zoom.setRange(10, 500)
        self.spin_zoom.setValue(50) # Inicia en +- 50 µV
        self.spin_zoom.setPrefix("± ")
        self.spin_zoom.setSuffix(" µV")
        self.spin_zoom.valueChanged.connect(self.update_zoom)
        top_bar.addWidget(self.spin_zoom)

        # Botón de Filtro
        self.btn_toggle_filter = QPushButton("Filtro: ACTIVADO")
        self.btn_toggle_filter.setCheckable(True)
        self.btn_toggle_filter.setChecked(True)
        self.btn_toggle_filter.toggled.connect(lambda c: self.btn_toggle_filter.setText("Filtro: ACTIVADO" if c else "Filtro: DESACTIVADO"))
        top_bar.addWidget(self.btn_toggle_filter)

        # Botón de Pausa
        self.btn_pause = QPushButton("Pausar Gráfica")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self.toggle_pause)
        top_bar.addWidget(self.btn_pause)

        layout.addLayout(top_bar)

        # 3. Configurar PyQtGraph (Graphics Layout para Carriles Separados)
        pg.setConfigOptions(antialias=True)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#060a10')
        layout.addWidget(self.graphics_layout)

        self.plots = []
        self.curves = []
        channel_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        colors = ['#ff1744', '#00e5ff', '#00e676', '#ffea00', '#d500f9', '#ff9100', '#00b0ff', '#1de9b6']

        # Crear un carril (plot) por cada canal
        for i in range(N_CHANNELS):
            p = self.graphics_layout.addPlot(row=i, col=0)
            p.setYRange(-50, 50) # Escala inicial +- 50
            p.showGrid(x=True, y=True, alpha=0.3)
            
            # Etiqueta de la izquierda con el nombre del canal y color
            p.setLabel('left', channel_names[i], color=colors[i])
            
            # Ocultar el eje X para todos menos el de hasta abajo (para ahorrar espacio visual)
            if i < N_CHANNELS - 1:
                p.hideAxis('bottom')
            else:
                p.setLabel('bottom', 'Tiempo (Segundos)')

            # Sincronizar el eje X con el primer canal. Si haces zoom en el tiempo en un canal, afecta a todos.
            if i > 0:
                p.setXLink(self.plots[0])

            # Crear la línea de datos
            curve = p.plot(pen=pg.mkPen(color=colors[i], width=1.5))
            
            self.plots.append(p)
            self.curves.append(curve)

        # 4. Timer de Actualización
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Iniciar dispositivo
        self.start_device()

    def start_device(self):
        try:
            self.engine.start()
            self.lbl_status.setText("ESTADO: ADQUIRIENDO DATOS")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 14px;")
            self.timer.start(33) # ~30 FPS
        except Exception as e:
            self.lbl_status.setText(f"ERROR: {e}")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 14px;")

    def toggle_pause(self, checked):
        """Maneja el comportamiento del botón de pausa."""
        if checked:
            self.btn_pause.setText("Reanudar Gráfica")
            self.lbl_status.setText("ESTADO: PAUSADO (Adquisición en 2do plano)")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #ffea00; font-size: 14px;")
        else:
            self.btn_pause.setText("Pausar Gráfica")
            self.lbl_status.setText("ESTADO: ADQUIRIENDO DATOS")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 14px;")

    def update_zoom(self, value):
        """Ajusta el rango del eje Y para todos los canales."""
        for p in self.plots:
            p.setYRange(-value, value)

    def update_plot(self):
        # Si está pausado, saltamos la actualización visual (la imagen se congela)
        if self.btn_pause.isChecked():
            return

        snapshot = self.engine.buffer.snapshot()
        
        if self.engine.buffer.total_written < self.display_samples:
            return 
            
        view_data = snapshot[-self.display_samples:, :]
        time_axis = np.linspace(0, self.window_seconds, self.display_samples)

        if self.btn_toggle_filter.isChecked():
            view_data = apply_filter_chain(view_data, self.filter_chain)

        # Dibujar cada canal en su respectivo carril
        for ch in range(N_CHANNELS):
            y_data = view_data[:, ch]
            
            # Centrar la señal en cero restando la media de ~150µV
            y_centered = y_data - np.mean(y_data)
            
            # Actualizar la línea
            self.curves[ch].setData(time_axis, y_centered)

    def closeEvent(self, event):
        self.timer.stop()
        self.engine.stop()
        event.accept()

if __name__ == '__main__':
    load_dotenv()
    SERIAL = os.environ.get("UNICORN_SERIAL")
    
    if not SERIAL:
        print("❌ ERROR: No se encontró UNICORN_SERIAL en el archivo .env")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = EEGVisualizer(SERIAL)
    window.show()
    sys.exit(app.exec())