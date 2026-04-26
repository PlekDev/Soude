import sys
import time
from PyQt5 import QtWidgets, QtCore, QtGui

class SSVEP_Display(QtWidgets.QWidget):
    def __init__(self, frequency=15.0):
        super().__init__()
        self.freq = frequency
        
        # Calculamos la duración exacta de medio ciclo (tiempo en Blanco o Negro)
        self.period = 1.0 / self.freq
        self.half_period = self.period / 2.0
        
        # Variables de estado
        self.is_white = False
        self.last_flip_time = time.perf_counter()
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"SSVEP Passthought - {self.freq} Hz")
        # Fondo inicial negro
        self.setStyleSheet("background-color: black;")
        
        # Usamos un timer de alta precisión corriendo a 1ms
        # Actúa como un bucle de juego para interceptar el momento exacto del cambio
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.check_flip)
        self.timer.start(1)
        
    def check_flip(self):
        current_time = time.perf_counter()
        
        # Si el tiempo transcurrido superó la mitad del periodo, invertimos el color
        if (current_time - self.last_flip_time) >= self.half_period:
            self.is_white = not self.is_white
            
            # CRÍTICO: Sumamos el half_period en lugar de igualar a current_time
            # Esto evita que los micro-retrasos del procesador se acumulen con el tiempo
            self.last_flip_time += self.half_period 
            
            if self.is_white:
                self.setStyleSheet("background-color: white;")
            else:
                self.setStyleSheet("background-color: black;")

    def keyPressEvent(self, event):
        # Permitir salir fácilmente presionando la tecla Escape
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Forzar la sincronización con el refresco de la pantalla (VSync)
    format = QtGui.QSurfaceFormat()
    format.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(format)
    
    # Iniciar la ventana a 15 Hz
    window = SSVEP_Display(frequency=15.0)
    
    # FullScreen inmersivo es clave para que el estímulo sature el campo visual
    window.showFullScreen() 
    
    sys.exit(app.exec())