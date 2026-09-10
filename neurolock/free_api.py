import serial
import time
import struct
import numpy as np


# --- Configuración del puerto ---

PUERTO = 'COM10'

BAUD_RATE = 115200


nchan = 16

fsample = 250


# Secuencia exacta de inicio para el Unicorn Hybrid Black

start_acq = bytes([0x61, 0x7C, 0x87])

stop_acq = bytes([0x63, 0x5C, 0xC5])


def iniciar_escucha():

try:

# Inicializa la conexión con el puerto serie

conexion = serial.Serial(PUERTO, BAUD_RATE, timeout=5)

print(f"[*] Conexión exitosa al puerto {PUERTO} a {BAUD_RATE} baudios.")


# Enviar comando de inicio de adquisición al hardware

print("[*] Enviando comando de inicio al Unicorn...")

conexion.write(start_acq)

# Leer la respuesta de inicio (3 bytes)

response = conexion.read(3)

if response != b'\x00\x00\x00':

print(f"[!] Advertencia: Respuesta de inicio inusual: {response.hex()}")

else:

print("[*] ¡Dispositivo iniciado correctamente! Leyendo sensores...")


print("-" * 65)


while True:

# El Unicorn envía paquetes de 45 bytes por muestra

payload = conexion.read(45)

if len(payload) < 45:

continue


# Validar cabecera y pie de paquete

if payload[0:2] != b'\xC0\x00' or payload[43:45] != b'\x0D\x0A':

continue


# Calcular batería y canales EEG

battery = 100 * float(payload[2] & 0x0F) / 15

eeg = np.zeros(8)

for ch in range(0, 8):

# Extraer canales de 3 bytes (Big-Endian)

raw_bytes = b'\x00' + payload[(3 + ch * 3):(6 + ch * 3)]

eegv = struct.unpack('>i', raw_bytes)[0]

# Manejo correcto del complemento a dos para el signo en Python

if (eegv & 0x00800000):

eegv -= 0x01000000

eeg[ch] = float(eegv) * 4500000. / 50331642.


# Acelerómetro y Giroscopio

accel = np.zeros(3)

accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.

accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.

accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.


gyro = np.zeros(3)

gyro[0] = float(struct.unpack('<h', payload[33:35])[0]) / 32.8

gyro[1] = float(struct.unpack('<h', payload[35:37])[0]) / 32.8

gyro[2] = float(struct.unpack('<h', payload[37:39])[0]) / 32.8


counter = struct.unpack('<L', payload[39:43])[0]


# --- IMPRIMIR LOS DATOS DIRECTAMENTE EN CONSOLA ---

# Imprimimos cada 250 muestras (1 segundo) para no saturar la pantalla con 250 líneas por segundo

if (counter % fsample) == 0:

eeg_str = ", ".join([f"{val:.2f}" for val in eeg])

print(f"[{counter}] Bat: {int(battery)}% | EEG (µV): [{eeg_str}]")


except serial.SerialException as e:

print(f"[!] Error en el puerto serie: {e}")

except KeyboardInterrupt:

print("\n[*] Deteniendo adquisición...")

if 'conexion' in locals() and conexion.is_open:

conexion.write(stop_acq)

finally:

if 'conexion' in locals() and conexion.is_open:

conexion.close()

print("[*] Puerto cerrado correctamente.")


if __name__ == '__main__':

iniciar_escucha()

