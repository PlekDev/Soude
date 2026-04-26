import pygame
import sys
from pylsl import StreamInfo, StreamOutlet
import time

# --- Configuración Crítica ---
FREQ_OBJETIVO = 15  # Frecuencia SSVEP deseada en Hz
FPS_MONITOR = 60    # CAMBIA ESTO a 120 si configuraste tu monitor a 120Hz

# Cálculo exacto de frames por ciclo
frames_por_ciclo = FPS_MONITOR // FREQ_OBJETIVO
frames_mitad = frames_por_ciclo // 2

def main():
    # Inicializar Pygame
    pygame.init()
    
    # Es fundamental activar DOUBLEBUF y vsync=1 para amarrarnos al hardware
    # Usamos FULLSCREEN para máxima inmersión
    flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
    
    try:
        # Intentamos forzar V-Sync (solo funciona en pygame 2.0+)
        pantalla = pygame.display.set_mode((0, 0), flags, vsync=1)
    except Exception as e:
        print("Advertencia: No se pudo forzar V-Sync estricto. Revisa tu compositor en Arch.")
        pantalla = pygame.display.set_mode((0, 0), flags)

    reloj = pygame.time.Clock()

    # Colores
    blanco = (255, 255, 255)
    negro = (0, 0, 0)
    
    # Opcional: En lugar de toda la pantalla, podemos dibujar un cuadrado en el centro
    # para que sea menos agotador visualmente.
    ancho_pantalla, alto_pantalla = pantalla.get_size()
    tamano_cuadrado = 400
    rect_centro = pygame.Rect(
        (ancho_pantalla // 2) - (tamano_cuadrado // 2),
        (alto_pantalla // 2) - (tamano_cuadrado // 2),
        tamano_cuadrado, tamano_cuadrado
    )

    contador_frames = 0
    corriendo = True

    print(f"Iniciando estímulo a {FREQ_OBJETIVO} Hz...")
    print(f"Basado en un monitor de {FPS_MONITOR} Hz. Ciclo: {frames_por_ciclo} frames.")
    print("Presiona ESC para salir.")

    print("Creando stream de marcadores LSL...")
    # Creamos un stream de tipo 'Markers' que envía strings (texto) de forma irregular (0 Hz)
    info_marcador = StreamInfo('SSVEP_Marcadores', 'Markers', 1, 0, 'string', 'estimulo_15hz')
    outlet_marcador = StreamOutlet(info_marcador)

    marcador_enviado = False

    while corriendo:
        # Manejo de eventos (como salir)
        for evento in pygame.event.get():
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    corriendo = False

        # --- Lógica basada en Hardware (Frames) ---
        # Si estamos en la primera mitad del ciclo, pintamos blanco; si no, negro.
        if contador_frames % frames_por_ciclo < frames_mitad:
            color_actual = blanco
        else:
            color_actual = negro

        # Dibujar (Elige una de las dos opciones)
        
        # Opción 1: Pantalla completa (Estímulo muy fuerte)
        # pantalla.fill(color_actual)
        
        # Opción 2: Solo un cuadrado en el centro (Más cómodo, fondo gris oscuro)
        pantalla.fill((20, 20, 20)) 
        pygame.draw.rect(pantalla, color_actual, rect_centro)

        # Volcar el dibujo a la pantalla física (espera automáticamente al V-Sync)
        pygame.display.flip()
        
        # Enviar el marcador EXACTAMENTE después del primer dibujo en pantalla
        if not marcador_enviado:
            # Enviamos una lista con un solo string
            outlet_marcador.push_sample(['INICIO_15HZ'])
            print("¡Marcador 'INICIO_15HZ' enviado a la red!")
            marcador_enviado = True

        contador_frames += 1
        
        # Le decimos a Pygame que no exceda los FPS del monitor
        reloj.tick(FPS_MONITOR)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()