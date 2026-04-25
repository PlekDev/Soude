import pylsl

print("Buscando el stream del Unicorn en la red...")

# CAMBIO AQUÍ: Usamos resolve_byprop en lugar de resolve_streams
# 'type' es la propiedad, 'EEG' es el valor
streams = pylsl.resolve_byprop('type', 'EEG', timeout=5)

if not streams:
    print("No se encontró nada. Revisa que la App de Unicorn LSL esté en 'Start'.")
else:
    inlet = pylsl.StreamInlet(streams[0])
    print(f"¡Éxito! Conectado a: {streams[0].name()}")

    while True:
        sample, timestamp = inlet.pull_sample()
        print(f"Cerebro recibiendo -> {sample}")