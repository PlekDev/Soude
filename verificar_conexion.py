import pylsl

print("🔍 Buscando transmisiones cerebrales en la red local...")

# Le damos 5 segundos para buscar en el Wi-Fi cualquier stream etiquetado como 'EEG'
streams = pylsl.resolve_byprop('type', 'EEG', timeout=5)

if not streams:
    print("❌ No se detectó ninguna señal.")
    print("Posibles causas:")
    print(" 1. No están en el mismo Wi-Fi.")
    print(" 2. El Firewall de Windows de tu amigo está bloqueando la transmisión (Dile que le dé permisos a Python).")
    print(" 3. El código de tu amigo no está corriendo.")
else:
    # Si lo encuentra, nos "enchufamos" a la transmisión
    inlet = pylsl.StreamInlet(streams[0])
    print(f"✅ ¡Conexión exitosa al stream: {streams[0].name()}!")
    print("Mostrando los datos crudos en tiempo real (Presiona Ctrl+C para detener):\n")

    try:
        while True:
            # Jalamos una muestra del aire
            sample, timestamp = inlet.pull_sample()
            
            # Imprimimos la marca de tiempo y los números de los primeros 4 electrodos
            # (Cortamos la lista para que no sature tu pantalla de texto)
            print(f"[{timestamp:.2f}] Datos -> {sample[:4]} ...")
            
    except KeyboardInterrupt:
        print("\n🔌 Prueba de conexión finalizada.")