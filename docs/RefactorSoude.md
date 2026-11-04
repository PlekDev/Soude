# Rediseño técnico de Soude

## 1. Propósito del documento

Este documento redefine Soude como un sistema de autenticación biométrica basado en EEG, no como un simple demostrador de detección de potenciales evocados tipo P300. La intención es cerrar la brecha entre la prueba de concepto y una solución arquitectónicamente correcta, segura, medible y verificable desde el punto de vista de rendimiento biométrico y seguridad computacional.

La versión actual del proyecto demuestra que es posible detectar P300 en un flujo de estimulación visual. Sin embargo, esa capacidad no equivale, por sí sola, a autenticación segura. Para que un sistema de este tipo pueda considerarse viable, debe cumplir con los requisitos de:

- identidad biométrica individual por usuario;
- almacenamiento seguro de plantillas y referencias del usuario;
- trazabilidad completa de cada intento de autenticación;
- calidad objetiva de la señal y rechazo de artefactos;
- separación clara entre datos clínicos/analíticos y datos operativos;
- decisiones probabilísticas en lugar de umbrales rígidos;
- evidencia reproducible para evaluación de FAR, FRR, EER y robustez ante impostores.

---

## 2. Diagnóstico técnico del proyecto actual

### 2.1. Fortalezas existentes

El repositorio ya tiene una base relevante:

- adquisición EEG con Unicorn y simulador mock;
- filtrado y extracción de epochs para P300;
- pipeline de autenticación con detección basada en diferencias de amplitud;
- grabación estructurada de sesiones;
- soporte de flujo LSL y monitorización de señal;
- base para entrenamiento de modelos ML.

Estas piezas son una base útil para investigación y desarrollo, pero no están integradas como un sistema biométrico completo.

### 2.2. Deficiencias críticas

1. El proyecto se comporta más como un detector de eventos que como un autenticador.
   - La decisión final se basa fundamentalmente en un valor de amplitud promedio en una ventana específica.
   - El modelo no incorpora identidad del usuario como variable estructural del sistema.

2. Falta una firma biométrica por persona.
   - La P300 no es solo un evento; en un usuario hay patrones de latencia, topografía, amplitud y respuesta cognitiva únicos.
   - Actualmente no se construye un template de identidad asociada a cada persona.

3. No hay modelo de seguridad para datos biométricos.
   - No existen mecanismos de protección para plantillas EEG, referencias de usuario o perfiles de autenticación.
   - No se distingue adecuadamente entre datos de producción, datos de entrenamiento y evidencias de auditoría.

4. La calidad de la señal no se gestiona como primer-class citizen.
   - El sistema no tiene una política formal de rechazo de sesiones por ruido, mala impedancia, artefactos motores o pérdida de contacto.
   - No hay métricas de canal con umbrales operativos, ni re-challenge automático.

5. LSL no está operando como canal técnico robusto de adquisición y sincronización.
   - El flujo tiene potencial, pero necesita definirse como un mecanismo de transporte y validación de streaming con metadatos y sincronía entre señal y marcas.

6. No existe un mecanismo de decisión probabilística suficiente.
   - El umbral simple de amplitud es útil para prototipo, pero no es un criterio de seguridad biométrica.
   - Hace falta un modelo de verosimilitud o un DDM para distinguir autenticación genuina, impostor y sesión no confiable.

7. Falta trazabilidad completa de sesión.
   - Para un sistema biométrico, cada intento debe ser reconstruible y auditado.
   - El registro actual no alcanza nivel de seguridad ni de análisis forense.

---

## 3. Estado actual del proyecto y flujo real de datos

### 3.1. Qué hace el proyecto en este momento

El repositorio actual ya implementa una prueba de concepto funcional de autenticación cerebral basada en P300. La arquitectura vigente está centrada en estos elementos:

- `run.py` lanza la aplicación principal y habilita el flujo de autenticación.
- `neurolock/brain_engine.py` abstrae la adquisición de EEG desde hardware real (`RealUnicorn`), simulador (`MockUnicorn`) o flujo LSL.
- `neurolock/stimulus_runner.py` genera el paradigma visual oddball con imágenes y marca el momento de cada estímulo.
- `neurolock/signal_processing.py` aplica filtrado, extracción de epochs, promedio de señales objetivo y no objetivo, y toma la decisión final con un umbral de amplitud.
- `neurolock/data_logger.py` guarda la sesión en `logs/<timestamp>/` con `markers.csv`, `epochs.npz` y `auth_result.json`.
- `tools/lsl_sender.py` y `tools/lsl_monitor.py` permiten el flujo de streaming vía LSL entre máquinas.
- `ml/train_model.py` y `ml/test_model.py` apuntan a una segunda línea de trabajo con clasificador SVM / Passthought, aunque esta parte no forma la capa principal de autenticación.
- `tests/` valida componentes clave del pipeline sin depender del hardware real.

En términos de comportamiento operativo, el flujo actual consiste en:

1. El usuario selecciona 3 imágenes como contraseña mental.
2. Se lanza una secuencia de estimulación visual con imágenes aleatorias.
3. El sistema captura EEG en tiempo real.
4. Se extrae una ventana temporal post-estímulo.
5. Se filtra y corrige la línea base.
6. Se compara la media de la clase objetivo frente a la clase no objetivo.
7. Si la diferencia supera umbral, se concede acceso.
8. Se guarda la sesión con métricas de decisión y evidencia temporal.

### 3.2. Organización actual del repositorio

La estructura vigente del proyecto es la siguiente:

- `neurolock/`: lógica principal del sistema (hardware, filtrado, pipeline y UI).
- `ml/`: scripts para modelado y validación de segundo nivel.
- `tools/`: utilidades para visualización, diagnósticos y streaming LSL.
- `tests/`: tests para el pipeline y validación funcional.
- `logs/`: almacenamiento de sesiones actuales.
- `data/`: datasets y artefactos generados por los modelos.
- `docs/`: documentación de arquitectura y guiones.
- `assets/`: recursos visuales y gráficos del proyecto.

La organización actual deja claro que el proyecto está dividido por funcionalidad técnica, pero todavía no está separada la capa de autenticación biométrica, la capa de seguridad, la capa de sesión y la capa de análisis estadístico.

### 3.3. Flujo real de datos existente

El flujo actual puede resumirse como:

```text
Unicorn / Mock / LSL
        │
        ▼
BrainEngine
        │
        ├── ring buffer (señal en bruto)
        ├── mark_stimulus() para eventos de estímulo
        ▼
SignalProcessing
        │
        ├── filtrado (1–10 Hz + notch 60 Hz)
        ├── epoching + baseline correction
        ├── artifact rejection
        ├── grand average target vs non-target
        └── AuthResult (delta_uv, snr_db, n_target, n_nontarget)
        │
        ▼
SessionLogger
        │
        ├── markers.csv
        ├── epochs.npz
        ├── auth_result.json
        └── summary.txt
```

Este flujo es funcional como demostrador experimental, pero todavía no incorpora:

- perfil biométrico por usuario;
- comparación frente a plantilla segura;
- decisión probabilística;
- separación entre identidad y evidencia biométrica;
- almacenamiento protegido para plantillas;
- política clara de rechazos por calidad de señal.

### 3.4. Estado actual de seguridad y trazabilidad

El estado actual del proyecto presenta avances relevantes, pero aún no cumple un estándar de seguridad biométrica. En particular:

- la sesión se guarda localmente en `logs/` con datos de señal y resultados;
- la autenticación no está ligada a un perfil de usuario con template cifrado;
- la identidad de la persona no existe como estado persistente ni como registro de biometría;
- no hay una capa de vault, cifrado, control de acceso ni gestión de secretos;
- la decisión final se basa en un umbral estadístico simple, no en una entidad de identidad biométrica y riesgo;
- no hay una política de retención ni clasificación de datos sensibles por dominio.

En otras palabras, la solución ya captura y procesa EEG, pero todavía no dispone de la capa de seguridad y autenticación que haría viable un sistema biométrico real.

### 3.5. Línea base para el cambio

El punto de partida para el rediseño es este estado actual:

- se tiene una base funcional de despliegue P300;
- se tiene evidencia de flujo de EEG, filtrado y validación de sesión;
- se tiene un registro estructurado de eventos y métricas;
- falta la capa de identidad biométrica, protección y decisión probabilística.

La transición debe hacerse desde este punto, sin perder los avances técnicos ya existentes, pero reestructurando la solución para convertirla en un sistema de autenticación cerebral más sólido y defensible.

---

## 4. Objetivos de rediseño

### 3.1. Objetivo principal

Diseñar Soude como un sistema de autenticación cerebral con identidad biométrica, validación probabilística, seguridad de datos y trazabilidad operacional.

### 3.2. Objetivos funcionales

- Realizar enrolamiento biométrico individual por usuario.
- Construir plantillas y referencias para validación posterior.
- Ejecutar autenticación con estimulación P300 robusta.
- Evaluar la calidad de la señal antes de aceptar una decisión.
- Registrar la sesión completa para análisis forense y diagnóstico técnico.
- Usar modelos de decisión probabilística para separar autenticación genuina, impostor y rechazo seguro.

### 3.3. Objetivos no funcionales

- Seguridad: aislamiento, cifrado, minimización de datos sensibles.
- Reproducibilidad: cada sesión debe poder reanalizarse.
- Robustez: rechazo de artefactos y mala calidad.
- Auditabilidad: trazabilidad de decisiones y eventos.
- Escalabilidad: soporte de múltiples usuarios y sesiones futuras.
- Privacidad: separación entre identidad, biometría y análisis clínico.

---

## 4. Requerimientos de arquitectura

### 4.1. Separación de responsabilidades

El sistema debe dividirse en subsistemas funcionales claramente aislados:

- Adquisición
- Procesado de la señal
- Extracción de features
- Modelado biométrico
- Decisión de autenticación
- Almacenamiento de plantillas y perfiles
- Registro de sesión
- Visualización y diagnóstico
- Seguridad y política de acceso

Esto evita acoplar hardware, ML, seguridad y UI en una sola capa monolítica.

### 4.2. Regla arquitectónica central

El sistema no debe guardar la respuesta biométrica como un secreto legible o reutilizable sin transformación. Debe trabajar con representaciones derivadas cifradas o con plantillas protegidas, no con datos EEG crudos como identidad primaria.

### 4.3. Diagrama conceptual

```text
Usuario
  │
  ▼
UI / flujo de autenticación
  │
  ▼
Orquestador de sesión
  ├── Adquisición EEG (Unicorn / Mock / LSL)
  ├── Calidad de señal
  ├── Filtrado y epoching
  ├── Extracción features P300
  ├── Comparación biométrica vs plantilla
  ├── Modelo DDM / probabilístico
  ├── Registro de sesión y auditoría
  └── Seguridad / Vault / cifrado
```

### 4.4. Contrato técnico real entre módulos

La refactorización solo será entendible si se conserva el contrato actual entre archivos. El repositorio ya tiene un flujo de datos bastante definido, aunque todavía no está formalizado como una arquitectura modular clara.

#### 4.4.1. Parámetros y valores fijos que usa el sistema

| Parámetro | Valor real | Uso en la implementación |
|---|---:|---|
| `SAMPLE_RATE` | 250 Hz | frecuencia de muestreo nativa del Unicorn Hybrid Black |
| `N_CHANNELS` | 8 | canales EEG: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 |
| `BUFFER_SECONDS` | 120 s | tamaño del ring buffer de adquisición |
| `EPOCH_DURATION_S` | 0.800 s | duración de cada epoch post-estímulo |
| `BASELINE_DURATION_S` | 0.100 s | ventana previa al estímulo para corrección de línea base |
| `P300_ONSET_S` | 0.250 s | inicio de la ventana de interés del P300 |
| `P300_OFFSET_S` | 0.500 s | fin de la ventana de interés del P300 |
| `AUTH_THRESHOLD_UV` | 1.5 µV | diferencia media target-vs-non-target requerida para conceder acceso |
| `MIN_EPOCHS` | 5 | mínimo de epochs válidas por clase para mantener la decisión |
| `SOA_S` | 0.500 s | tiempo entre flashes del paradigma visual |
| `BLANK_S` | 0.075 s | tiempo de pantalla en blanco |
| `TOTAL_IMAGES` | 20 | catálogo total de imágenes por sesión |
| `N_TARGETS` | 3 | imágenes del password mental |
| `TARGET_REPEATS` | 5 | repeticiones por imagen objetivo |
| `NONTARGET_REPEATS` | 5 | repeticiones por distractor |

Con estos valores, una secuencia completa tiene:

- 3 targets × 5 repeticiones = 15 target epochs
- 17 distractores × 5 repeticiones = 85 nontarget epochs
- total de eventos = 100
- tiempo aproximado = 100 × (0.500 + 0.075) ≈ 57.5 s

El patrón está diseñado para mantener una proporción target cercana al 15%, que es la base del paradigma oddball visual.

#### 4.4.2. Flujo real de datos actual entre módulos

```text
run.py
  │
  ▼
neurolock/ui/app.py
  │
  ├── crea BrainEngine con Unicorn / Mock / LSL
  ├── configura StimulusRunner con la contraseña mental
  ├── dispara el paradigma visual
  │
  ▼
StimulusRunner.run_sync()
  │
  ├── _build_sequence() -> lista de 100 eventos
  ├── on_show(image_id, is_target)
  ├── engine.mark_stimulus(image_id)
  │      └── devuelve StimulusMarker {image_id, buffer_index, timestamp, is_target}
  ├── espera SOA_S + BLANK_S
  └── on_complete(events)

  │
  ▼
BrainEngine / ring buffer
  │
  ├── captura EEG en bloques de n_samples
  ├── normaliza a estructura UnicornDataPacket
  │      ├── eeg: (n_samples, 8)
  │      ├── accelerometer: (n_samples, 3)
  │      ├── gyroscope: (n_samples, 3)
  │      └── battery: (n_samples,)
  └── almacena el buffer y marca cada evento con posición del buffer

  │
  ▼
AuthenticationPipeline / signal_processing
  │
  ├── EpochExtractor.extract(marker)
  │      ├── toma ventana [buffer_index - 100 ms, buffer_index + 800 ms]
  │      ├── filtra con cadena band-pass 1–10 Hz + notch 60 Hz
  │      ├── baseline correction con 100 ms previos
  │      ├── descarta artefactos si p-p > 100 µV
  │      └── devuelve epoch (EPOCH_SAMPLES, N_CHANNELS)
  ├── SignalAverager acumula target/non-target
  ├── calcula diferencia media en ventana 250–500 ms
  └── genera AuthResult
        ├── granted
        ├── delta_uv
        ├── pre_sigma_uv
        ├── snr_db
        ├── n_target
        └── n_nontarget

  │
  ▼
SessionLogger
  │
  ├── log_marker(marker, epoch)
  │      └── almacena markers.csv + epochs.npz
  ├── log_auth_result(result, erp_data)
  │      └── escribe auth_result.json
  └── flush() -> summary.txt + evidencia de sesión
```

#### 4.4.3. Forma de los datos que se pasan entre módulos

| Módulo origen | Módulo destino | Estructura de datos | Forma/valor |
|---|---|---|---|
| `app.py` | `StimulusRunner` | `password_ids` | lista de `int` (por ejemplo `[0, 5, 12]`) |
| `StimulusRunner` | `BrainEngine` | `StimulusMarker` | `image_id`, `buffer_index`, `timestamp`, `is_target` |
| `BrainEngine` | `EpochExtractor` | `UnicornDataPacket` y buffer de EEG | `eeg.shape = (n_samples, 8)` |
| `EpochExtractor` | `SignalAverager` | `epoch` | `(EPOCH_SAMPLES, N_CHANNELS)` = `(200, 8)` |
| `SignalAverager` | `AuthenticationPipeline` | promedio target / no target | matrices `(200, 8)` |
| `AuthenticationPipeline` | `SessionLogger` | `AuthResult` | `granted`, `delta_uv`, `pre_sigma_uv`, `snr_db`, `n_target`, `n_nontarget` |

Concretamente:

- `BrainEngine` usa 250 Hz y 8 canales, por lo que una ventana de 800 ms equivale a `0.8 × 250 = 200` muestras.
- la corrección de línea base usa los primeros 100 ms, es decir `0.1 × 250 = 25` muestras.
- la ventana P300 de interés se define entre 250 ms y 500 ms, es decir `0.25 × 250 = 62.5` muestras y `0.5 × 250 = 125` muestras.
- una sesión de 100 estímulos con 500 ms SOA + 75 ms blank dura aproximadamente 57.5 s, pero el tiempo total real depende del jitter del sistema y de la sincronización del reloj de adquisición.

#### 4.4.4. Qué debe mantenerse durante la refactorización

La refactorización no debe destruir este contrato. Lo que sí debe hacerse es encapsularlo en capas más limpias:

- adquisición → `acquisition/`
- filtrado y epoching → `signal/`
- métricas de calidad → `quality/`
- modelado biométrico → `biometrics/`
- decisión final → `modeling/` o `decision/`
- persistencia de sesión → `sessions/`
- seguridad de plantillas y datos → `security/`

El punto clave es que la refactorización debe preservar la semántica actual de cada dato:

- el estímulo lleva `image_id` y `is_target`;
- la sesión guarda `buffer_index` y `timestamp` para reconstruir la señal;
- el epoch preserva la estructura temporal `200×8`;
- la decisión final conserva `delta_uv`, `pre_sigma_uv`, `snr_db` y `granted` para análisis posterior;
- la evidencia se registra para reanálisis offline y medición FAR/FRR/EER.

Esto es lo que permite que la nueva arquitectura conserve compatibilidad con el comportamiento existente mientras se mejora la seguridad, identidad biométrica y la calidad de la señal.

---

## 5. Modelo biométrico recomendado

### 5.1. Definición del perfil del usuario

Cada usuario debe contar con un perfil de autenticación biométrica estructurado con al menos estos componentes:

- identificador de usuario (UUID o equivalente);
- configuración de la contraseña mental;
- conjunto de estímulos asociados a la contraseña;
- template de respuesta P300 por patrón y canal;
- latencia promedio de pico;
- amplitud promedio objetivo vs no objetivo;
- valor de SNR y variabilidad por sesión;
- metadata de enrolamiento y versión del algoritmo;
- hash o derivación de la plantilla cifrada;
- historial de sesiones y tasas de error.

### 5.2. Plantilla biométrica

La plantilla no debe contener la señal completa en bruto como elemento de identidad principal. Debe contener un vector compacto de características, por ejemplo:

- amplitud media en ventana P300 por canal;
- diferencia media target-vs-non-target;
- latencia de pico por canal;
- topografía espacial;
- energía de la señal en bandas relevantes;
- varianza y ruido de línea base;
- score de confiabilidad por sesión.

Esto permite comparación probabilística y minimiza la exposición de datos sensibles.

### 5.3. Enrolamiento

El flujo de enrolamiento debe contemplar varias sesiones, no una sola:

- sesión 1: calibración e inicialización de template;
- sesión 2–N: validación del template y ajuste estadístico;
- rechazo de sesiones de baja calidad;
- actualización controlada del template con políticas de drift.

La actualización del template no debe hacerse automáticamente sobre cualquier sesión. Debe tener umbrales de confianza y revisión de calidad.

---

## 6. Seguridad y privacidad

El documento [docs/SECURITY.md](docs/SECURITY.md) complementa este plan con el modelo completo de seguridad. En este bloque se resumen los requerimientos mínimos que deben aplicarse en el sistema.

### 6.1. Principios básicos

- minimización de datos: almacenar solo lo indispensable;
- no persistir EEG crudo como secreto primario;
- cifrar plantillas y metadatos sensibles;
- separar identidad del análisis biométrico;
- no exponer información por canal no autorizado;
- imponer autenticación del backend y control de acceso por rol;
- registrar eventos de seguridad y decisiones de autenticación.

### 6.2. Datos sensibles

Se deben clasificar en al menos tres niveles:

1. Datos de identidad y configuración del usuario
2. Datos biométricos: plantillas, scores, métricas P300, metadatos de sesión
3. Evidencias técnicas: EEG crudo, epochs, logs, marcadores, resultados de análisis

Solo los niveles necesarios deben estar accesibles en cada capa.

### 6.3. Requisitos de almacenamiento

- utilizar cifrado en reposo para datos persistidos;
- proteger acceso a archivos de sesión en almacenamiento local o distribuido;
- separar la base de perfiles biométricos de la base analítica;
- usar almacenamiento aislado para evidencias de auditoría;
- no guardar contraseñas o claves en texto plano.

---

## 7. Calidad de señal y validación operativa

### 7.1. Requisitos de validación por canal

Cada sesión debe evaluarse con al menos estas métricas:

- variancia por canal;
- amplitud pico a pico;
- línea base; 
- ratio de artefactos;
- SNR por canal;
- alineación temporal con marcadores;
- proporción de epochs rechazadas;
- calidad de contacto por electrodo.

### 7.2. Política de rechazo

La sesión debe rechazarse cuando:

- demasiados electrodos tengan baja calidad;
- la línea de base es inestable;
- la variancia excede rangos razonables;
- hay artefactos motores o de parpadeo sin control;
- la señal no cumple el mínimo de epochs válidas por clase;
- la diferencia target-vs-non-target no es estadísticamente confiable.

### 7.3. Visualización operativa

Se requiere una vista de adquisición en vivo con:

- 8 canales del Unicorn;
- sincronismo con marcadores;
- componente de calidad por canal;
- overlays de target/no-target;
- rango temporal configurable;
- vista de epochs y promedio temporal;
- gráfica de ERP en tiempo real.

La visualización debe tratarse como herramienta de diagnóstico, no solo como componente estético.

---

## 8. LSL, streaming y sincronización

### 8.1. Objetivo

El flujo LSL debe funcionar como un canal de transporte y sincronización para EEG y marcadores entre distintos procesos o máquinas. Esto permite separar adquisición, análisis y visualización.

### 8.2. Streams requeridos

- `Unicorn_EEG`: datos crudos del EEG (
  - 8 canales,
  - muestreo nativo,
  - timestamps por muestra o por bloque)
- `Stimulus_Markers`: eventos de estimulación visual
- `P300_Markers`: marcas de eventos detectados o clasificados
- `Session_Metadata`: ID de usuario, tipo de sesión, versión del pipeline, parámetros de configuración

### 8.3. Garantías técnicas

- mantener reloj común o vector de tiempo equivalente;
- validar latencia y pérdida de muestras;
- registrar timestamps del evento y del stream EEG;
- asegurar que la sesión pueda reconstruirse offline;
- evitar correlación débil entre señal y marcadores.

### 8.4. Receptor

El receptor debe:

- verificar que existe el stream esperado;
- validar integridad del flujo;
- registrar sincronización de marcas;
- reconstituir epochs a partir de eventos y tiempos del buffer;
- permitir exportación a formato de análisis reproducible.

---

## 9. Pipeline de inferencia y decisión final

### 9.1. Etapa de extracción de features

El pipeline debe transformar la señal en un conjunto de features útiles para autenticación y decisión:

- amplitud media de P300;
- área bajo la curva 250–500 ms;
- latencia de pico;
- topografía por canal;
- diferencia target-non-target por canal;
- SNR de la sesión;
- número de epochs válidas;
- score de confiabilidad por clase.

### 9.2. Decisión probabilística

Debe reemplazarse el flujo de umbral simple por una estrategia probabilística basada en evidencias acumuladas.

Recomendación de arquitectura:

- `P300_detector`: identificación del evento y de la magnitud del potencial evocado;
- `Feature_extractor`: cálculo de features biométricas;
- `BiometricComparator`: comparación del patrón actual contra la plantilla del usuario;
- `DDM / BayesianDecisionModel`: decisión final con probabilidad de autenticidad;
- `PolicyLayer`: decisión final con política de aceptado, rechazado o reto adicional.

### 9.3. Lógica de decisión

La salida no debe ser únicamente `GRANTED` o `DENIED`.

Debe existir un estado intermedio:

- aceptado / alta confianza;
- rechazado / baja confianza;
- reintento / calidad insuficiente;
- challenge required / evidencia ambigua.

Esto es esencial para un sistema biométrico y reduce ataques por fuerza bruta o manipulación con ruido.

---

## 10. Modelo DDM recomendado

Un DDM o modelo de difusión de evidencia es apropiado porque la autenticación basada en EEG no debe depender solo de un valor puntual. La decisión se fundamenta en la acumulación gradual de evidencia a través de la respuesta del usuario.

### 10.1. Variables de entrada

- amplitud del P300;
- latencia y forma del pico;
- SNR de la sesión;
- confiabilidad por canal;
- número de epochs válidas;
- similitud del patrón frente a la plantilla del usuario;
- probabilidad de target según la estimulación visual.

### 10.2. Salidas

- $P(target)$
- $P(genuine)$
- $P(impostor)$
- $P(ambiguous)$

### 10.3. Política operativa

- si $P(genuine)$ supera el umbral de aceptación y la calidad es válida, se concede acceso;
- si $P(genuine)$ está por debajo del umbral, se rechaza;
- si la evidencia es ambigua, se solicita reintento o verificación adicional;
- si la calidad es insuficiente, se aborta la sesión sin privilegiar la autenticación.

Esto mejora la seguridad y evita decisiones sobre señales contaminadas.

---

## 11. Registro y auditoría de sesiones

La sesión completa debe ser registrable bajo un esquema bien definido.

### 11.1. Datos de sesión

- ID de sesión
- timestamp de inicio y fin
- ID del usuario o intento
- tipo de sesión: enrolamiento, autenticación, impostor, validación
- configuración del protocolo
- identificador del dispositivo
- versión del software y modelo de decisión
- lista de estímulos y marcadores
- epochs válidas / descartadas
- calidad por canal
- métricas de decisión
- salida del modelo y puntuación final

### 11.2. Formato de salida

Cada sesión debe exportarse en un formato estructurado con:

- `session.json`: metadatos y decisión final
- `markers.csv`: eventos y tiempo
- `epochs.npz`: señales y ventanas extraídas
- `quality.json`: métricas por canal
- `summary.txt`: resumen operativo
- `raw/`: copia controlada de datos no tratados, protegida y limitada

### 11.3. Política de retención

- datos de producción: periodo acotado y con política definida;
- datos de análisis: almacenados con trazabilidad y control de acceso;
- datos de auditoría: conservación específica para incidentes y validación de seguridad.

---

## 12. Requisitos de pruebas y validación

El proyecto debe evaluarse con criterios métricos reales, no solo con demostraciones visuales.

### 12.1. Métricas mínimas

- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- EER (Equal Error Rate)
- ROC / AUC
- sensibilidad por usuario
- robustez ante ruido y artefactos
- retención de efecto ante sesiones con impostores

### 12.2. Casos de prueba recomendados

- autenticación genuina con calidad aceptable;
- autenticación genuina con ruido moderado;
- sesion de impostor con mismo patrón visual;
- sesión con mala calidad de contacto;
- sesión con baja cantidad de epochs válidas;
- reacción frente a marcadores desalineados;
- rechazo seguro ante evidencia ambigua.

---

## 13. Estructura propuesta del repositorio

```text
Soude/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── auth_flow.py
│   ├── enrollment_flow.py
│   └── ui/
│       ├── auth_view.py
│       ├── enrollment_view.py
│       ├── live_eeg_view.py
│       └── result_view.py
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── time_sync.py
│   └── constants.py
│
├── acquisition/
│   ├── __init__.py
│   ├── unicorn/
│   │   ├── device.py
│   │   ├── ring_buffer.py
│   │   ├── markers.py
│   │   └── quality.py
│   ├── lsl/
│   │   ├── stream_sender.py
│   │   ├── stream_receiver.py
│   │   ├── stream_validator.py
│   │   └── sync.py
│   └── simulator/
│       ├── mock_unicorn.py
│       └── synthetic_p300.py
│
├── signal/
│   ├── __init__.py
│   ├── filters.py
│   ├── epochs.py
│   ├── quality_metrics.py
│   ├── feature_extractor.py
│   └── p300_detector.py
│
├── biometrics/
│   ├── __init__.py
│   ├── user_profile.py
│   ├── enrollment.py
│   ├── template_builder.py
│   ├── comparator.py
│   └── drift_policy.py
│
├── security/
│   ├── __init__.py
│   ├── vault.py
│   ├── crypto.py
│   ├── secret_store.py
│   └── policy.py
│
├── modeling/
│   ├── __init__.py
│   ├── ml_p300.py
│   ├── ddm.py
│   ├── scoring.py
│   └── validation.py
│
├── sessions/
│   ├── __init__.py
│   ├── recorder.py
│   ├── schema.py
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── data/
│   ├── datasets/
│   ├── templates/
│   └── models/
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_quality.py
│   ├── test_lsl.py
│   ├── test_biometric_profile.py
│   └── test_ddm.py
│
├── tools/
│   ├── eeg_visualizer.py
│   ├── debug_viewer.py
│   ├── lsl_sender.py
│   ├── lsl_monitor.py
│   └── export_session.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PLAN_REDISEÑO_SOUDE.md
│   ├── SECURITY.md
│   └── VIDEO_SCRIPT.md
│
├── scripts/
│   ├── train_model.py
│   ├── validate_auth.py
│   └── benchmark_dataset.py
└── examples/
    └── sample_session/
```

---

## 14. Fases recomendadas de implementación

### Fase 0 — Requerimientos y modelo de riesgo

- Definir identidad del usuario.
- Definir supuestos de seguridad.
- Establecer métricas FAR, FRR, EER.
- Definir retención, cifrado y clasificación de datos.

### Fase 1 — Perfil biométrico y enrolamiento

- Construir template por usuario.
- Guardar plantilla protegida.
- Diseñar flujo de enrolamiento multisesión.

### Fase 2 — Calidad de señal y adquisición

- mejorar visualización del EEG;
- monitorear canal por canal;
- validar el flujo LSL y sincronización temporal;
- rechazar sesiones con artefactos de mala calidad.

### Fase 3 — Detección P300 y features

- extraer epochs válidas;
- generar features de discriminación;
- entrenar/simular modelos de clasificación;
- definir un score de confianza inicial.

### Fase 4 — Decisión final con DDM o modelo bayesiano

- integrar evidencia acumulada;
- combinar amplitud, latencia, ruido y personalización;
- separar `accepted`, `rejected`, `ambiguous` y `challenge_required`.

### Fase 5 — Auditoría, validación y seguridad

- grabar la sesión completa;
- medir métricas operativas;
- validar con impostores;
- aplicar revisión de seguridad y controles de acceso.

---

## 15. Recomendación final

Soude tiene una base valiosa como demostrador de P300, pero debe dejar de considerarse un prototipo aislado y pasar a una arquitectura de autenticación biométrica segura. La evolución técnica correcta pasa por seis pilares: identidad biométrica, seguridad, calidad de señal, sincronización y trazabilidad, toma de decisiones probabilística y validación experimental con métricas reales.

La solución no debe preguntarse solo si detecta un P300, sino si puede distinguir con confianza entre un usuario genuino y un impostor, sin aceptar señales contaminadas ni guardar secretos biométricos en formato sensible.

Ese cambio convierte a Soude de un detector de eventos a una infraestructura de autenticación cerebral con base científica, trazabilidad operativa y defensa técnica adecuada ante amenazas reales.
