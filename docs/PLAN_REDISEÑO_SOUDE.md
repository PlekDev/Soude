# Plan de rediseño del proyecto Soude

## 1. Diagnóstico general

El proyecto actual tiene una base sólida como demostrador de detección de potenciales evocados tipo P300 y como prueba de concepto de autenticación cerebral, pero en su estado actual no cumple aún la lógica de un sistema autenticador biométrico robusto.

Los principales problemas detectados son:

- La lógica se centra más en detectar un evento P300 que en autenticar a una persona específica.
- No existe una identidad biométrica sólida por usuario: cada individuo tiene patrones únicos en su EEG, pero no se está modelando esa individualidad.
- No existe un esquema real de almacenamiento seguro para credenciales biométricas ni de contraseña cerebral.
- No hay una capa de trazabilidad para la sesión completa de autenticación.
- La visualización de canales EEG del Unicorn y del flujo LSL no está integrada como herramienta de calidad ni como canal de validación clínica o de diagnóstico.
- No se está incorporando una estrategia de decisión probabilística más avanzada como un modelo DDM (Drift Diffusion Model) o una variante bayesiana para distinguir autenticación genuina vs impostor.
- La solución aún no integra una definición clara de seguridad, privacidad, pseudonimización y validación de fraude.

En otras palabras, se tiene una prueba de concepto funcional de detección de P300, pero aún falta convertirla en un sistema seguro, reproducible, medible y profesional.

---

## 2. Objetivo del rediseño

Convertir Soude en un sistema de autenticación cerebral completo con estas propiedades:

1. Identidad biométrica real: cada usuario tiene una firma neural de referencia.
2. Enrolamiento y validación con criterio de autenticación reproducible.
3. Seguridad y privacidad: almacenamiento seguro, no dejar señales crudas sin protección, ni credenciales en texto plano.
4. Monitoreo de calidad de señal: validación por canal, calidad de contacto, artefactos y SNR.
5. Captura y visualización de la sesión completa, incluyendo metadata, marcadores y eventos P300.
6. Detección y clasificación basada en análisis estadístico y modelos de decisión probabilística.
7. Registro completo para auditoría, análisis posterior y mejora del algoritmo.

---

## 3. Cambios necesarios por bloque

### 3.1. Autenticación biométrica real

Actualmente se usa un umbral de diferencia de amplitud P300 para aceptar o rechazar. Eso no es suficiente como autenticador, porque:

- no distingue entre una persona legítima y un intruso con comportamiento similar;
- no usa una firma biométrica por usuario;
- no valida que la respuesta sea única para cada individuo;
- no incorpora un patrón de referencia del usuario.

Cambios requeridos:

- Crear un perfil biométrico por usuario con una plantilla de EEG basada en:
  - promedio de señales objetivo vs no objetivo;
  - latencia de pico P300;
  - amplitud media por canal;
  - topografía espacial (Cz, Pz, Oz, etc.);
  - varianza y SNR por sesión;
  - patrón de respuesta asociado a la contraseña mental.
- Separar claramente dos conceptos:
  1. autenticación por estímulo visual / P300;
  2. identidad biométrica del usuario.
- Enrolamiento del usuario en dos pasos:
  - registro de identidad y credenciales administrativas;
  - registro neural de la contraseña mental y sus características biométricas.
- Para cada usuario se debe guardar una referencia de autenticación, no la sesión completa en bruto como secret almacenada.

### 3.2. Seguridad y almacenamiento

No se debe guardar una contraseña “cerebral” como un objeto directamente legible o vulnerable. La solución actual no incluye hash, cifrado ni protección de la información sensible.

Cambios recomendados:

- Guardar únicamente representaciones derivadas y cifradas del template biométrico.
- Usar cifrado del repositorio de perfiles y secretos.
- Usar almacenamiento en memoria segura o en un vault compatible con KMS/secret manager.
- Usar salting, HMAC o derivación criptográfica para secretos de acceso o claves de sesión.
- No guardar señales EEG crudas como credencial primaria; solo como evidencia de auditoría y análisis forense.
- Diferenciar entre:
  - datos de autenticación (persistentes y protegidos);
  - datos de análisis (sesiones, logs, debugging, entrenamiento);
  - datos operativos (métricas, canal de calidad, eventos, marcas temporales).

### 3.3. Calidad de canales y visualización EEG

La parte actual de visualización parece limitada para revisión del flujo y no entrega una gráfica de calidad por canal ni una inspección robusta de datos provenientes del Unicorn. Esto es crítico para un sistema biométrico basado en EEG.

Cambios requeridos:

- Implementar un visor de canales con:
  - 8 canales del Unicorn;
  - corrientes en microvoltios;
  - ventana de tiempo configurable;
  - overlays de eventos / marcas de estímulo;
  - estado de calidad por canal (buena, media, mala);
  - indicadores de artefactos, contacto, ruido y saturación.
- Añadir métricas de calidad:
  - variancia por canal;
  - amplitud pico a pico;
  - línea base;
  - SNR por canal;
  - proporción de artefactos por sesión;
  - estimación de contacto de electrodos.
- Separar dos vistas:
  - vista de adquisición en vivo;
  - vista de análisis de sesiones guardadas.

### 3.4. Integración con LSL y captura de marcadores

Se necesita un flujo robusto entre la adquisición del Unicorn y otra máquina para visualización, análisis y etiquetado. La propuesta de LSL debe pasar de ser un “modo de prueba” a un canal operativo real.

Cambios necesarios:

- Crear un flujo de streaming de EEG en LSL:
  - `EEG_Unicorn` con canales del headset;
  - `Markers` para eventos de estímulo y P300;
  - `Session_Metadata` con ID de usuario, tipo de sesión y configuración.
- Implementar un script de receptor LSL con validación de:
  - streams activos;
  - latencia;
  - pérdida de muestras;
  - sincronización entre EEG y marcadores.
- Integrar un “marker de P300” detectado por ML o por modelo estadístico, no solo por el estímulo visual.
- Guardar el conjunto completo de datos de la sesión:
  - señal cruda;
  - canales filtrados;
  - epochs;
  - marcas de estímulo;
  - decisiones del sistema;
  - metadata del usuario y del entorno.

### 3.5. Registro completo de la sesión

El proyecto ya guarda algunos logs, pero no hay un enfoque holístico de sesión. Para un sistema biométrico, cada intento debe ser completamente auditable.

Se debe registrar:

- identidad del usuario o intento;
- tipo de sesión: enrolamiento, autenticación, impostor, validación;
- información del headset y configuración;
- fecha y hora;
- número de estímulos y secuencia;
- canales usados y calidad;
- tiempos de evento;
- matriz de respuesta por imagen;
- scores del modelo y decisiones por clase;
- resultados del P300 / DDM;
- artefactos, reintentos y avisos de calidad.

### 3.6. Modelo DDM y decisión probabilística

El uso de umbrales simples sé puede mostrar en prototipo, pero para un sistema real de autenticación es insuficiente. La lógica de decisión debe pasar a un enfoque probabilístico.

Recomendación:

- Usar un modelo de decisión basado en DDM (Drift Diffusion Model) o un modelo bayesiano compacto que combine:
  - amplitud del P300;
  - latencia;
  - SNR;
  - confiabilidad por canal;
  - número de trials;
  - patrón de respuesta del usuario.
- Diferenciar dos decisiones:
  - probabilidad de que el estimulo sea target;
  - probabilidad de que la sesión sea genuina vs impostor.
- Incorporar un rechazo explícito: si el score es ambiguo, no otorgar acceso.

Esto mejora la robustez frente a FPR/FAR y permite mejores métricas de seguridad.

---

## 4. Fases del trabajo

### Fase 0: definición de requisitos y modelo de riesgo

Objetivo: definir qué se quiere proteger y qué significa autenticación segura.

Tareas:

- definir identidad de usuario;
- definir modelo de fraude y ataques;
- definir criterios de aceptación biométrica;
- definir métricas FAR, FRR, EER y rechazo;
- definir datos sensibles y procesos de anonimización.

Salida esperada:

- especificación funcional de autenticación cerebral;
- políticas de privacidad y protección; 
- reglas de seguridad para almacenamiento y sesión.

### Fase 1: infraestructura de enrolamiento y perfiles biométricos

Objetivo: construir la base de la identidad del usuario.

Tareas:

- crear el flujo de enrolamiento;
- definir la plantilla del usuario;
- guardar la referencia biométrica cifrada;
- definir la comparación de sesión vs plantilla;
- preparar la base para autenticación repetible.

Salida esperada:

- perfil de usuario con template biométrico;
- flujo estable de enrolamiento y validación;
- sistema de almacenamiento seguro.

### Fase 2: adquisición, calidad y visualización

Objetivo: ajustar el canal de adquisición y validación de la señal.

Tareas:

- construir visualizador de canales del Unicorn;
- estimar calidad por canal;
- recolectar y conservar señales en bruto y epochs;
- integrar LSL y flujo de marcador;
- añadir revisión en otra máquina sin perder sincronización.

Salida esperada:

- visualización del EEG en tiempo real;
- control de calidad por canal;
- pipeline LSL robusto para streaming y análisis.

### Fase 3: detección P300 y clasificación ML

Objetivo: transformar la señal en evidencia de autenticación.

Tareas:

- segmentar epochs por estímulo;
- filtrar y corregir la línea base;
- construir promedio de target y no target;
- entrenar modelo de clasificación con validación honesta;
- añadir detección automática de P300 con ML y/o análisis estadístico.

Salida esperada:

- detector P300 con métricas reproducibles;
- modelo de individuo o de patrón de respuesta;
- flujo de clasificación estable para producción.

### Fase 4: autenticación final con DDM

Objetivo: reemplazar decisiones binarias simples por decisiones probabilísticas.

Tareas:

- modelar la evidencia acumulativa de la P300;
- incorporar latencia, amplitud y ruido;
- definir umbrales de confianza;
- distinguir autenticar vs rechazar vs pedir desafío adicional.

Salida esperada:

- pipeline de decisión DDM;
- métrica final de confianza;
- mejor control de FAR/FRR.

### Fase 5: auditoría, sesión y validación

Objetivo: dejar el sistema listo para pruebas, trazabilidad y observación.

Tareas:

- grabar sesión completa y eventos;
- crear informes de calidad y decisiones;
- preparar base de validación con impostores;
- verificar métricas y robustez operativa.

Salida esperada:

- evidencias de autenticación por sesión;
- métricas de seguridad y calidad;
- perfiles de riesgo y de validación.

---

## 5. Estructura tentativa del repositorio

Se recomienda reorganizar el repositorio para separar claramente adquisición, análisis, decisión, seguridad y UI.

```text
Soude/
├── README.md
├── requirements.txt
├── pyproject.toml                # opcional, para packaging y dependencias
├── .env.example
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ui/
│   │   ├── auth_view.py
│   │   ├── enrollment_view.py
│   │   ├── live_eeg_view.py
│   │   └── result_view.py
│   └── workflows/
│       ├── enrollment_flow.py
│       ├── auth_flow.py
│       └── session_manager.py
│
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
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
│   │   ├── marker_manager.py
│   │   └── sync.py
│   └── simulator/
│       ├── mock_unicorn.py
│       └── synthetic_p300.py
│
├── signal/
│   ├── __init__.py
│   ├── filters.py
│   ├── epochs.py
│   ├── p300_detector.py
│   ├── feature_extractor.py
│   └── quality_metrics.py
│
├── biometrics/
│   ├── __init__.py
│   ├── user_profile.py
│   ├── template_builder.py
│   ├── comparison.py
│   └── enrollment.py
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
│   ├── models/
│   ├── templates/
│   └── datasets/
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_quality.py
│   ├── test_lsl.py
│   ├── test_biometric_profile.py
│   └── test_ddm.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PLAN_REDISEÑO_SOUDE.md
│   ├── SECURITY.md
│   └── VIDEO_SCRIPT.md
│
├── tools/
│   ├── eeg_visualizer.py
│   ├── debug_viewer.py
│   ├── lsl_sender.py
│   ├── lsl_monitor.py
│   └── export_session.py
│
└── scripts/
    ├── train_model.py
    ├── validate_auth.py
    └── benchmark_dataset.py
```

El objetivo de esta división es evitar mezclar hardware, ML, seguridad y UI en un solo paquete. La arquitectura debe reflejar funciones diferentes y evitar que el sistema crezca como un prototipo monolítico.

---

## 6. Funcionamiento propuesto del proyecto

El flujo funcional ideal para Soude debe ser el siguiente:

1. Registro del usuario
   - el usuario crea un perfil;
   - se le asigna una contraseña mental o conjunto de estímulos;
   - el sistema captura varias sesiones de enrolamiento;
   - se extrae la firma neural del usuario.

2. Generación de la plantilla biométrica
   - se procesan epochs objetivos y no objetivo;
   - se calcula latencia, amplitud y distribución espacial;
   - se generan features de identidad y de respuesta P300;
   - se guarda una plantilla protegida y cifrada.

3. Autenticación
   - se presenta el paradigma visual al usuario;
   - el sistema registra la evidencia EEG y los marcadores;
   - el flujo LSL captura y sincroniza la señal y los eventos;
   - se extraen features y se computa la probabilidad de target y genuino.

4. Decisión final
   - se usa P300 + DDM + score biométrico para producir un resultado final;
   - si la confianza es alta, se concede acceso;
   - si es baja o ambigua, se rechaza o se solicita verificación adicional.

5. Registro y auditoría
   - se guarda la sesión completa con visualización posible;
   - se deja evidencia para evaluaciones posteriores de FAR/FRR, rendimiento y seguridad.

---

## 7. Recomendaciones específicas para este proyecto

### 7.1. No usar “contraseña” como si fuera un secreto de texto

La contraseña cerebral debe ser una elección personal con una huella neural asociada, no solo un conjunto de imágenes. Debe existir un enfoque que mezcle:

- imagen seleccionada por usuario;
- patrón de respuesta del usuario a esa imagen;
- perfil biométrico del individuo;
- decisión probabilística final.

### 7.2. Separar evidencias de entrenamiento y evidencias de autentificación

Nunca mezclar la base de datos de entrenamiento con los perfiles de usuarios reales. Las sesiones de pruebas deben estar totalmente separadas de la autenticación en producción.

### 7.3. Guardar datos con trazabilidad

La trazabilidad debe permitir reconstruir una sesión completa. Eso incluye:

- señal cruda;
- señales filtradas;
- eventos y marcas;
- resultado del análisis;
- metadata del entorno y del usuario;
- versión del modelo y del algoritmo utilizado.

### 7.4. Mejorar la validación de calidad

No basta con “ver una gráfica”. Hace falta un control objetivo de calidad que impida aceptar una sesión con artefactos, mal contacto o ruido excesivo.

### 7.5. Diseñar para evaluación experimental

El producto debe poder evaluarse con:

- FAR; 
- FRR;
- EER;
- Tasa de rechazo por ruido;
- Robustez ante impostores;
- Diferentes condiciones de iluminación, usuario, electrodos y entorno.

---

## 8. Recomendación final

El proyecto actual tiene un gran valor como prototipo y demostrador de P300, pero aún no está diseñado como un sistema auténtico de autenticación cerebral. La siguiente evolución debe enfocarse en tres pilares:

1. identidad biométrica real del usuario;
2. seguridad y privacidad de datos;
3. validación operativa con canales, marcadores, sesiones y decisión probabilística.

Si se ejecuta esta transformación, Soude dejará de ser solo un detector de eventos P300 y se convertirá en una línea de investigación y desarrollo con enfoque real en autenticación cerebral segura, reproducible y auditable.

---

## 9. Siguientes pasos recomendados

1. Definir la arquitectura final y los módulos de seguridad.
2. Reorganizar el repositorio según la estructura propuesta.
3. Implementar perfil biométrico + flujo de enrolamiento.
4. Añadir calidad de señal, LSL y visualización completa.
5. Integrar clasificación P300 y DDM.
6. Grabar y auditar la sesión completa de cada intento.
7. Validar con métricas FAR/FRR/EER y pruebas de impostor.

Este plan debe servir como base de trabajo para la siguiente iteración del proyecto y como referencia para definir tareas y entregables del equipo.
