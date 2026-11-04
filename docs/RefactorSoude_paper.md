# Plan de rediseño de Soude (v2 + v3): alcance del paper y arquitectura objetivo

## 1. Alcance y decisión de alcance

Este documento concentra el rediseño que se tomará como base para el paper en los próximos 11 semanas. Se considera únicamente la visión ejecutable de la v2 y la arquitectura objetivo de la v3.

Se desestima el alcance completo de la v1 como objetivo inmediato porque:

- no se puede ejecutar en 11 semanas sin comprometer la calidad del estudio;
- el paper necesita un sistema reproducible, medible y validable, no una arquitectura teórica completa;
- la versión v2 ya define una línea base realista para un estudio EMBC y la v3 resuelve la estructura a largo plazo sin bloquear la recolección.

Por tanto:

- v1: referencia futura / arquitectura ideal de evolución;
- v2: plan ejecutable para el paper y la recolección;
- v3: arquitectura objetivo post-paper, con estructura limpia y gobernanza de código, pero sin invadir el alcance inmediato.

---

## 2. Protocolo experimental y alcance del paper

### 2.1. Hipótesis

La hipótesis principal del estudio es que la combinación de conocimiento del estímulo y la respuesta EEG del usuario permite distinguir entre autenticación genuina y tentativa de suplantación con mayor robustez que una regla basada únicamente en amplitud del P300.

Se evaluará la hipótesis mediante un diseño experimental con tres tipos de sesión:

- enrolamiento (S1): adquisición de la plantilla del usuario;
- autenticación genuina (S2): el usuario legítimo intenta acceder con su contraseña mental;
- impostor (S3): un atacante o participante no autorizado intenta acceder con la misma contraseña visual, y en un segundo escenario también intenta cumplir la tarea de atención, pero sin ser el usuario legítimo.

### 2.2. Pregunta de investigación

La investigación responde a las siguientes preguntas centrales:

1. ¿Qué tan bien discrimina Soude entre autenticación genuina y impostor ingenuo cuando el atacante no conoce el patrón correcto? (T1)
2. ¿Qué tan bien discrimina Soude entre autenticación genuina y impostor informado cuando el atacante conoce la contraseña y la atiende? (T2)
3. ¿Qué mejora aporta el componente biométrico (plantilla y patrón individual) respecto a la regla de amplitud actual del P300?
4. ¿Cuánta pérdida de rendimiento se produce por mala calidad de señal, rechazo de artefactos y rechazo por sesión?

### 2.3. Alcance del claim

El paper no afirma que Soude es un sistema biométrico definitivo ni un sistema de autenticación generalizable a producción. El claim que sí puede sostenerse con rigor es:

> Soude es un prototipo funcional de autenticación EEG basada en P300 que combina reconocimiento visual de imagen objetivo y patrón cerebral individual, y muestra potencial discriminativo frente a impostores en un protocolo experimental controlado con 8 canales y una arquitectura mínima de adquisición y análisis.

---

## 3. Diseño experimental y protocolo del estudio

### 3.1. Participantes

Se pretende incluir un conjunto de participantes adultos sanos, sin antecedentes de trastornos neurológicos, sin epilepsia ni uso reciente de fármacos psicoactivos que puedan afectar la señal EEG, con visión normal o corregida y sin contraindicación para la colocación del casco.

Se recomienda un objetivo de 20 a 30 participantes completos, con un margen para pérdidas por calidad de señales, incomodidad o deserción experimental. El número final debe justificarse en el paper como un estudio piloto controlado con protocolo bien definido y con intervalos de confianza por bootstrap.

Criterios de inclusión:

- edad adulta y capacidad para prestar consentimiento informado;
- disponibilidad para al menos dos sesiones de captura;
- capacidad para cumplir la tarea de atención visual y la estimulación P300.

Criterios de exclusión:

- migraña o convulsiones activas;
- uso de sustancias que alteren la atención o la actividad cortical durante la sesión;
- mala calidad de contacto del electrodos o imposibilidad de realizar la tarea.

### 3.2. Esquema de sesiones

Cada participante completa un protocolo con tres tipos de sesión:

- S1: enrolamiento. El sujeto observa y atiende a su contraseña mental para construir la plantilla biométrica.
- S2: autenticación genuina. El mismo sujeto repite la tarea con su contraseña mental para evaluar FRR y capacidad de reproducibilidad.
- S3: autenticación impostora. Se evalúan dos escenarios: impostor ingenuo (sin conocimiento de la contraseña) e impostor informado (que conoce la contraseña y la atiende).

### 3.3. Procedimiento experimental

El protocolo usa una pantalla con 20 estímulos visuales sobre fondo constante y única fuente visual para minimizar artefactos visuales no cognitivos. Cada imagen corresponde a un símbolo distinto y la contraseña del usuario se define con 3 símbolos específicos.

Cada bloque experimental incluye:

- 20 imágenes en secuencia aleatoria;
- 3 imágenes objetivo, repetidas cinco veces cada una;
- 17 distractores, repetidos cinco veces cada uno;
- presentación de cada estímulo con `SOA = 0.5 s` y `BLANK = 0.075 s`;
- duración aproximada por bloque de `57.5 s`.

Con esta configuración, el estudio mantiene una tasa objetivo cercana al 15%, que es la base del paradigma oddball visual.

### 3.4. Adquisición y captura

La señal se registra con 8 canales EEG a 250 Hz usando el mismo formato que el repositorio actual:

- Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
- ventana post-estímulo de 800 ms
- línea base de 100 ms previa al estímulo
- ventana de interés P300 entre 250 ms y 500 ms

La captura se realiza en bruto y luego se procesa en epoching con filtrado 1–10 Hz y notch de 60 Hz. Los eventos de marcador se registran con `image_id`, `buffer_index`, `timestamp` e `is_target` para reconstruir la secuencia de estímulos y asegurar la sincronía.

### 3.5. Rechazo de sesiones y calidad de señal

Antes de aceptar una sesión como válida, se aplica una compuerta de calidad basada en:

- varianza por canal;
- frecuencia de artefactos;
- pico a pico por canal;
- proporción de epochs rechazadas;
- sincronía entre marcadores y muestras;
- número mínimo de epochs válidas por clase.

Las sesiones que no cumplan estos criterios no se consideran para la autenticación final y se cuentan en la tasa de fallo de adquisición.

### 3.6. Variables experimentales

Variables principales:

- amplitud media del P300 en la ventana objetivo;
- diferencia target vs non-target;
- latencia de pico;
- topografía por canal;
- SNR por sesión;
- número de epochs válidas;
- similitud de la respuesta frente a la plantilla del usuario;
- decisión final (`ACCEPT`, `REJECT`, `RETRY`, `CONTINUE`).

Variables de control:

- posición del electrodo;
- calibre de la señal;
- nivel de ruido de línea;
- tipo de sesión;
- versión del algoritmo y configuración del protocolo.

### 3.7. Consideraciones éticas y de privacidad

El protocolo debe ejecutarse bajo consentimiento informado, con pseudonimización del sujeto y almacenamiento de los datos fuera del repositorio. Las sesiones se tratan como datos sensibles y se cifran en reposo. La relación entre identidad real y pseudónimo queda fuera del código y de los artefactos del proyecto.

---

## 4. Estado real del repositorio como línea base

La versión actual del repositorio ya tiene una base funcional que debemos preservar:

| Elemento | Estado real |
|---|---|
| Regla de decisión | Doble criterio en `AuthenticationPipeline.evaluate()`: `delta >= 1.5 µV` y `delta >= 1.5 × pre_sigma_uv`, con polaridad positiva |
| Métricas | `analysis/` ya calcula FAR, FRR, ROC, EER y grand-average por grupo |
| Calidad de señal | `signal_quality.py` mide varianza por canal y hay rechazo por artefacto en la extracción de épocas |
| Estímulos | 20 símbolos, mismo fondo, 3 objetivos × 5 repeticiones + 17 × 5 = 100 eventos, SOA 0.5 s + 75 ms en blanco |
| Identidad | No existe `subject_id` formal; el tipo de sesión se maneja con `SOUDE_SESSION_TYPE` y manualmente en `labels.csv` |
| Enrolamiento | Es una demostración de selección visual de 3 imágenes, no una plantilla biométrica persistente real |
| `ml/` | Existe un flujo SVM / Rest-Passthought, pero no es el eje del paper |
| Datos de sesión | Los logs actuales sirven para depuración y análisis, no para la licencia del paper |

### 3.1. Contrato actual de datos que debe conservarse

La refactorización que haga el paper no debe romper el contrato funcional ya implementado:

- frecuencia de muestreo: 250 Hz
- canales: 8 (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)
- ventana epoch: 800 ms = 200 muestras
- línea base: 100 ms = 25 muestras
- ventana útil P300: 250–500 ms
- `StimulusMarker {image_id, buffer_index, timestamp, is_target}`
- `AuthResult {granted, delta_uv, pre_sigma_uv, snr_db, n_target, n_nontarget}`

Esto es clave para no romper la compatibilidad del flujo actual mientras se implementa la versión final.

---

## 5. Modelo de amenaza para el paper

| ID | Atacante | Cobertura en el estudio |
|---|---|---|
| T1 | Impostor ingenuo: no conoce la contraseña | Sí, se evalúa |
| T2 | Impostor informado: conoce la contraseña y la atiende | Sí, se evalúa |
| T3 | Inyección de EEG sintético o manipulación de marcadores | Mitigación por diseño y discusión, pero no foco principal |

La calidad de señal o la fatiga no se convierten automáticamente en ataque; se tratan como causa de rechazo seguro o reintento.

---

## 6. Arquitectura incremental del alcance del paper

La regla esencial es: hacer una mejora real sin reestructurar todo el proyecto de golpe.

```text
neurolock/
├── brain_engine.py          (preservar contrato de adquisición)
├── filters.py               (preservar filtro 1–10 Hz + notch 60 Hz)
├── signal_processing.py     (baseline actual; se mantiene como línea base)
├── signal_quality.py        (extender con QualityGate)
├── stimulus_runner.py       (preservar paradigma oddball)
├── data_logger.py           (extender con session.json y metadata)
├── session_schema.py        (NUEVO: subject_id, session_number, block_type)
├── features.py              (NUEVO: features por época y por sesión)
├── biometrics/              (NUEVO)
│   ├── template.py          (plantilla del usuario)
│   ├── enrollment.py        (flujo multibloque)
│   └── comparator.py        (similitud sesión–plantilla)
├── decision.py              (NUEVO: acumulación secuencial + política)
├── security/                (NUEVO, mínimo funcional)
│   ├── crypto.py            (AES-256-GCM)
│   └── keystore.py          (clave fuera del repo)
├── ui/                     (adaptar flujo sin rediseño visual completo)
├── analysis/                (extender evaluación y bootstrap)
├── tests/                   (agregar pruebas del nuevo módulo)
└── legacy/                  (adaptador temporales si hace falta)
```

### 5.1. Regla de no ruptura

No se reestructura la arquitectura completa a la raíz. Lo que se hace es extender `neurolock/` con capas nuevas y mantener `run.py` operativo y `tests/test_pipeline.py` pasando.

---

## 7. Modelo de decisión para el paper

### 6.1. Dos puntuaciones

1. Puntuación de conocimiento (K): evidencia de la discriminación objetivo vs no objetivo, con un clasificador por época o con una regla de evidencia acumulada.
2. Puntuación de identidad (B): similitud entre la respuesta actual y la plantilla biométrica del usuario.

Se usa un enfoque secuencial basado en evidencia acumulada, pero no se habla de DDM como modelo principal porque la respuesta no es un proceso de decisión conductual; lo correcto para este paper es modelar evidencia acumulada con criterios probabilistas y de política.

### 6.2. Política final

Los estados de decisión deben ser:

- ACCEPT
- REJECT
- RETRY
- CONTINUE

Esto es clave porque un sistema biométrico real no debe decidir de forma rígida sobre una señal de baja calidad.

### 6.3. Umbrales y validación

Los umbrales deben fijarse con datos de enrolamiento y nunca ajustarse con datos de evaluación. La evaluación debe resumirse con:

- FAR para T1 y T2 por separado
- FRR
- EER
- ROC/AUC
- tasa de fallo de adquisición
- tiempo de decisión

La contribución del componente biométrico (B) se mide como reducción del FAR en T2.

---

## 8. Componentes que sí se implementan para el paper

### 7.1. Esquema de sesión (`session_schema.py`, `data_logger.py`)

Campos mínimos que debe tener cada sesión:

- `session_id`
- `subject_id` (pseudónimo)
- `session_number` (1 a N)
- `block_type` (`enrollment`, `genuine`, `impostor_naive`)
- `software_version`
- `config` del paradigma
- `quality` por canal
- `decision`
- `scores` (K, B, evidencia)

Esto reemplaza la lógica informal de `SOUDE_SESSION_TYPE` y `labels.csv` como fuente de verdad.

### 7.2. QualityGate

Reglas mínimas para el paper:

- varianza por canal dentro de rango válido
- proporción de artefactos por bloque bajo máximo
- mínimo de épocas válidas por clase
- sincronía marcador–muestra dentro de tolerancia

La salida debe ser `OK`, `RETRY` o `ABORT`, con motivo. Esto cuenta la tasa de fallo de adquisición del sistema.

### 7.3. Enrolamiento y plantilla

Se implementa un flujo multibloque y una plantilla protegida con los siguientes elementos:

- vector promedio objetivo por canal y ventana
- estadísticos de ruido y línea base
- versión del algoritmo
- metadata del sujeto y del protocolo

La plantilla no contiene EEG crudo. Se guarda cifrada en reposo con AES-256-GCM. El papel no requiere una solución avanzada de biometría cancelable; sí una solución segura y reproducible.

### 7.4. Seguridad mínima funcional

Se implementan los controles mínimos que son importantes para el paper:

- plantilla cifrada en reposo
- clave fuera del repositorio
- pseudonimización de sujeto
- logs con metadata y decisión final sin exposición de información innecesaria

No se hace una arquitectura completa de seguridad empresarial, pero sí lo necesario para ser defendible metodológicamente.

### 7.5. UI y operación

Sin rediseño visual completo. Los cambios mínimos son:

- capturar `subject_id` y `session_number`
- seleccionar `block_type`
- mostrar mensajes de estado `ACCEPT/REJECT/RETRY`
- permitir enrolamiento multi-bloque

Esto es un cambio funcional y operativo, no un rediseño total de UX.

---

## 9. Arquitectura objetivo v3 (post-paper)

La v3 define la estructura correcta a largo plazo, que debe quedar documentada para el siguiente ciclo de desarrollo.

### 8.1. Estructura final recomendada

```text
soude/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
│
├── src/soude/
│   ├── core/
│   │   ├── contracts.py
│   │   ├── constants.py
│   │   ├── config.py
│   │   ├── time_sync.py
│   │   └── logging_config.py
│   │
│   ├── acquisition/
│   │   ├── base.py
│   │   ├── unicorn/
│   │   ├── lsl/
│   │   └── simulator/
│   │
│   ├── paradigm/
│   │   ├── catalog.py
│   │   ├── sequence.py
│   │   └── runner.py
│   │
│   ├── dsp/
│   │   ├── filters.py
│   │   ├── epochs.py
│   │   ├── artifacts.py
│   │   └── averaging.py
│   │
│   ├── quality/
│   │   ├── channel_metrics.py
│   │   ├── gate.py
│   │   └── sync_check.py
│   │
│   ├── features/
│   │   ├── erp_features.py
│   │   └── windowed.py
│   │
│   ├── modeling/
│   │   ├── epoch_classifier.py
│   │   ├── scoring.py
│   │   └── validation.py
│   │
│   ├── biometrics/
│   │   ├── enrollment.py
│   │   ├── template.py
│   │   ├── comparator.py
│   │   └── profile.py
│   │
│   ├── decision/
│   │   ├── sprt.py
│   │   └── policy.py
│   │
│   ├── security/
│   │   ├── crypto.py
│   │   ├── keystore.py
│   │   ├── vault.py
│   │   └── access_policy.py
│   │
│   ├── sessions/
│   │   ├── schema.py
│   │   ├── recorder.py
│   │   ├── export.py
│   │   └── retention.py
│   │
│   ├── flows/
│   │   ├── auth_flow.py
│   │   └── enrollment_flow.py
│   │
│   └── app/
│       ├── cli.py
│       └── ui/
│
├── analysis/
├── tools/
├── scripts/
├── tests/
├── docs/
├── examples/
└── legacy/
```

### 8.2. Qué queda para después del paper

Se dejan fuera del alcance del paper y se documentan como futuros módulos o extensiones:

- rotación y gestión avanzada de secretos
- roles y permisos de acceso
- gestiones de drift y actualización de plantillas
- streams LSL completos con four-stream model
- biometría cancelable o protección avanzada de plantilla
- soporte multi-dispositivo y recolección masiva

---

## 10. Evaluación y análisis para el paper

### 9.1. Modo de evaluación

Se trabaja con protocolo predefinido:

- S1: enrolamiento
- S2 y S3: evaluación operativa
- Datos de S1 no se usan para evaluar el mismo sistema
- Se documenta el procedimiento antes de sacar resultados

### 9.2. Métricas obligatorias

- FAR para T1
- FAR para T2
- FRR
- EER
- ROC/AUC
- tasa de fallo de adquisición
- tiempo de decisión
- bootstrap por participantes

### 9.3. Análisis de ablaición

Se comparan al menos tres modos:

1. Baseline actual: regla de amplitud doble criterio
2. K: evidencia de conocimiento y clasificación por época
3. K + B: evidencia de conocimiento + identidad biométrica

La serie de resultados debe mostrar claramente si la parte biométrica aporta valor real o si solo ayuda marginalmente.

---

## 11. Pruebas requeridas para el paper

Se deben mantener y ampliar estas pruebas:

- `test_pipeline.py` existente
- test de esquema de sesión
- test de QualityGate con canal caído, ruido y marcadores desalineados
- test de plantilla cifrada y lectura correcta
- test de comparador genuino > impostor sintético
- test de decisión secuencial y límite de `CONTINUE`
- test de evaluación con datos sintéticos y bootstrap

Se usa `MockUnicorn` para producir datos sintéticos navegables sin depender del hardware físico.

---

## 12. Plan operativo de 11 semanas

| Semana | Objetivo | Entregable |
|---|---|---|
| 1 | Congelar línea base, definir protocolos y alcance | versión base del paper |
| 2 | Contratos y esquema de sesión | `session_schema.py` + tests |
| 3 | Calidad + DSP | `QualityGate` y validación técnica |
| 4 | Features + plantilla + comparador | biometría funcional sobre datos sintéticos |
| 5 | Congelar la rama del paper (`paper/embc`) | versión estable para recolección |
| 6–8 | Recolección de datos | S1–S3 con participantes |
| 8–9 | Análisis y figuras | métricas, curvas, tablas |
| 10 | Redacción | borrador completo |
| 11 | Ajustes finales y envío | versión final |

### 11.1. Regla de congelado

En la semana 5, la rama del paper se congela. Todas las mejoras de arquitectura posterior se desarrollan en la rama de evolución (`main`), sin tocar la versión que se va a usar para análisis.

---

## 13. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| La calidad de señal es muy baja | Reportar como tasa de fallo de adquisición y reforzar la compuerta de calidad |
| La biometría discrimina poco | Reportarlo explícitamente como hallazgo del estudio; no convertirlo en fallo |
| El equipo se dispersa con demasiadas tareas | Mantener foco en las capas P, E y F del paper; la v3 queda como base futura |
| La reestructuración rompe la demo | Preservar `neurolock` y migrar por capas con tests |
| La recolección tarda | Priorizar un protocolo compacto y reproducible |

---

## 14. Diferencia entre v1, v2 y v3

### v1
- intención original: arquitectura ideal y completa
- enfoque: enfoque a futuro, teórico y muy amplio
- valor: útil como referencia arquitectónica para rediseño posterior
- uso real: no forma parte del alcance del paper

### v2
- enfoque: plan ejecutable para el paper
- valor: define la investigación, los modelos, la evaluación y la seguridad mínima funcional
- uso real: base del estudio EMBC

### v3
- enfoque: arquitectura target post-paper
- valor: estructura de desarrollo y gobernanza para el producto final
- uso real: rama de evolución y mejora técnica futura

---

## 15. Recomendación final

El paper debe construirse sobre la v2 como eje operativo, no sobre la v1. La v3 se toma como arquitectura objetivo de mediano/largo plazo y como hoja de ruta para la siguiente fase. Esta separación reduce riesgo, mantiene reproducibilidad y hace viable cumplir la fecha de entrega sin perder la visión técnica y biométrica del sistema.

En resumen:

- v2 = lo que realmente haremos para el paper
- v3 = lo que dejamos estructurado para la próxima etapa
- v1 = referencia futura, no parte del alcance inmediato
