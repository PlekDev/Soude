# Plan del artículo EMBC: Soude, autenticación EEG con P300 (11 semanas)

**Relación con otros documentos**

| Documento | Papel |
|---|---|
| `PLAN_REDISEÑO_SOUDE.md` | Plan completo de largo plazo. Este plan ejecuta un subconjunto y lo deja listo para continuar |
| `SECURITY.md` | Estado de cada control (`[P]`, `[E]`, `[F]`) |
| `PROTOCOLO_EXPERIMENTAL_SOUDE.md` | Diseño del estudio para el comité de ética |

**Leyenda de estado:** **[P]** se implementa y se usa en el artículo · **[E]** esqueleto con interfaz, documentación y un test de contrato, sin lógica completa · **[F]** futuro, solo documentado.

Los valores marcados con (†) deben coincidir con el protocolo de ética.

---

## 1. Pregunta de investigación

Soude es hoy un detector de P300 con una regla de decisión sobre la diferencia objetivo−no objetivo. Eso comprueba que el usuario **atiende a los símbolos correctos** (factor de conocimiento), no que **sea quien dice ser** (factor biométrico).

El artículo evalúa una **autenticación de dos factores con un EEG de 8 canales**:

- **Conocimiento:** el usuario reconoce y atiende a su contraseña mental (3 símbolos de 20).
- **Biometría:** la forma de su respuesta coincide con la plantilla enrolada.

**Preguntas que se responden con datos:**

1. ¿Rechaza el sistema a un impostor que **no conoce** la contraseña? (T1)
2. ¿Rechaza a uno que **sí la conoce** y la atiende? (T2) ¿Cuánto aporta la plantilla frente a una regla solo de amplitud?
3. ¿Cuál es el costo en tasa de fallo de adquisición y en tiempo de autenticación?

Si la discriminación biométrica resulta modesta con 8 canales y ~15 épocas objetivo por bloque, **es un resultado válido** y se reporta como tal. El artículo no depende de que la plantilla discrimine mucho.

### Amenazas

| ID | Atacante | En el estudio |
|---|---|---|
| T1 | Impostor ingenuo (no conoce la contraseña) | Bloques dedicados en S2 y S3 |
| T2 | Impostor informado (la conoce y la atiende) | Offline: sesión genuina de otro participante contra la plantilla evaluada (contraseña común) |
| T3 | Replay, inyección sintética, manipulación de marcadores | Mitigaciones básicas (secuencia aleatoria por intento y hash de duplicados), con prueba de replay offline. La evaluación avanzada es [F] |

La mala calidad de señal, fatiga o mala colocación se trata como **calidad** (compuerta), no como ataque.

---

## 2. Línea base (estado real del repositorio)

| Elemento | Estado |
|---|---|
| Regla de decisión | Doble criterio: `delta ≥ 1.5 µV` y `delta ≥ 1.5 × pre_σ`, polaridad positiva, mínimo 5 épocas por clase. Se conserva como **baseline** |
| Métricas | `analysis/` ya calcula FAR/FRR, ROC, EER, AUC y grand-average |
| Calidad | `signal_quality.py` mide varianza por canal, pero **no condiciona la decisión**. El rechazo por artefactos (p-p > 100 µV) está en la extracción |
| Estímulos | 20 símbolos con el mismo fondo y fuente; 3 objetivos × 5 + 17 × 5 = 100 estímulos; SOA 0.5 s + 75 ms en blanco (≈ 57.5 s) |
| Identidad | No existe `subject_id`. El tipo de sesión sale de `SOUDE_SESSION_TYPE` y `labels.csv` manual |
| Enrolamiento y Vault | Demostración (3 imágenes; contraseñas fijas en el código) |
| `ml/` | SVM Rest/Passthought: paradigma distinto, fuera del alcance |
| `logs/` | **Se descartan** para el artículo (5 sujetos con 5 genuinas y 5 de impostor cada uno). Se archivan cifrados fuera del repo solo para depuración y estimar tamaños de efecto |

**Contrato de datos que se conserva:** ventana de 0.8 s a 250 Hz = **200 muestras × 8 canales**, línea base de 100 ms, ventana P300 de 250–500 ms, `StimulusMarker{image_id, buffer_index, timestamp, is_target}` y `AuthResult{delta_uv, pre_sigma_uv, snr_db, n_target, n_nontarget}`.

---

## 3. Modelo de decisión del artículo

### 3.1. Dos puntuaciones

1. **Conocimiento (K).** Clasificador de época objetivo vs. no objetivo, específico del usuario (LDA con shrinkage sobre medias en ventanas de ~50 ms por canal, 0–800 ms), entrenado con las épocas de enrolamiento. Cada época aporta un log-cociente de verosimilitud; K es la evidencia acumulada.
2. **Identidad (B).** Similitud entre el promedio objetivo de la sesión (y/o la onda diferencia) y la plantilla enrolada, con una distancia o correlación espacio-temporal sobre los 8 canales.

### 3.2. Decisión secuencial

**SPRT** sobre la evidencia acumulada. No se usa "DDM", porque no hay respuesta conductual que modelar.

| Estado | Condición |
|---|---|
| `ACCEPT` | K y B superan sus umbrales con calidad válida |
| `REJECT` | Evidencia suficiente en contra |
| `RETRY` | Calidad insuficiente (no se decide) |
| `CONTINUE` | Evidencia insuficiente; más repeticiones hasta un máximo y, si persiste, `REJECT` |

Son estados de **política**, no probabilidades de un modelo.

**Alcance:** el paradigma tiene 5 repeticiones fijas, así que la parada temprana y `CONTINUE` se evalúan **offline** procesando las épocas en orden. La versión adaptativa en línea es [F].

### 3.3. Umbrales sin fuga de información

- Se determinan **solo con datos de enrolamiento (S1)**, con validación cruzada por bloque (genuino = bloque retenido; impostor = enrolamientos de otros participantes).
- S2 y S3 solo se usan para evaluar; no se ajusta ningún umbral después de verlos.
- Se reporta también el EER y la curva ROC con barrido de umbral, rotulados como cota optimista.

### 3.4. Comparaciones (ablación)

1. **Baseline:** regla actual de doble criterio.
2. **K:** clasificador de época + SPRT.
3. **K + B:** con plantilla biométrica.

La contribución de B es la reducción del FAR en T2.

---

## 4. Evaluación

- **Diseño:** plantilla con S1; prueba con S2 y S3; nunca se entrena y se evalúa con la misma sesión.
- **Contraseña común** para todos los participantes: la sesión genuina de cada uno puntuada contra la plantilla de otro es un intento T2 sin sesiones extra.
- **Métricas:** FAR (T1 y T2 por separado), FRR, EER, ROC/AUC, tasa de fallo de adquisición y tiempo de decisión.
- **Incertidumbre:** *bootstrap* remuestreando **participantes**. Los intentos T2 comparten participantes y no son independientes.
- **Plan de análisis congelado** antes de recolectar datos.
- **Sensibilidad:** número de bloques de enrolamiento y de épocas por decisión.
- **Prueba de replay offline (T3):** reenviar sesiones previas y comprobar que se rechazan por secuencia o por hash duplicado.
- Un segundo clasificador de época (por ejemplo, regresión logística) como análisis de sensibilidad es opcional.

El detalle de sujetos, sesiones y bloques está en el protocolo de ética: 20 participantes mínimo (30 deseable), 3 sesiones en días distintos, S1 enrolamiento, S2 y S3 autenticación con un bloque de impostor ingenuo.

---

## 5. Alcance por módulo

Estructura bajo `src/soude/` (ver el plan completo para el árbol y las reglas de dependencia). Estado en este plan:

| Módulo | Contenido | Estado |
|---|---|---|
| `core` | `contracts.py`, `constants.py`, `config.py`, `time_sync.py`, `logging_config.py` | [P] |
| `acquisition` | `base`, `unicorn/`, simulador con usuarios sintéticos, `lsl/receiver` + `validator` + `jitter` | [P] |
| `acquisition/lsl` | `sender`, `multi_stream_sync` (4 streams) | [E] |
| `paradigm` | catálogo, secuencia aleatoria por intento (generador criptográfico), runner | [P] |
| `dsp` | filtros, épocas, artefactos, promediado | [P] |
| `quality` | métricas por canal, compuerta (`OK/RETRY/ABORT`), chequeo de sincronía | [P] |
| `features` | features por época y por promedio de sesión | [P] |
| `modeling` | clasificador de época, puntuación, validación | [P] |
| `biometrics` | enrolamiento, plantilla, comparador, perfil | [P] |
| `biometrics/drift_policy` | actualización de plantilla | [E] |
| `decision` | SPRT, política | [P] |
| `security` | `crypto` (AES-256-GCM) y `keystore` | [P] |
| `security` | `vault`, `secret_store`, `access_policy`, `audit_log` | [E] |
| `sessions` | esquema, recorder, exportación, hash de duplicados | [P] |
| `sessions/retention` | borrado programado | [E] |
| `flows` | orquestadores de autenticación y enrolamiento | [P] |
| `app` | consola del experimentador, enrolamiento multibloque, resultados | [P] |
| `analysis` | evaluación entre sesiones, bootstrap, figuras (extiende lo existente) | [P] |

**Un módulo [E] solo tiene** interfaz, docstring, una ADR corta y un test que verifica su contrato.

### 5.1. Esquema de sesión

`session.json`: `session_id`, `subject_id` (pseudónimo), `session_number` (1–3), `block_type` (`enrollment` | `genuine` | `impostor_naive`), versión del software, parámetros del paradigma, versión de modelo y plantilla, calidad por canal, motivo de rechazo, decisión y puntuaciones K y B. El `block_type` lo fija el experimentador en la UI (se elimina `SOUDE_SESSION_TYPE`).

**Hecho cuando:** una sesión se puede reanalizar completa desde `session.json`, `markers.csv` y `epochs.npz`, sin información externa.

### 5.2. Compuerta de calidad

Reglas fijadas antes de recolectar: varianza por canal en rango (se reutiliza `ImpedanceChecker`), proporción de épocas rechazadas bajo un máximo, mínimo de épocas válidas por clase (hoy 5; confirmar con pruebas técnicas) y sincronía marcador–muestra dentro de tolerancia. Cada `RETRY/ABORT` cuenta para la tasa de fallo de adquisición.

**Hecho cuando:** hay tests con ruido, canal desconectado y marcadores desalineados que activan la regla correcta.

### 5.3. Plantilla y seguridad mínima

- La plantilla guarda parámetros del clasificador, vector promedio objetivo, estadísticos de ruido y versión; **no guarda EEG crudo**.
- Plantilla y lista de imágenes objetivo cifradas con AES-256-GCM, clave fuera del repositorio. El Vault de demostración se retira o se marca como demo.
- **Hecho cuando:** alterar un byte del archivo cifrado produce error de autenticación, y no hay claves ni listas de objetivos en texto plano en el repositorio.

### 5.4. Sincronización

Validar el stream (canales, frecuencia de muestreo, huecos, timestamps) y **medir jitter y latencia marcador–muestra**, como dato de métodos.

### 5.5. Pruebas

Se conserva `test_pipeline.py`. Nuevos tests: esquema de sesión, compuerta de calidad, plantilla (ida y vuelta cifrada), comparador (genuino sintético > impostor sintético), SPRT, evaluación (bootstrap con resultado conocido) y replay. Se usa `MockUnicorn` para generar P300 sintético con usuarios que difieren en latencia, amplitud y topografía.

---

## 6. Organización del equipo (5–9 integrantes)

| Flujo | Responsabilidad en el paper | Fusión con 5 personas |
|---|---|---|
| A. Integración y arquitectura | Contratos, revisiones, ramas, releases, CI | A (líder) |
| B. Adquisición y LSL | Dispositivo, simulador, jitter, validación de stream | B + C |
| C. DSP y calidad | Filtrado, épocas, artefactos, compuerta | B + C |
| D. Biometría y decisión | Features, clasificador, plantilla, SPRT | D |
| E. Seguridad y sesiones | Cifrado, esquema de sesión, exportación | E + F |
| F. UI y paradigma | Consola del experimentador, enrolamiento, estímulos | E + F |
| G. Evaluación y artículo | Métricas, bootstrap, figuras, redacción | G + H |
| H. Protocolo y operación | Ética, reclutamiento, sesiones | G + H |
| I. Calidad y documentación | Tests, datos sintéticos, ADRs | repartido |

Cada flujo tiene un responsable y un suplente.

**Prácticas:** rama por tarea, *pull request* con una revisión (dos en `core/contracts`, `security` y `decision`), `CODEOWNERS`, CI obligatorio y una ADR por decisión de arquitectura.

**Cuellos de botella fijos:** la aprobación de ética, el número de dispositivos Unicorn y las sesiones de recolección no se aceleran con más personas. Por eso H arranca en la semana 1.

**Usos opcionales del equipo (sin tocar el núcleo):** reclutar y recolectar en paralelo (meta de 30) si hay más de un dispositivo; prueba de replay offline; segundo clasificador de época como sensibilidad. Cualquier medición adicional con participantes (por ejemplo, un cuestionario de usabilidad) requiere una enmienda al protocolo.

---

## 7. Ramas y congelamiento

- Etiqueta `v0-hackathon` del estado actual.
- **`paper/embc`** se congela en la semana 5 (etiqueta `v1.0-paper`). Los datos se recolectan con esta versión. Solo entran correcciones de errores con revisión del líder.
- **`main`** continúa con los módulos [E] y el rediseño final desde la semana 5, sin bloquear al paper.
- Cada `session.json` registra la versión del software.

Migración (estrangulamiento): se crea `src/soude/` con los contratos; se migra de abajo hacia arriba (`acquisition` → `dsp` → `quality` → `paradigm`) dejando adaptadores en `legacy/neurolock/`, y los tests existentes siguen en verde. El código no migrado sigue funcionando.

### Contratos congelados en la semana 2

`StimulusMarker`, `Epoch{data (200×8), marker, valid, reject_reason}`, `QualityReport{channel_status, rejected_ratio, sync_ok, verdict, reasons}`, `FeatureSet`, `AuthDecision{state, k_score, b_score, n_epochs, reasons}` y `SessionMeta`. No cambian sin una ADR.

---

## 8. Cronograma de 11 semanas

| Sem. | Entregable global | Criterio de salida |
|---|---|---|
| 1 | Etiqueta del repositorio, esqueleto y CI, archivo cifrado de `logs/`, amenazas y diseño experimental | Protocolo listo para enviar |
| 2 | Envío a ética. **Contratos congelados**, `session_schema`, `crypto`, migración de `acquisition` y `dsp` | Pipeline de extremo a extremo con simulador |
| 3 | `quality/gate`, features, plantilla y comparador sobre datos sintéticos, jitter y validador LSL | Usuarios sintéticos separados |
| 4 | Clasificador de época, SPRT offline, plantilla cifrada ida y vuelta, marco de evaluación con bootstrap | Pipeline offline reproducible |
| 5 | Enrolamiento y consola en la UI, umbrales de calidad calibrados con pruebas técnicas. **Congelar `paper/embc` y el plan de análisis** | Simulacro de sesión completo sin intervención manual |
| 6 | Pruebas técnicas del equipo y arranque de recolección (con aprobación) | Primeros participantes con S1 |
| 7–8 | Recolección | S1–S3 de ≥ 20 participantes |
| 8–9 | Análisis y figuras | Tablas con intervalos |
| 10 | Redacción completa | Borrador completo |
| 11 | Colchón y envío | Enviado |

### Reparto por flujo (resumen)

| Flujo | Sem. 1–5 | Sem. 6–11 |
|---|---|---|
| A | Esqueleto, contratos, revisiones, integración de `flows` | Mantiene `main`; apoya el artículo |
| B | Migración de `acquisition`, simulador, jitter y validador | Opera dispositivos en la recolección |
| C | Migración de `dsp`, compuerta de calidad y calibración | `main`: módulos [E] → [P] |
| D | Features, plantilla, clasificador, SPRT | `main`: biometría final |
| E | Esquema de sesión, cifrado, exportación | `main`: bóveda y política |
| F | Consola del experimentador, enrolamiento multibloque | `main`: UI final |
| G | Marco de evaluación, plan de análisis congelado | Análisis, figuras y redacción |
| H | Protocolo, ética, guía del experimentador, simulacro | Recolección y cierre |

---

## 9. Criterios de "listo" por hito

- **Contratos (sem. 2):** todos los flujos ejecutan un pipeline de extremo a extremo con datos del simulador, aunque sea con componentes triviales.
- **Congelamiento (sem. 5):** tests y CI en verde, simulacro de sesión completo (enrolamiento + autenticación + exportación) sin intervención manual, y análisis offline que reproduce resultados sobre datos sintéticos.
- **Fin de recolección (sem. 8):** ≥ 20 participantes con S1–S3 completas, o un registro justificado de las que faltan.
- **Entrega del artículo (sem. 11):** resultados con intervalos de confianza, plan de análisis tal como se congeló y versión del software citada.

---

## 10. Riesgos y alternativas

| Riesgo | Alternativa |
|---|---|
| Ética tarda más de lo previsto | Desarrollar con datos sintéticos y `logs/` archivados; comprimir la recolección a 2 semanas; reducir a 2 sesiones declarando la limitación |
| La plantilla discrimina poco | Reportar como resultado: el conocimiento protege frente a T1 y el factor biométrico queda como mejora acotada en T2 |
| Calidad de señal baja o muchos rechazos | Revisar el protocolo de colocación; reportar la tasa de fallo de adquisición |
| Menos de 20 participantes completos | Reclutar ~25 para cubrir deserción; si aún no alcanza, presentar como estudio piloto con intervalos amplios |
| Alguien modifica el sistema durante la recolección | Rama `paper/embc` protegida y versión en cada `session.json` |
| Conflictos de integración | Contratos congelados, reglas de dependencia en CI, ramas cortas |
| Tiempo insuficiente | Mantener el núcleo (5.1–5.3 y la evaluación); recortar sincronización y pulido de UI |

---

## 11. Seguimiento posterior al artículo

Al cerrar el artículo se continúa con el plan completo. Entradas ya preparadas por este plan:

1. **Fase 7 del plan completo** (endurecimiento): pasar los módulos [E] a [P] en `main` (bóveda y rotación de claves, roles, bloqueo por intentos, auditoría a prueba de manipulación, retención automática, *drift* de plantilla, multi-stream LSL).
2. **Paradigma adaptativo en línea:** convertir la parada temprana de SPRT, hoy offline, en decisión durante la sesión.
3. **Protección de plantilla** (biometría cancelable o comparación en dominio cifrado) y evaluación avanzada de replay e inyección.
4. **Retiro de `legacy/neurolock/`** y alineación de `SECURITY.md` y `ARCHITECTURE.md` con el código.
5. **Lecciones del estudio:** revisar umbrales de calidad, tasa de fallo de adquisición y discriminación de B con los datos reales, y decidir qué cambia en el diseño final.

---

## Registro de cambios respecto a v2 y v3

- Se fusionaron el plan v2 (modelo, evaluación, riesgos) y el v3 (estructura, ramas, equipo).
- La estructura completa de carpetas y las reglas de dependencia se movieron al plan completo; aquí solo se indica el estado de cada módulo.
- Se agregó la sección de seguimiento posterior al artículo, para enlazar con la Fase 7 del plan completo.