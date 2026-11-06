# SECURITY.md (v2)

Documento de seguridad de Soude, sistema de autenticación basada en EEG (P300). Cada control tiene un estado:

- **[P]** implementado y verificable en el código del artículo
- **[E]** esqueleto con interfaz y pruebas de contrato, sin lógica completa
- **[F]** diseño o trabajo futuro, no implementado

Este documento debe mantenerse consistente con `PLAN_REDISEÑO_SOUDE_v3.md` y con el protocolo de ética (`PROTOCOLO_EXPERIMENTAL_SOUDE.md`). Si uno cambia, los otros se revisan.

---

## 1. Objetivo y alcance

Proteger la identidad del usuario, sus plantillas biométricas, los registros de sesión y la integridad del flujo de autenticación.

La EEG y las respuestas cognitivas se tratan como **datos personales sensibles**. Aun pseudonimizados, pueden conservar riesgo de reidentificación, y así se informa a los participantes.

**Fuera de alcance de esta versión:** despliegue comercial, certificación de seguridad, y evaluación empírica de ataques sofisticados de inyección o replay (solo se evalúa un replay básico, ver 4.3).

---

## 2. Modos de operación

| Modo | Uso | Datos | Controles |
|---|---|---|---|
| Desarrollo | Pruebas con simulador y datos sintéticos | Sintéticos únicamente | Sin datos reales en el repositorio |
| Investigación | Recolección con participantes bajo protocolo aprobado | EEG pseudonimizado, en volumen cifrado | Todo lo marcado [P] |
| Despliegue | Uso real fuera de estudio | Plantillas cifradas, sin EEG crudo por defecto | Requiere implementar los controles [E] y [F] antes de usarse |

El modo investigación **sí guarda EEG** (lo exige el análisis), y el modo despliegue no debería. Ambos modos no deben compartir directorio ni claves.

---

## 3. Modelo de amenazas

| ID | Amenaza | Qué protege el sistema | Estado |
|---|---|---|---|
| T1 | Impostor que **no conoce** la contraseña mental | Factor de conocimiento (P300 a los símbolos correctos) | [P] evaluado en el estudio |
| T2 | Impostor que **conoce y atiende** la contraseña | Factor biométrico (plantilla de respuesta) | [P] evaluado; protección **limitada y medida**, no garantizada |
| T3 | Replay de EEG, inyección de señal sintética o manipulación de marcadores | Integridad del flujo | [P] mitigaciones básicas; [F] evaluación empírica completa |
| T4 | Robo de plantillas o de datos de sesión en reposo | Confidencialidad | [P] cifrado en reposo y separación de identidad |
| T5 | Acceso no autorizado de personal o colaboradores | Confidencialidad, trazabilidad | [P] acceso mínimo por proceso; [E] roles y auditoría |
| T6 | Señal de mala calidad (ruido, fatiga, mala colocación) usada para forzar o degradar una decisión | Robustez | [P] compuerta de calidad |

**Limitaciones que se reconocen explícitamente:**

- Un impostor informado puede producir un P300. La plantilla biométrica reduce, pero no elimina, esa aceptación; el estudio mide cuánto.
- La plantilla de EEG no se puede revocar como una contraseña. Una plantilla filtrada es un problema permanente para esa persona. Por eso se cifra y se limita su vida útil; la protección con comparación sin descifrar es [F].
- El EEG es un dato biométrico y puede permitir reidentificación, aun sin nombre.

---

## 4. Controles

### 4.1. Separación de dominios de datos

| Dominio | Contenido | Ubicación | Estado |
|---|---|---|---|
| Identidad | Nombre, correo, tabla de enlace código↔persona | Volumen cifrado independiente, otra clave, otro medio | [P] (procedimiento), fuera del software |
| Biometría | Plantillas, parámetros de modelo | Almacén cifrado de Soude | [P] |
| Sesión | EEG, marcadores, calidad, decisiones | `SOUDE_DATA_DIR` (volumen cifrado, **fuera del repositorio**) | [P] |
| Análisis | Tablas y figuras agregadas | Repositorio, solo resultados agregados o sintéticos | [P] |

Datos de desarrollo, validación y producción no comparten almacén.

### 4.2. Cifrado y gestión de claves

- Cifrado autenticado **AES-256-GCM** (biblioteca `cryptography`) para plantillas y la lista de imágenes objetivo. Nonce aleatorio de 96 bits por archivo y datos asociados (`subject_id`, versión) para ligar el cifrado al contexto. [P]
- Claves **fuera del repositorio**: llavero del sistema o variable de entorno, nunca en código ni en `.env` versionado. [P]
- `.gitignore` y detección de secretos en *pre-commit* y CI. [P]
- Rotación periódica de claves y bóveda de secretos. [E]
- Cifrado de disco/volumen para `SOUDE_DATA_DIR` según protocolo. [P] (procedimiento)

### 4.3. Integridad del flujo y anti-replay

- Validación de stream: existencia, número de canales, frecuencia de muestreo, timestamps monótonos, huecos de datos. [P]
- Medición de jitter y latencia entre marcador y muestra, con rechazo si excede la tolerancia. [P]
- **Secuencia de estimulación aleatoria por intento**, generada con un generador criptográfico (`secrets`) y registrada en la sesión. Un registro reproducido tendría una secuencia distinta a la del intento actual y no coincidiría. [P]
- Detección de duplicados: se calcula un hash de las épocas y marcadores de cada sesión, y se rechaza un intento cuyo contenido coincida con una sesión previa. [P]
- Sincronización multi-stream (cuatro streams LSL). [E]
- Evaluación empírica de replay con señales sintéticas más sofisticadas. [F]

### 4.4. Compuerta de calidad y rechazo seguro

La política de decisión prioriza seguridad sobre comodidad:

- Calidad de señal insuficiente → `RETRY` (no se decide).
- Evidencia ambigua → `CONTINUE` hasta un máximo de repeticiones; si sigue ambigua → `REJECT`.
- Confianza baja → `REJECT`.
- Sesión con sincronía inválida → no se emite decisión de aceptación.

Los umbrales se fijan antes de la recolección y no se ajustan con los resultados de prueba. [P]

Bloqueo tras intentos fallidos consecutivos (límite por usuario y ventana de tiempo). [E]

### 4.5. Manejo del secreto (contraseña mental) y de la plantilla

- La lista de imágenes objetivo se guarda **cifrada**, no hasheada, porque el sistema necesita conocer cuáles son para etiquetar épocas. Se descifra solo en memoria durante la sesión. [P]
- La plantilla biométrica se guarda cifrada y se descifra en memoria solo durante la comparación. Un hash no sirve aquí, porque la comparación biométrica es por similitud y no por igualdad exacta. [P]
- Ni el secreto ni la plantilla aparecen en registros de depuración ni en mensajes de error. [P]
- Protección de plantilla con biometría cancelable o comparación en dominio cifrado. [F]
- Actualización de plantilla por *drift*. [E/F]

### 4.6. Control de acceso y trazabilidad

- Principio de menor privilegio: el proceso de análisis no necesita acceso a la tabla de enlace; el software de adquisición no necesita acceso a las plantillas de otros. [P] (separación por almacén)
- Roles (investigador, operador, desarrollador, auditor) y permisos por módulo. [E]
- Registro de auditoría a prueba de manipulación (cadena de hashes). [E]
- Registro de decisiones con versión de software, modelo y plantilla, y calidad por canal, para reconstruir cualquier decisión. [P]

### 4.7. Retención y borrado

- Tabla de enlace: destruida a más tardar [6 meses] después de la publicación, según protocolo. [P] (procedimiento)
- EEG pseudonimizado: retención [X años] y borrado seguro después. [P] (procedimiento)
- Borrado programado automático. [E]
- Copias temporales y exportaciones minimizadas; sin envío por correo ni a nube pública. [P] (procedimiento)

---

## 5. Privacidad y cumplimiento

- Minimización: solo se recolectan los datos listados en el protocolo (EEG, marcadores, edad, sexo, lateralidad, contacto).
- Consentimiento informado con advertencia explícita sobre el riesgo de reidentificación del EEG.
- Los resultados se publican de forma agregada; los datos crudos no se comparten sin nueva autorización del comité y consentimiento adicional.
- Normativa aplicable en México y requisitos de aviso de privacidad: **verificar con el comité o asesoría legal de la escuela**. Este documento no sustituye esa revisión.

---

## 6. Gestión de incidentes (versión de investigación)

**Eventos que se tratan como incidente:** pérdida o robo de equipo o medio con datos, acceso no autorizado a datos de sesión o a la tabla de enlace, exfiltración de plantillas, alteración del flujo de adquisición o de una sesión, subida accidental de datos reales al repositorio.

**Procedimiento mínimo:**

1. Contener: aislar el equipo o cuenta afectada y revocar claves si aplica.
2. Preservar evidencia: no borrar registros, hacer copia forense cifrada.
3. Notificar al investigador principal y al asesor de inmediato, y al comité de ética según sus reglas.
4. Evaluar el impacto: qué participantes y qué datos.
5. Informar a los participantes afectados si procede y remediar.
6. Documentar causa raíz y acciones correctivas.

Si se sube un dato real al repositorio por error, se considera expuesto aunque se borre del último *commit*: se reescribe el historial, se rotan las claves y se notifica.

---

## 7. Verificación

Cada control [P] debe tener una prueba o un procedimiento comprobable:

| Control | Verificación |
|---|---|
| Cifrado AES-GCM | Test: alterar un byte del archivo produce error de autenticación; ida y vuelta exacta |
| Claves fuera del repo | Escaneo de secretos en CI; test que falla si hay claves o listas de objetivos en texto plano |
| Datos fuera del repo | `.gitignore` y comprobación en CI de que no hay EEG ni sesiones reales versionados |
| Compuerta de calidad | Tests con ruido, canal desconectado y marcadores desalineados |
| Validación de sincronía | Test con timestamps no monótonos, huecos y jitter alto |
| Anti-replay básico | Test: reenviar una sesión previa es rechazado por secuencia o por hash duplicado |
| Rechazo seguro | Tests de política: calidad baja → `RETRY`; ambigüedad persistente → `REJECT` |
| Registro de decisiones | Una sesión se puede reconstruir completa desde sus archivos de metadatos |
| Separación de identidad | Revisión manual: los archivos de sesión no contienen nombre ni correo |

---

## 8. Resumen de estado

| Control | Estado |
|---|---|
| Cifrado de plantilla y objetivos (AES-256-GCM), clave externa | [P] |
| Pseudonimización y separación de dominios | [P] |
| Compuerta de calidad y rechazo seguro | [P] |
| Validación de stream, jitter, secuencia aleatoria, detección de duplicados | [P] |
| Registro de decisiones con versión de modelo | [P] |
| Bóveda y rotación de claves | [E] |
| Roles, bloqueo por intentos, auditoría a prueba de manipulación | [E] |
| Retención y borrado automático | [E] |
| Multi-stream LSL | [E] |
| Biometría cancelable y plantilla revocable | [F] |
| Evaluación empírica avanzada de replay e inyección | [F] |
| Certificación y evaluación de despliegue | [F] |

---

## 9. Principio rector

La seguridad no es una capa final: se decide desde la adquisición hasta la decisión. En este proyecto cada control se declara con su estado real y su forma de verificarse, y lo que todavía no está implementado se dice como tal.