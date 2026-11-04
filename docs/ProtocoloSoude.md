# Protocolo de investigación: Autenticación biométrica basada en potenciales evocados P300 con un dispositivo EEG de 8 canales

> **Borrador para adaptar al formato del comité de ética.** Los textos entre `[corchetes]` son campos por completar. Los parámetros marcados con (†) dependen del diseño final del sistema y deben confirmarse antes del envío.

---

## 1. Datos generales

| Campo | Contenido |
|---|---|
| Título | Autenticación biométrica basada en potenciales evocados P300 con EEG de bajo costo: evaluación entre sesiones frente a impostores ingenuos e informados |
| Investigador principal | [Nombre, programa, correo] |
| Asesor / responsable académico | [Nombre, cargo, correo] |
| Institución | [Nombre de la escuela] |
| Lugar de realización | [Laboratorio / aula, edificio] |
| Duración estimada del estudio | [11 semanas totales; recolección en ~3 semanas, fechas por definir] |
| Tipo de estudio | Experimental, observacional, no clínico, sin intervención terapéutica |
| Nivel de riesgo propuesto | Riesgo mínimo |

---

## 2. Antecedentes y justificación

Los potenciales evocados P300 son respuestas cerebrales a estímulos relevantes poco frecuentes y pueden registrarse con electroencefalografía (EEG) no invasiva. Un sistema que pida al usuario atender mentalmente a un conjunto secreto de imágenes ("contraseña mental") puede detectar si esa persona reconoce los estímulos de su contraseña, y la forma de su respuesta (latencia, amplitud, distribución espacial) podría aportar un componente biométrico adicional.

Este estudio busca evaluar, con datos de participantes reales y un dispositivo EEG de bajo costo, dos preguntas:

1. ¿Puede el sistema distinguir a un usuario genuino de un impostor que **no conoce** la contraseña?
2. ¿Puede distinguirlo de un impostor que **sí conoce** la contraseña y la atiende, usando la plantilla biométrica de respuesta?

Los resultados se utilizarán para un trabajo académico (artículo científico de congreso) y el desarrollo del prototipo Soude.

---

## 3. Objetivos

### 3.1. Objetivo general
Evaluar el desempeño de un sistema de autenticación basado en P300 y un EEG de 8 canales, en términos de tasas de falsa aceptación (FAR), falso rechazo (FRR) y tasa de error igual (EER).

### 3.2. Objetivos específicos
- Construir una plantilla biométrica por participante a partir de una sesión de enrolamiento.
- Medir el desempeño de autenticación genuina en sesiones posteriores realizadas en días distintos.
- Estimar FAR frente a impostores ingenuos (no conocen la contraseña) y frente a impostores informados (la conocen).
- Caracterizar la tasa de fallos de adquisición (sesiones rechazadas por mala calidad de señal) y el tiempo de autenticación.

### 3.3. Hipótesis (exploratorias)
- H1: Las sesiones genuinas producen una diferencia P300 objetivo vs. no objetivo mayor que las sesiones de impostor ingenuo.
- H2: La plantilla individual reduce la aceptación de impostores informados frente a una regla de amplitud sin plantilla.

---

## 4. Diseño del estudio

### 4.1. Estructura

Estudio intra-sujeto con tres sesiones por participante, en días distintos (separación mínima de 24 h y máxima de 14 días entre sesiones†).

| Sesión | Propósito | Contenido |
|---|---|---|
| S1 | Enrolamiento | 4 bloques† de estimulación con la contraseña común; se construye la plantilla |
| S2 | Autenticación | 2 bloques† genuinos + 1 bloque de impostor ingenuo |
| S3 | Autenticación | 2 bloques† genuinos + 1 bloque de impostor ingenuo |

Un **bloque** consiste en la presentación de 100 estímulos (20 imágenes distintas, mostradas en orden pseudoaleatorio, 3 de ellas "objetivo") con duración aproximada de 1 minuto†. Entre bloques hay descansos de al menos 1 minuto.

### 4.2. Contraseña común

Todos los participantes usan el mismo conjunto de 3 imágenes objetivo. Esto permite evaluar, de forma offline, a cada participante como **impostor informado** frente a las plantillas de los demás, sin pedir sesiones adicionales. Las imágenes son símbolos geométricos con el mismo fondo y tamaño de fuente, para evitar que diferencias de luminancia o color generen respuestas visuales que no dependan de la atención.

### 4.3. Condiciones de impostor

- **Impostor informado (T2):** análisis offline; la sesión genuina de un participante se puntúa contra la plantilla de otro participante.
- **Impostor ingenuo (T1):** en un bloque de S2 y S3, se pide al participante que atienda a 3 símbolos distintos a los de la contraseña, o que no atienda a ninguno en particular. Se le explica con claridad en qué consiste el bloque; **no hay engaño**.

### 4.4. Orden de bloques
El orden de los bloques genuinos y de impostor ingenuo se aleatoriza por participante y por sesión, para controlar fatiga y efectos de orden.

---

## 5. Participantes

### 5.1. Población
Estudiantes de [escuela], de **18 a 25 años** aproximadamente.

### 5.2. Tamaño de muestra
- Meta mínima: **20 participantes**. Meta deseable: 30, sujeta a tiempo y disponibilidad.
- No se realiza un cálculo de potencia formal. El tamaño se justifica por factibilidad y por comparabilidad con estudios exploratorios de biometría EEG. Los resultados se reportarán con intervalos de confianza (bootstrap a nivel de sujeto) y se declarará la limitación del tamaño de muestra.

### 5.3. Criterios de inclusión
- Edad ≥ 18 años.
- Estudiante de [escuela].
- Visión normal o corregida con lentes/lentes de contacto.
- Disponibilidad para tres sesiones en días distintos.
- Consentimiento informado firmado.

### 5.4. Criterios de exclusión
- Antecedentes personales o familiares directos de epilepsia, o crisis convulsivas previas.
- Antecedentes de fotosensibilidad o migraña desencadenada por estímulos visuales.
- Lesiones o afecciones de piel en la zona de colocación de electrodos.
- Condiciones neurológicas diagnosticadas que afecten el EEG.
- Consumo de medicamentos o sustancias que, a criterio del participante, alteren su estado de alerta el día de la sesión (pueden reprogramar).

### 5.5. Criterios de retiro y eliminación
- Retiro voluntario en cualquier momento, sin explicación ni consecuencia.
- Incumplimiento de las tres sesiones (los datos disponibles se conservan solo si el participante lo autoriza).
- Malestar durante la sesión (interrupción inmediata).

### 5.6. Reclutamiento
- Mediante invitación abierta (carteles / mensajes institucionales) [especificar medio].
- **Sin compensación económica ni académica.** La participación no otorga créditos ni calificación.
- Para evitar presión indebida, los docentes de los participantes no participan en el reclutamiento ni conocen quiénes participan. El investigador principal no recluta a estudiantes sobre quienes tenga relación de evaluación.

---

## 6. Procedimiento

### 6.1. Tamizaje y consentimiento
1. El participante recibe el consentimiento informado (Anexo A) con antelación y puede hacer preguntas.
2. Responde el cuestionario de tamizaje (Anexo B). Si cumple algún criterio de exclusión, no continúa y sus respuestas se destruyen.
3. Firma el consentimiento en dos copias (una para él/ella).

### 6.2. Cada sesión (duración aproximada 40 min†)
1. Colocación del dispositivo y revisión de calidad de señal (calidad por canal).
2. Instrucciones del bloque (atender mentalmente a los 3 símbolos objetivo, minimizar parpadeos y movimientos durante la secuencia).
3. Bloques de estimulación con descansos.
4. Retiro del dispositivo, limpieza y cierre.

Si la calidad de señal es insuficiente, se reajusta el dispositivo. Si persiste, la sesión se reprograma y se registra como fallo de adquisición.

### 6.3. Equipo
- Dispositivo EEG Unicorn Hybrid Black (g.tec), 8 canales (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8), 250 Hz.
- Computadora con software Soude [versión†] y pantalla para estimulación.
- Material de limpieza y desinfección del dispositivo entre participantes.

---

## 7. Datos recolectados

| Categoría | Datos |
|---|---|
| Identificación (separada) | Nombre, correo de contacto |
| Demográficos mínimos | Edad, sexo, lateralidad manual |
| Fisiológicos | EEG de 8 canales, acelerómetro y giroscopio del dispositivo |
| Del sistema | Marcadores de estímulo, calidad por canal, resultado de cada decisión, versión del software |
| De sesión | Fecha, hora, número de sesión, tipo de bloque, notas de incidencias |

No se recolectan datos clínicos, de salud mental, ni se toman imágenes o video de los participantes. No se realiza ninguna interpretación clínica del EEG.

---

## 8. Riesgos, molestias y beneficios

### 8.1. Riesgos
- **Mínimos.** El EEG es una técnica no invasiva que solo registra actividad eléctrica.
- Posibles molestias: incomodidad o presión leve por el dispositivo, fatiga visual o mental durante los bloques, irritación cutánea leve en zonas de contacto.
- Las secuencias de estímulos visuales presentan imágenes a baja frecuencia (aproximadamente una imagen cada 0.58 s†), muy inferior a las frecuencias asociadas a crisis fotosensibles, y se excluye a personas con antecedentes relevantes. Aun así, se informa al participante y puede detener la sesión en cualquier momento.
- **Riesgo de privacidad:** el EEG puede considerarse un dato biométrico sensible; ver sección 9.

### 8.2. Medidas de mitigación
- Tamizaje previo, descansos entre bloques, sesiones cortas.
- Limpieza y desinfección del dispositivo entre participantes.
- Presencia del investigador durante toda la sesión.
- Protección de datos descrita en la sección 9.

### 8.3. Beneficios
- No hay beneficio directo para el participante.
- Beneficio social y científico: contribuye a evaluar la viabilidad y los límites de la autenticación biométrica por EEG.

### 8.4. Hallazgos incidentales
El sistema no es una herramienta diagnóstica. Si un participante pregunta, se le indica que los registros no permiten detectar ni descartar condiciones de salud. Si presenta cualquier malestar físico durante la sesión, se detiene y se canaliza con [servicio médico de la escuela, si existe].

---

## 9. Manejo de datos y confidencialidad

### 9.1. Pseudonimización
- Cada participante recibe un código (`P001`, `P002`…) generado al azar.
- Los datos de identificación (nombre, correo) se guardan en una **tabla de enlace separada** de los datos de EEG. Los archivos de sesión solo contienen el código.

### 9.2. Almacenamiento
- **Datos de EEG y de sesión:** en una unidad cifrada (cifrado de disco completo o volumen cifrado) en el equipo del laboratorio [especificar], con copia de respaldo cifrada en [ubicación].
- **Tabla de enlace:** en un volumen cifrado independiente, almacenado en un medio distinto al de los datos de EEG, con clave separada.
- **Plantillas biométricas** generadas por el sistema: cifradas en reposo con una clave fuera del repositorio de código.
- Los datos no se suben a servicios en la nube públicos ni se envían por correo.

### 9.3. Acceso
- Solo el investigador principal y el asesor [nombres] tienen acceso a los datos de EEG.
- Solo el investigador principal tiene acceso a la tabla de enlace.
- Si se incorporan otros colaboradores, se registrará quién y con qué propósito.

### 9.4. Retención y destrucción
- La tabla de enlace se destruye de forma segura al concluir el análisis y como máximo [6 meses] después de la publicación de resultados, tras lo cual los datos quedan sin vínculo directo con la identidad.
- Los datos de EEG pseudonimizados se conservan [X años†] para reproducibilidad del estudio y después se eliminan de forma segura.
- Mientras exista la tabla de enlace, el participante puede solicitar que sus datos sean eliminados.
- Se informa al participante que, por su naturaleza biométrica, el EEG podría conservar cierto riesgo de reidentificación incluso sin nombre.

### 9.5. Difusión de resultados
- Se reportan resultados agregados y gráficas de grupo. No se publican registros individuales.
- Los datos crudos **no se publican ni comparten** con terceros. Si en el futuro se quisiera compartir un conjunto de datos, requerirá una nueva autorización del comité y consentimiento adicional explícito.

### 9.6. Marco normativo
Se atenderá la normativa aplicable en México en materia de protección de datos personales [verificar con el comité o asesoría legal de la escuela cuál aplica y qué requisitos de aviso de privacidad se requieren] y los lineamientos de la institución. Se incluirá un aviso de privacidad integrado o anexo al consentimiento.

---

## 10. Análisis de datos

- **Procesamiento:** filtrado, extracción de épocas, rechazo de artefactos y compuerta de calidad definidos antes de analizar los datos (se fija el procedimiento antes de ver los resultados de impostores).
- **Evaluación:** plantilla construida solo con S1; evaluación con S2 y S3. Nunca se usan datos de la misma sesión para entrenar y evaluar.
- **Métricas:** FAR (separadas para T1 y T2), FRR, EER, curva ROC y AUC, tasa de fallo de adquisición y tiempo de autenticación.
- **Incertidumbre:** intervalos de confianza por bootstrap remuestreando **participantes**, no sesiones.
- **Análisis de sensibilidad:** número de bloques de enrolamiento, número de épocas por decisión.
- Los valores atípicos o sesiones excluidas se reportan con su motivo.

---

## 11. Consideraciones éticas

- **Autonomía:** participación voluntaria, consentimiento informado, derecho de retiro sin consecuencias.
- **Relación de poder:** ver sección 5.6 (sin evaluación ni compensación ligada a la participación).
- **Transparencia:** todas las tareas, incluido el bloque de impostor ingenuo, se explican por adelantado; no hay engaño.
- **Beneficencia y no maleficencia:** riesgo mínimo, tamizaje de exclusión, interrupción ante malestar.
- **Justicia:** reclutamiento abierto, sin criterios que excluyan de forma injustificada.
- **Privacidad:** minimización de datos, pseudonimización, cifrado, acceso limitado y política de destrucción.
- **Conflictos de interés:** [declarar, si los hay].

---

## 12. Cronograma

| Etapa | Periodo |
|---|---|
| Envío a comité de ética | [Semana 2] |
| Respuesta y ajustes | [Semanas 2–5] |
| Reclutamiento y recolección | [Semanas 6–8, tras la aprobación] |
| Análisis | [Semanas 8–9] |
| Redacción y envío del artículo | [Semanas 10–11] |

---

## Anexo A. Consentimiento informado (borrador)

**Título del estudio:** Autenticación biométrica basada en potenciales evocados P300 con EEG de bajo costo

**Investigador principal:** [Nombre, contacto]  **Asesor:** [Nombre, contacto]  **Institución:** [Nombre]

**¿De qué trata este estudio?**
Estamos evaluando un sistema que usa la actividad eléctrica del cerebro (EEG) para verificar la identidad de una persona. Se te pedirá que mires secuencias de imágenes en una pantalla mientras un dispositivo ligero, parecido a una diadema con electrodos, registra tu actividad cerebral. El registro no lee tus pensamientos ni permite diagnosticar ninguna condición de salud.

**¿Qué tendrás que hacer?**
Participarás en **tres sesiones** de unos 40 minutos, en días distintos. En cada sesión te colocaremos el dispositivo, verás varias secuencias de imágenes de cerca de 1 minuto y se te pedirá concentrarte en ciertas imágenes. En algunas secuencias te pediremos atender a otras imágenes distintas; esto forma parte del estudio y te lo explicaremos antes.

**¿Qué riesgos hay?**
El riesgo es mínimo. Podrías sentir cansancio visual o mental, o una leve molestia por el dispositivo o irritación de la piel. Las imágenes cambian a una velocidad baja, pero si tienes antecedentes de epilepsia o de molestias por luces o estímulos visuales no puedes participar. Puedes detener la sesión en cualquier momento.

**¿Qué beneficios hay?**
No recibirás un beneficio directo ni compensación económica o académica. Tu participación contribuirá a la investigación sobre seguridad biométrica.

**¿Qué datos se recopilan y cómo se protegen?**
Recopilamos tu nombre y correo (para coordinar las sesiones), edad, sexo, mano dominante y tu registro de EEG. Tu nombre se guarda separado de tus datos de EEG, que se identifican solo con un código. Los datos se guardan cifrados y solo el equipo de investigación tiene acceso. Los resultados se publicarán de forma agregada. **Tus registros individuales no se publican ni se comparten con terceros.** Debido a que el EEG es un dato biométrico, no es posible garantizar que sea imposible reidentificarlo, aunque tomamos medidas para reducir ese riesgo. El archivo que vincula tu nombre con tu código se destruirá a más tardar [6 meses] tras la publicación de resultados; los datos de EEG sin nombre se conservarán [X años] y después se eliminarán.

**Tus derechos**
- Tu participación es voluntaria y no afecta tu situación académica de ninguna forma.
- Puedes retirarte en cualquier momento sin dar explicaciones.
- Puedes pedir que se eliminen tus datos mientras exista el archivo que los vincula contigo.
- Puedes hacer preguntas en cualquier momento a [contacto]. Si tienes dudas sobre tus derechos como participante, puedes contactar al comité de ética: [contacto del comité].

**Aviso de privacidad:** [enlace o anexo según la normativa aplicable].

**Declaración**
Declaro que he leído (o me han leído) este documento, que he tenido oportunidad de preguntar y que acepto participar voluntariamente.

Nombre del participante: ______________________  Firma: ______________  Fecha: ____________

Nombre del investigador: _______________________  Firma: ______________  Fecha: ____________

☐ Autorizo que, de no completar las tres sesiones, se conserven los datos ya recolectados.

---

## Anexo B. Cuestionario de tamizaje

Las respuestas se usan solo para determinar tu elegibilidad. Si no eres elegible, se destruirán.

1. ¿Tienes 18 años o más? ☐ Sí ☐ No
2. ¿Has tenido alguna vez una convulsión o te han diagnosticado epilepsia? ☐ Sí ☐ No
3. ¿Algún familiar directo (padres, hermanos) tiene epilepsia? ☐ Sí ☐ No
4. ¿Te molestan o te provocan dolor de cabeza, mareo o malestar las luces parpadeantes o patrones visuales? ☐ Sí ☐ No
5. ¿Tienes alguna lesión, herida o afección en la piel de la cabeza? ☐ Sí ☐ No
6. ¿Tienes algún diagnóstico neurológico? ☐ Sí ☐ No
7. ¿Tienes visión normal o corregida con lentes? ☐ Sí ☐ No
8. ¿Puedes asistir a tres sesiones en días distintos? ☐ Sí ☐ No

Datos para el estudio (solo si eres elegible): edad ___  sexo ___  mano dominante ☐ Derecha ☐ Izquierda ☐ Ambidiestra

---

## Anexo C. Hoja de registro por sesión (investigador)

| Campo | Valor |
|---|---|
| Código de participante | |
| Número de sesión (S1 / S2 / S3) | |
| Fecha y hora | |
| Duración | |
| Calidad por canal al inicio (OK / pobre) | Fz__ C3__ Cz__ C4__ Pz__ PO7__ Oz__ PO8__ |
| Orden de bloques | |
| Bloques completados / rechazados por calidad | |
| Incidencias (malestar, interrupciones, ajustes) | |
| Versión del software / parámetros | |

---

## Anexo D. Esquema de datos que debe soportar el sistema

Requisitos mínimos del software para que el protocolo sea ejecutable y reproducible:

- Cada sesión se guarda bajo `subject_id` (código) y `session_number`; nunca con nombre.
- Campo `block_type`: `enrollment`, `genuine`, `impostor_naive`.
- Se registra la versión del software y los parámetros del paradigma (SOA, número de repeticiones, ventana de análisis).
- La calidad por canal y el motivo de rechazo de épocas o sesiones se guardan junto con la decisión.
- Las plantillas se cifran en reposo; la clave no se guarda en el repositorio.
- Se genera un archivo de metadatos por sesión que permita reconstruir el análisis completo sin acceso a la tabla de enlace.