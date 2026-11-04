 # SECURITY.md

## 1. Objetivo

Este documento define la estrategia de seguridad para Soude como sistema de autenticación biométrica basado en señales EEG. Su propósito es proteger la identidad del usuario, los perfiles biométricos, la información de sesión y la integridad del flujo de autenticación.

Soude no debe tratar la señal cerebral como un dato inocuo. La EEG contiene evidencia altamente sensible de identidad y comportamiento cognitivo, por lo que debe manejarse con controles de acceso, cifrado, trazabilidad y aislamiento de datos.

---

## 2. Estado actual del proyecto desde la perspectiva de seguridad

El proyecto actual ya incorpora captura EEG, procesamiento de epochs y registro de sesiones con métricas de autenticación. Sin embargo, la seguridad actual todavía se encuentra en un nivel de prototipo:

- las sesiones se almacenan en `logs/<timestamp>/` con señales, epochs y resultados;
- no existe una base de perfiles biométricos por usuario;
- no existe un vault ni un esquema de cifrado para plantillas o referencias de autenticación;
- no hay separación formal entre identidad, biometría y evidencias técnicas;
- el sistema guarda evidencia analítica sensible sin distinguir entre producción, validación y análisis de desarrollo;
- la decisión final se basa en un umbral de amplitud, sin un modelo probabilístico ni una política de riesgo por sesión.

Esto implica que el proyecto ya tiene capacidad técnica de adquisición y análisis, pero aún no dispone de la capa de control de riesgo, gestión de identidad y protección de datos que exige un sistema biométrico real.

---

## 3. Principios de seguridad

### 2.1. Minimización de datos

Se debe recopilar únicamente la información estrictamente necesaria para:

- enrolamiento biométrico;
- validación de autenticación;
- diagnóstico técnico de calidad de señal;
- auditoría y análisis forense interno.

No se deben almacenar datos innecesarios ni copias redundantes de señales sin necesidad operacional.

### 2.2. Separación de dominios

Los datos deben dividirse en:

1. Datos de identidad del usuario
2. Datos biométricos y plantillas
3. Datos de sesión y evidencias técnicas
4. Datos de análisis y entrenamiento

Cada dominio debe tener permisos y acceso distintos. Nunca deben mezclarse criterios de producción con bases de prueba o artefactos de investigación sin control.

### 2.3. No almacenamiento de secretos en texto plano

- no se debe guardar ninguna plantilla biométrica en formato legible;
- no se debe persistir la contraseña mental de forma equivalente a texto plano;
- no deben crearse archivos de sesión con valores sensibles en formato fácilmente extraíble;
- cualquier token, secreto o clave de acceso debe almacenarse en mecanismo seguro con cifrado y rotación.

### 2.4. Cifrado por defecto

Todo dato persistente sensible debe estar cifrado en reposo. En caso de almacenamiento local o en archivos, se recomienda:

- cifrado AES-256 o equivalente;
- claves gestionadas fuera del repositorio;
- rotación periódica de credenciales;
- separación entre la identidad del usuario y la clave de cifrado.

---

## 4. Activos de seguridad

### 3.1. Datos altamente sensibles

- EEG cruda y epochs de autenticación
- plantillas biométricas del usuario
- metadata de identidad asociada a la sesión
- scores de autenticación y decisiones finales
- historial de validaciones e intentos de acceso

### 3.2. Datos moderadamente sensibles

- parámetros de configuración del protocolo
- decisiones internas del pipeline
- métricas de calidad por canal
- registros de operación y análisis

### 3.3. Datos no sensibles operativos

- configuración general del software
- métricas de rendimiento no vinculadas a la identidad del usuario
- datos de diagnóstico de laboratorio o simulación

---

## 5. Amenazas relevantes

### 4.1. Suplantación de identidad

Un atacante podría intentar:

- reproducir un patrón visual sin autenticar de verdad;
- manipular la señal o los marcadores para producir una decisión falsa;
- inyectar un flujo de EEG sintético para engañar al sistema.

Mitigación:

- validar calidad de señal antes de aceptar decisión;
- usar comparación biométrica con plantilla del usuario;
- aplicar una decisión probabilística con umbrales de confianza;
- requerir evidencia suficiente antes de conceder acceso.

### 4.2. Robo de plantillas biométricas

Si la plantilla de EEG o el template de usuario se expone, el atacante podría reutilizarlo para autenticación fraudulenta.

Mitigación:

- cifrado fuerte en reposo;
- almacenamiento separado de identidad y biometría;
- limitación de acceso por rol;
- hashes o derivaciones irreversibles para referencias internas.

### 4.3. Manipulación de marcadores o timestamps

Un atacante podría alterar tiempos de estímulo o desalinear la señal con marcadores para producir alucinación de P300 o forzar aceptación.

Mitigación:

- validar la sincronía entre señales y eventos;
- registrar marcas con timestamps de origen;
- difundir checks de integridad del flujo LSL;
- guardar evidencias para auditoría posterior.

### 4.4. Ataques por ruido o artefactos

Se puede intentar forzar una decisión con ruido, movimiento, fatiga o mala colocación del electrodos.

Mitigación:

- análisis de calidad por canal;
- rechazo automático de artefactos;
- reintento recomendado si la sesión no cumple calidad mínima;
- no aceptar sesion ambigua.

---

## 6. Controles de seguridad obligatorios

### 5.1. Control de acceso

- acceso mínimo necesario por usuario y rol;
- permisos diferenciados para desarrollo, validación y producción;
- registro de accesos y acciones relevantes;
- bloqueo tras intentos fallidos o acceso sospechoso.

### 5.2. Integridad del pipeline

- guardar metadatos de versión del algoritmo;
- registrar configuración del protocolo por sesión;
- versionar plantillas, modelos y reglas de autenticación;
- impedir modificaciones silenciosas en la definición del modelo.

### 5.3. Rechazo seguro

La política de decisión debe ser conservadora:

- si la confianza es baja, se rechaza;
- si la calidad de la señal es insuficiente, se rechaza;
- si la evidencia es ambigua, se exige re-challenge o segunda validación.

El sistema debe preferir seguridad sobre comodidad.

### 5.4. Registro forense

Cada sesión debe incluir:

- ID del usuario o intento;
- timestamp del evento;
- metadata del dispositivo;
- calidad por canal;
- lista de marcadores y epochs;
- score final y decisión;
- versión del modelo y del algoritmo de decisión.

Esto permite reconstruir la autenticación e investigar incidentes.

---

## 7. Seguridad del almacenamiento

### 6.1. Persistencia de plantillas

Las plantillas biométricas deben almacenarse en una capa de datos separada, con:

- cifrado en reposo;
- control de acceso explícito;
- políticas de retención claras;
- hash de identidad para soporte de auditoría sin revelar datos sensibles.

### 6.2. Sesiones y logs

Las sesiones deben almacenarse en un almacenamiento protegido con:

- organización por ID de sesión;
- ACLs puntuales;
- políticas de eliminación y archivo;
- separación de `raw`, `processed`, `audit` y `reports`.

### 6.3. Datos de entrenamiento

Los datos de benchmarking o entrenamiento no deben mezclarse con perfiles de usuarios reales. Debe existir una separación clara entre:

- data de investigación,
- data de validación,
- data de producción.

---

## 8. Seguridad del flujo de adquisición y LSL

### 7.1. Validación de stream

Antes de aceptar un stream, el sistema debe comprobar:

- que el stream exista;
- que la cantidad de canales sea la esperada;
- que los timestamps sean consistentes;
- que la frecuencia de muestreo sea compatible;
- que no haya pérdida importante de datos.

### 7.2. Tolerancia a manipulación

- no aceptar streams sin metadatos mínimos;
- marcar una sesión como sospechosa si el flujo presenta latencia irregular o pérdida repetida;
- invalidar decisiones si la sincronía marca inconsistencia.

---

## 9. Política de privacidad y cumplimiento

El sistema debe asumir que la EEG y los patrones de respuesta cognitiva son información personal altamente sensible. En consecuencia:

- restringir acceso a usuarios autorizados;
- evitar compartir sesiones con terceros sin consentimiento y control formal;
- descartar o limitar copias temporales;
- documentar retención, uso y riesgos asociados al tratamiento de datos biométricos.

---

## 10. Política de incidentes

### 9.1. Eventos críticos

Se deben reportar como incidentes y gestionar con prioridad:

- pérdida o exfiltración de plantillas biométricas;
- manipulación de sesiones de autenticación;
- alteración de timestamps o flujo LSL;
- acceso no autorizado a datos de sesión;
- vulneración del modelo o de una política de seguridad.

### 9.2. Procedimiento mínimo

1. aislar el componente afectado;
2. congelar y guardar evidencia;
3. revisar integridad del flujo de autenticación;
4. invalidar plantillas o sesiones comprometidas;
5. documentar el incidente y remediar la causa raíz.

---

## 11. Recomendación final de seguridad

Soude no puede ser una solución de autenticación cerrada sobre un repositorio sin controles de seguridad. Su valor y su riesgo están en la señal cerebral: no es solo un dato técnico, sino una huella biométrica. Por ello, la implementación debe asumir una política de seguridad rigurosa, con cifrado, separación de dominios, trazabilidad, rechazo de calidad insuficiente y toma de decisiones conservadora.

La seguridad no debe considerarse como una capa final; debe ser un principio arquitectónico desde la adquisición hasta la decisión final de autenticación.
