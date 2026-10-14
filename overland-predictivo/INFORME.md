# Correa overland: alertas antes de una detención

Demostración con datos completamente sintéticos. Dos reductores, tres sensores por reductor.
Un solo sensor ≥16 genera warning; un solo sensor ≥20 detiene toda la correa. El trip tiene prioridad.
La unidad de vibración no fue indicada: se usa una unidad arbitraria, sin asumir mm/s ni norma industrial.

## Histórico simulado

Abril a septiembre de 2026, seis meses completos con timestamps UTC. Frecuencia: 60 minutos.
4,392 filas y 26,352 lecturas de sensores. Se introducen
30 detenciones, mantenimiento de 8–16 horas y recuperación posterior.
La fila TRIP conserva el valor que cruza 20; después los sensores bajan mientras el equipo está detenido.
Es un histórico acelerado para enseñar el modelo, no una estimación de frecuencia real de fallas.

Patrones simulados: aumento sostenido durante 48–85 horas, crecimiento correlacionado entre sensores
del mismo reductor, transitorios benignos de corta duración y cinco fallas repentinas sin precursor.
Estas últimas permiten observar el límite de un predictor basado exclusivamente en vibración.

## Predicción y validación

Objetivo: identificar, cuando la correa está funcionando y aún no ha alcanzado 20,
si habrá un trip en las próximas 24 horas. Se usan valores actuales, medias,
desviaciones, incrementos, pendientes de 6/12/24 h y persistencia sobre 14.
Las ventanas se reinician después del mantenimiento. Se requieren 24 horas de historial de operación;
durante arranque o pérdida de datos el predictor carece de score, aunque los umbrales siguen operando.

Entrenamiento: abril–julio; validación: agosto; prueba: septiembre. Se purgan las últimas
24 horas del entrenamiento y validación para que sus etiquetas no usen eventos del período siguiente.
Se excluyen las últimas 24 horas del histórico porque no se conoce su futuro completo.
Filas elegibles train/validación/prueba: [2169, 533, 502].

Modelo Random Forest de 250 árboles. El score es una señal de riesgo sin calibrar, no una probabilidad validada.
El umbral 0.55 se eligió únicamente en agosto, minimizando
`5 × detenciones omitidas + falsos episodios`; los empates favorecen anticipación y menor tiempo en alerta.
Estos costes son una hipótesis de demostración y deben ajustarse con operación.
La alerta requiere dos muestras consecutivas sobre el umbral, separadas por 1 h.
Los umbrales de warning y trip actúan de inmediato, sin esa persistencia.

## Resultado fuera de entrenamiento (septiembre)

| Método | Detenciones anticipadas | Mediana de anticipación | Falsos episodios |
|---|---:|---:|---:|
| Random Forest | 4/5 | 14.5 h | 1 |
| Warning ≥16 | 4/5 | 19 h | 0 |
| Extrapolación lineal | 4/5 | 18 h | 5 |


Se cuenta como anticipada una detención si existe alerta en `(trip − horizonte, trip)`.
La anticipación es el tiempo entre la primera muestra en alerta dentro de esa ventana y el trip.
Un falso episodio es un inicio de alerta sin trip en las próximas 24 h; si una alerta
empieza demasiado temprano, ese inicio se cuenta falso aunque posteriormente coincida con una ventana real.
La comparación aplica a las mismas horas elegibles de prueba, incluyendo las restricciones de arranque.
Las medianas de métodos pueden usar distintos eventos: la tabla por evento permite comparar pares.

Métricas horarias ML: precision 0.800,
recall 0.467, average precision
0.763. Prevalencia positiva:
0.239. Son métricas por muestra, no por detención.

El conjunto de prueba contiene sólo cinco detenciones. El modelo anticipa cuatro, pero su mediana
de anticipación es menor que la del warning. En esta ejecución no demuestra una mejora global
frente a la protección por umbral; el evento 29 sí agrega siete horas de anticipación.

## Variables y patrones

Variables más influyentes en validación por permutación (caída en average precision):

- max_global: 0.0466.
- R2_S1_std_24h: 0.0155.
- R2_max: 0.0155.
- R2_dispersion: 0.0125.
- R2_S3_std_24h: 0.0045.
- R1_S1_media_12h: 0.0038.
- R1_max: 0.0038.
- R1_dispersion: 0.0034.

La importancia individual se reparte entre sensores y estadísticas correlacionadas. No prueba causa física.
Las pendientes, persistencia y medias ayudan a separar deterioro de picos aislados; las gráficas por evento
muestran si efectivamente hubo una alerta temprana. Los tipos de falla e inicios del deterioro son metadatos
exclusivos de la simulación: nunca entran al modelo.

## Lógica operativa propuesta

1. Cualquier sensor ≥20: detener la correa con el sistema de protección existente.
2. Cualquier sensor ≥16 y <20: warning inmediato.
3. Score ML sobre umbral durante dos muestras: inspección predictiva, indicando reductor y sensor máximo.
4. Sin score por historial insuficiente: mantener protecciones físicas y reportar predictor no disponible.

La predicción no cambia ni desactiva los umbrales. En el histórico las alarmas no disparan mantenimiento
preventivo, para poder medir anticipación contra detenciones observadas. No se demuestra que una alerta
hubiera evitado el trip ni se estiman ahorros.

## Uso con datos reales

Acordar la unidad, punto y tipo de medición RMS/pico, frecuencia, horizonte útil y coste de falsas alarmas.
Agregar carga, velocidad, temperatura, estado de marcha y mantenimientos para separar cambios operativos
de deterioro. Reentrenar y validar con eventos reales, más meses y evaluación temporal repetida; calibrar
el riesgo y operar primero en modo observación. La simulación demuestra el flujo, no eficacia industrial.

## Archivos

- `historico_6_meses.csv`: seis sensores, estado, etiquetas, score y partición.
- `eventos_simulados.csv`: verdad de simulación; no usar como entrada del predictor.
- `predicciones_prueba.csv`: decisiones del último mes, después de filtrar horas elegibles.
- `anticipacion_prueba.csv`: comparación de anticipación por detención.
- `metricas.json`, `ajuste_umbral_validacion.csv`, `importancia_variables.csv`.
- `modelo.joblib`: modelo, lista de variables, umbrales y parámetros.
- `historico.png`, `evento_*.png`: gráficos de los seis meses y cada detención de prueba.
- `overland.py`: simulación y pipeline reproducible.

## Ejecutar

Python 3.12 recomendado. Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python overland.py --out resultados
python overland.py --seed 17 --horizon 24 --step-minutes 30 --out resultados_alternativos
```

Cada salida se regenera con semilla fija. No se reajusta el umbral con septiembre.
Los timestamps se almacenan en UTC; para Chile convertir explícitamente a `America/Santiago`.
