# Alertas predictivas · correa overland

Proyecto ejecutado con datos sintéticos: dos reductores, tres sensores cada uno, seis meses completos de abril a septiembre de 2026. Lecturas horarias y riesgo de trip en las próximas 24 horas.

**Abrir `INFORME.md` para metodología y resultados, o `exploracion.ipynb` para revisar datos y gráficos.**

El warning inmediato se activa con cualquier sensor ≥16. El trip se activa con cualquier sensor ≥20. La alerta ML no modifica las protecciones.

En septiembre el modelo anticipó 4 de 5 detenciones, con mediana de 14,5 horas y un falso episodio. El warning de 16 anticipó las mismas cuatro con mediana de 19 horas. Esta ejecución no demuestra superioridad global del ML; en un evento sí agregó 7 horas respecto del warning. La falla repentina no tuvo precursor detectable.

## Regenerar

```bash
python -m pip install -r requirements.txt
python overland.py --out resultados
```

Parámetros: `--seed 42`, `--horizon 24`, `--step-minutes 60` (15, 30 o 60).

## Aplicar el modelo a un histórico nuevo

El archivo necesita timestamp explícito UTC, las columnas R1_S1, R1_S2, R1_S3, R2_S1, R2_S2, R2_S3 y operando (0/1) proveniente del estado de marcha. Misma unidad y cadencia que el entrenamiento. Se rechazan intervalos faltantes o irregulares. En producción hace falta una política validada de calidad de datos.

```bash
python predecir.py historico_6_meses.csv --model modelo.joblib --out predicciones_nuevas.csv
```

Las estadísticas usan únicamente la muestra actual y anteriores; se requieren 24 horas continuas de operación. La persistencia se calcula desde el histórico que se pasa al comando. Para una ejecución por hora, conservar al menos 24 horas más dos muestras y el estado de marcha. El modelo conserva sólo señales de vibración: el tipo de falla y su inicio no se utilizan.

## Contenido

Código de simulación/entrenamiento/inferencia, notebook, CSV de sensores y predicciones, modelo, métricas, informe y gráficos. Python 3.12; dependencias fijadas a las versiones usadas. Jupyter es opcional y se instala aparte para abrir el notebook.

Datos completamente artificiales y pequeño conjunto de prueba. Validar con datos y fallas reales antes de uso operativo. `modelo.joblib` es un artefacto Python para cargar únicamente desde una fuente confiable.
