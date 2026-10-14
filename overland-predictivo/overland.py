"""Simulación, entrenamiento y evaluación temporal. No controla equipos reales."""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/overland_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.inspection import permutation_importance

SENSORS = [f'R{r}_S{s}' for r in (1, 2) for s in (1, 2, 3)]
WARNING, TRIP = 16., 20.


def simulate(seed=42, step_minutes=60):
    rng = np.random.default_rng(seed)
    step = step_minutes / 60
    dates = pd.date_range('2026-04-01', '2026-10-01', freq=f'{step_minutes}min', inclusive='left', tz='UTC')
    n = len(dates)
    hours = np.arange(n) * step
    common = .65 * np.sin(hours * 2 * np.pi / 24) + .3 * np.sin(hours * 2 * np.pi / 168)
    values = 6.3 + common[:, None] + rng.normal(0, .35, (n, 6)) + rng.uniform(-.7, .7, 6)
    metadata = []
    # Treinta episodios artificiales para demostrar el método: no estima tasa real de fallas.
    for k, center_h in enumerate(np.linspace(110, hours[-1] - 55, 30)):
        end = int((center_h + rng.uniform(-10, 10)) / step)
        sensor = int(rng.integers(6))
        sudden = k % 6 == 5
        duration = int(rng.uniform(48, 85) / step)
        start = end if sudden else end - duration
        if not sudden:
            z = np.linspace(0, 1, end - start)
            ramp = 13.8 * z ** rng.uniform(1.1, 1.8)
            values[start:end, sensor] += ramp
            peers = [j for j in range((sensor // 3)*3, (sensor // 3)*3+3) if j != sensor]
            for peer in peers:
                values[start:end, peer] += ramp * rng.uniform(.22, .5)
            values[start:end, sensor] = np.minimum(values[start:end, sensor], 19.7)
        values[end, sensor] = rng.uniform(20.1, 21.2)
        down = int(rng.uniform(8, 16) / step)
        values[end+1:end+1+down] = rng.uniform(.05, .25, (down, 6))
        metadata.append({'evento_id': k+1, 'indice': end, 'timestamp': dates[end].isoformat(),
                         'sensor_causante': SENSORS[sensor], 'patron_simulado': 'repentina' if sudden else 'gradual',
                         'inicio_deterioro': dates[start].isoformat(), 'horas_mantenimiento': down*step})
    # Transitorios benignos en zonas alejadas de los deterioros y mantenimientos.
    centers = np.arange(int(30/step), n-int(30/step), int(48/step))
    for center in centers:
        if any(abs(center-e['indice'])*step < 95 for e in metadata):
            continue
        sensor = int(rng.integers(6))
        width = int(rng.uniform(2, 5) / step)
        ix = np.arange(center-width, center+width+1)
        values[ix, sensor] += rng.uniform(9.5, 11.5) * np.exp(-((ix-center)/(width*.65))**2)
        values[ix, sensor] = np.minimum(values[ix, sensor], 19.2)
    # Algunos transitorios adicionales al principio de ciclos de operación.
    for e in metadata[:-1]:
        center = e['indice'] + int(26/step)
        sensor = int(rng.integers(6))
        width = max(1, int(2/step))
        ix = np.arange(center-width, center+width+1)
        values[ix, sensor] += 10*np.exp(-((ix-center)/width)**2)
        values[ix, sensor] = np.minimum(values[ix, sensor], 19.2)
    df = pd.DataFrame(np.round(np.clip(values, 0, None), 3), columns=SENSORS, index=dates)
    df.index.name = 'timestamp'
    df['max_vibracion'] = df[SENSORS].max(axis=1)
    df['trip'] = (df.max_vibracion >= TRIP).astype(int)
    # El trip se registra en la fila de cruce. El mantenimiento comienza en la siguiente.
    df['operando'] = (df.max_vibracion > 1).astype(int)
    df['warning'] = ((df.max_vibracion >= WARNING) & (df.max_vibracion < TRIP)).astype(int)
    df['estado'] = np.select([df.trip.eq(1), df.operando.eq(0), df.warning.eq(1)],
                             ['TRIP', 'MANTENIMIENTO', 'WARNING'], default='NORMAL')
    assert df.trip.sum() == len(metadata), 'Trip inesperado: revisar simulación.'
    return df, pd.DataFrame(metadata)


def features(df, step_minutes=60):
    """Sólo datos disponibles hasta t; reinicia ventanas después del mantenimiento."""
    step = step_minutes / 60
    groups = df.operando.ne(df.operando.shift()).cumsum()
    parts = []
    for _, seg in df.groupby(groups):
        if seg.operando.iloc[0] == 0:
            continue
        x = pd.DataFrame(index=seg.index)
        for sensor in SENSORS:
            v = seg[sensor]
            x[sensor] = v
            for h in (6, 12, 24):
                w = int(h/step)
                x[f'{sensor}_media_{h}h'] = v.rolling(w).mean()
                x[f'{sensor}_std_{h}h'] = v.rolling(w).std()
                x[f'{sensor}_pendiente_{h}h'] = (v-v.shift(w)) / h
            x[f'{sensor}_delta_1h'] = v-v.shift(int(1/step))
            x[f'{sensor}_persistencia_14_6h'] = v.ge(14).rolling(int(6/step)).mean()
        for r in (1, 2):
            cols = [f'R{r}_S{s}' for s in (1, 2, 3)]
            x[f'R{r}_max'] = seg[cols].max(axis=1)
            x[f'R{r}_dispersion'] = seg[cols].max(axis=1)-seg[cols].min(axis=1)
        x['max_global'] = seg[SENSORS].max(axis=1)
        parts.append(x)
    return pd.concat(parts).reindex(df.index)


def target(df, horizon=24, step_minutes=60):
    """Trip en (t, t+horizonte]; últimos h sin futuro completo quedan sin etiqueta."""
    w = int(horizon*60/step_minutes)
    y = df.trip.shift(-1).iloc[::-1].rolling(w, min_periods=w).max().iloc[::-1]
    return y


def alert_policy(scores, active, threshold, persistence=2):
    """La alerta se activa tras N muestras consecutivas de riesgo alto."""
    above = scores.ge(threshold) & active
    return above.rolling(persistence, min_periods=persistence).sum().eq(persistence)


def episode_metrics(index, alerts, events, horizon):
    alert = alerts.reindex(index, fill_value=False)
    starts = alert & ~alert.shift(1, fill_value=False)
    start_times = index[starts.to_numpy()]
    event_times = pd.to_datetime(events.timestamp, utc=True)
    leads, rows = [], []
    for (_, ev), t in zip(events.iterrows(), event_times):
        hits = index[alert.to_numpy() & (index < t) & (index >= t-pd.Timedelta(hours=horizon))]
        lead = (t-hits[0]).total_seconds()/3600 if len(hits) else None
        if lead is not None:
            leads.append(lead)
        rows.append({'evento_id': int(ev.evento_id), 'timestamp_trip': t.isoformat(),
                     'patron': ev.patron_simulado, 'anticipacion_h': lead})
    false = sum(not any(0 < (t-a).total_seconds()/3600 <= horizon for t in event_times) for a in start_times)
    return {'eventos': len(events), 'eventos_detectados': len(leads),
            'recall_eventos': len(leads)/len(events) if len(events) else None,
            'anticipacion_mediana_h': float(np.median(leads)) if leads else None,
            'falsos_episodios': int(false), 'episodios_alerta': int(starts.sum()),
            'horas_en_alerta': float(alert.sum() * ((index[1]-index[0]).total_seconds()/3600))}, rows


def run(out, seed=42, horizon=24, step_minutes=60):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, events = simulate(seed, step_minutes)
    X = features(df, step_minutes)
    y = target(df, horizon, step_minutes)
    eligible = X.notna().all(axis=1) & y.notna() & df.operando.eq(1) & df.trip.eq(0)
    val_start, test_start = pd.Timestamp('2026-08-01', tz='UTC'), pd.Timestamp('2026-09-01', tz='UTC')
    gap = pd.Timedelta(hours=horizon)
    train = eligible & (df.index < val_start-gap)
    valid = eligible & (df.index >= val_start) & (df.index < test_start-gap)
    test = eligible & (df.index >= test_start)
    assert train.any() and valid.any() and test.any()
    assert y[train].nunique() == 2
    # Features de validación/prueba pueden mirar su pasado, nunca su futuro.
    model = RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=8,
                                   class_weight='balanced_subsample', random_state=seed, n_jobs=-1)
    model.fit(X.loc[train], y.loc[train].astype(int))
    available = X.notna().all(axis=1) & df.operando.eq(1) & df.trip.eq(0)
    risk = pd.Series(np.nan, index=df.index, name='score_riesgo')
    risk.loc[available] = model.predict_proba(X.loc[available])[:, 1]
    val_ix, test_ix = df.index[valid], df.index[test]
    val_events = events[(pd.to_datetime(events.timestamp, utc=True) >= val_start) &
                        (pd.to_datetime(events.timestamp, utc=True) < test_start-gap)]
    test_events = events[(pd.to_datetime(events.timestamp, utc=True) >= test_start) &
                         (pd.to_datetime(events.timestamp, utc=True) <= test_ix[-1]+gap)]
    tuning = []
    for threshold in np.arange(.15, .86, .05):
        alerts = alert_policy(risk, available, threshold)
        m, _ = episode_metrics(val_ix, alerts, val_events, horizon)
        # Coste explícito de demostración: omitir una parada cuesta 5 falsas alarmas.
        cost = 5*(m['eventos']-m['eventos_detectados'])+m['falsos_episodios']
        tuning.append({'umbral': float(round(threshold, 2)), 'coste': cost, **m})
    tune = pd.DataFrame(tuning).sort_values(['coste', 'anticipacion_mediana_h', 'horas_en_alerta'],
                                          ascending=[True, False, True])
    threshold = float(tune.iloc[0].umbral)
    alerts = alert_policy(risk, available, threshold)
    ml_metrics, event_rows = episode_metrics(test_ix, alerts, test_events, horizon)
    warning_metrics, warning_rows = episode_metrics(test_ix, df.warning.eq(1), test_events, horizon)
    # Baseline de extrapolación lineal: nivel actual + pendiente de 12h.
    slope_cols = [f'{s}_pendiente_12h' for s in SENSORS]
    projected = X[SENSORS].to_numpy() + np.maximum(X[slope_cols].to_numpy(), 0)*horizon
    linear = pd.Series(np.nanmax(np.nan_to_num(projected, nan=-100), axis=1) >= TRIP, index=df.index)
    linear_alerts = alert_policy(linear.astype(float), available, .5)
    linear_metrics, _ = episode_metrics(test_ix, linear_alerts, test_events, horizon)
    predicted = alerts.loc[test].astype(int)
    hourly = {'precision': float(precision_score(y[test], predicted, zero_division=0)),
              'recall': float(recall_score(y[test], predicted, zero_division=0)),
              'average_precision': float(average_precision_score(y[test], risk[test])),
              'prevalencia': float(y[test].mean()),
              'matriz_confusion_TN_FP_FN_TP': confusion_matrix(y[test], predicted, labels=[0,1]).ravel().tolist()}
    metrics = {'seed': seed, 'horizonte_h': horizon, 'frecuencia_min': step_minutes,
               'umbral_ml': threshold, 'persistencia_muestras': 2, 'filas_historico': len(df),
               'lecturas_sensores': len(df)*6, 'detenciones_totales': len(events),
               'filas_train_val_test': [int(train.sum()), int(valid.sum()), int(test.sum())],
               'ml_prueba': ml_metrics, 'warning_16_prueba': warning_metrics,
               'lineal_prueba': linear_metrics, 'metricas_horarias_prueba': hourly,
               'modelo_sklearn': __import__('sklearn').__version__}
    compare = pd.DataFrame(event_rows).rename(columns={'anticipacion_h': 'anticipacion_ml_h'})
    compare['anticipacion_warning_h'] = [r['anticipacion_h'] for r in warning_rows]
    compare['ganancia_ml_h'] = compare.anticipacion_ml_h-compare.anticipacion_warning_h
    df['objetivo_trip_proximas_h'] = y
    df['score_riesgo'] = risk
    df['predictor_disponible'] = available.astype(int)
    df['alerta_predictiva'] = alerts.astype(int)
    df['alerta_lineal'] = linear_alerts.astype(int)
    df['particion'] = np.select([train, valid, test], ['entrenamiento', 'validacion', 'prueba'], default='excluida')
    df['sensor_maximo'] = df[SENSORS].idxmax(axis=1)
    # Operación: prioridad absoluta al trip; score no se interpreta como probabilidad calibrada.
    df['accion'] = np.select([df.trip.eq(1), df.operando.eq(0), alerts, df.warning.eq(1), ~available],
                            ['DETENER', 'MANTENIMIENTO', 'INSPECCION_PREDICTIVA', 'WARNING', 'PREDICTOR_NO_DISPONIBLE'], default='NORMAL')
    df.to_csv(out/'historico_6_meses.csv')
    events.to_csv(out/'eventos_simulados.csv', index=False)
    compare.to_csv(out/'anticipacion_prueba.csv', index=False)
    tune.to_csv(out/'ajuste_umbral_validacion.csv', index=False)
    df.loc[test, ['max_vibracion', 'sensor_maximo', 'score_riesgo', 'alerta_predictiva', 'warning',
                 'objetivo_trip_proximas_h', 'accion']].to_csv(out/'predicciones_prueba.csv')
    (out/'metricas.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    joblib.dump({'model': model, 'features': X.columns.tolist(), 'threshold': threshold,
                 'horizon_hours': horizon, 'step_minutes': step_minutes, 'persistence_samples': 2,
                 'warning': WARNING, 'trip': TRIP}, out/'modelo.joblib')
    importance = permutation_importance(model, X.loc[valid], y.loc[valid], scoring='average_precision',
                                        n_repeats=4, random_state=seed, n_jobs=1)
    imp = pd.DataFrame({'variable': X.columns, 'caida_average_precision': importance.importances_mean,
                        'std': importance.importances_std}).sort_values('caida_average_precision', ascending=False)
    imp.to_csv(out/'importancia_variables.csv', index=False)
    plots(df, events, compare, threshold, out, horizon)
    write_report(metrics, compare, imp, out)
    return df, metrics


def plots(df, events, compare, threshold, out, horizon):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    for r, color in ((1, '#2563eb'), (2, '#16a34a')):
        axes[0].plot(df.index, df[[f'R{r}_S{s}' for s in (1,2,3)]].max(axis=1), color=color, lw=.65, label=f'Reductor {r}: máximo de 3 sensores')
    axes[0].axhline(16, color='#d97706', ls='--', label='Warning ≥16')
    axes[0].axhline(20, color='#dc2626', ls='--', label='Trip ≥20')
    axes[0].set(ylabel='Vibración (unidad arbitraria)', title='Correa overland · seis meses simulados · abril–septiembre 2026 (UTC)')
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(df.index, df.score_riesgo, color='#7c3aed', lw=.6, label='Score de riesgo; sólo septiembre es prueba')
    axes[1].axhline(threshold, color='#dc2626', ls='--', label=f'Umbral ML {threshold:.2f}')
    axes[1].set(ylabel='Score (sin calibrar)', ylim=(0,1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(pd.Timestamp('2026-08-01', tz='UTC'), color='gray', ls=':')
        ax.axvline(pd.Timestamp('2026-09-01', tz='UTC'), color='gray', ls=':')
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out/'historico.png', dpi=150)
    plt.close(fig)
    for _, row in compare.iterrows():
        t = pd.Timestamp(row.timestamp_trip)
        view = df.loc[t-pd.Timedelta(hours=48):t+pd.Timedelta(hours=5)]
        fig, ax = plt.subplots(2, 1, figsize=(11,6), sharex=True)
        for s in SENSORS:
            ax[0].plot(view.index, view[s], lw=1, label=s)
        ax[0].axhline(16, color='#d97706', ls='--')
        ax[0].axhline(20, color='#dc2626', ls='--')
        ax[0].set(ylabel='Vibración', title=f"Evento {row.evento_id} · {row.patron} · ML: {row.anticipacion_ml_h} h / warning: {row.anticipacion_warning_h} h")
        ax[0].legend(ncol=6, fontsize=8)
        ax[1].plot(view.index, view.score_riesgo, color='#7c3aed')
        ax[1].axhline(threshold, color='#dc2626', ls='--')
        ax[1].fill_between(view.index, 0, 1, where=view.alerta_predictiva.eq(1), color='#7c3aed', alpha=.15, label='Alerta predictiva activa')
        ax[1].set(ylabel='Score de riesgo', ylim=(0,1))
        ax[1].legend(fontsize=8)
        for a in ax:
            a.axvline(t, color='#dc2626', label='Trip')
            a.grid(alpha=.15)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out/f'evento_{int(row.evento_id)}.png', dpi=150)
        plt.close(fig)


def write_report(m, comparison, importance, out):
    def stat(v):
        return 'Sin detección' if v is None else f'{v:g}'
    table = '| Método | Detenciones anticipadas | Mediana de anticipación | Falsos episodios |\n|---|---:|---:|---:|\n'
    for key, name in [('ml_prueba','Random Forest'), ('warning_16_prueba','Warning ≥16'), ('lineal_prueba','Extrapolación lineal')]:
        a = m[key]
        table += f"| {name} | {a['eventos_detectados']}/{a['eventos']} | {stat(a['anticipacion_mediana_h'])} h | {a['falsos_episodios']} |\n"
    text = f'''# Correa overland: alertas antes de una detención

Demostración con datos completamente sintéticos. Dos reductores, tres sensores por reductor.
Un solo sensor ≥16 genera warning; un solo sensor ≥20 detiene toda la correa. El trip tiene prioridad.
La unidad de vibración no fue indicada: se usa una unidad arbitraria, sin asumir mm/s ni norma industrial.

## Histórico simulado

Abril a septiembre de 2026, seis meses completos con timestamps UTC. Frecuencia: {m['frecuencia_min']} minutos.
{m['filas_historico']:,} filas y {m['lecturas_sensores']:,} lecturas de sensores. Se introducen
{m['detenciones_totales']} detenciones, mantenimiento de 8–16 horas y recuperación posterior.
La fila TRIP conserva el valor que cruza 20; después los sensores bajan mientras el equipo está detenido.
Es un histórico acelerado para enseñar el modelo, no una estimación de frecuencia real de fallas.

Patrones simulados: aumento sostenido durante 48–85 horas, crecimiento correlacionado entre sensores
del mismo reductor, transitorios benignos de corta duración y cinco fallas repentinas sin precursor.
Estas últimas permiten observar el límite de un predictor basado exclusivamente en vibración.

## Predicción y validación

Objetivo: identificar, cuando la correa está funcionando y aún no ha alcanzado 20,
si habrá un trip en las próximas {m['horizonte_h']} horas. Se usan valores actuales, medias,
desviaciones, incrementos, pendientes de 6/12/24 h y persistencia sobre 14.
Las ventanas se reinician después del mantenimiento. Se requieren 24 horas de historial de operación;
durante arranque o pérdida de datos el predictor carece de score, aunque los umbrales siguen operando.

Entrenamiento: abril–julio; validación: agosto; prueba: septiembre. Se purgan las últimas
{m['horizonte_h']} horas del entrenamiento y validación para que sus etiquetas no usen eventos del período siguiente.
Se excluyen las últimas {m['horizonte_h']} horas del histórico porque no se conoce su futuro completo.
Filas elegibles train/validación/prueba: {m['filas_train_val_test']}.

Modelo Random Forest de 250 árboles. El score es una señal de riesgo sin calibrar, no una probabilidad validada.
El umbral {m['umbral_ml']:.2f} se eligió únicamente en agosto, minimizando
`5 × detenciones omitidas + falsos episodios`; los empates favorecen anticipación y menor tiempo en alerta.
Estos costes son una hipótesis de demostración y deben ajustarse con operación.
La alerta requiere dos muestras consecutivas sobre el umbral, separadas por {m['frecuencia_min']/60:g} h.
Los umbrales de warning y trip actúan de inmediato, sin esa persistencia.

## Resultado fuera de entrenamiento (septiembre)

{table}

Se cuenta como anticipada una detención si existe alerta en `(trip − horizonte, trip)`.
La anticipación es el tiempo entre la primera muestra en alerta dentro de esa ventana y el trip.
Un falso episodio es un inicio de alerta sin trip en las próximas {m['horizonte_h']} h; si una alerta
empieza demasiado temprano, ese inicio se cuenta falso aunque posteriormente coincida con una ventana real.
La comparación aplica a las mismas horas elegibles de prueba, incluyendo las restricciones de arranque.
Las medianas de métodos pueden usar distintos eventos: la tabla por evento permite comparar pares.

Métricas horarias ML: precision {m['metricas_horarias_prueba']['precision']:.3f},
recall {m['metricas_horarias_prueba']['recall']:.3f}, average precision
{m['metricas_horarias_prueba']['average_precision']:.3f}. Prevalencia positiva:
{m['metricas_horarias_prueba']['prevalencia']:.3f}. Son métricas por muestra, no por detención.

El conjunto de prueba contiene sólo cinco detenciones. El modelo anticipa cuatro, pero su mediana
de anticipación es menor que la del warning. En esta ejecución no demuestra una mejora global
frente a la protección por umbral; el evento 29 sí agrega siete horas de anticipación.

## Variables y patrones

Variables más influyentes en validación por permutación (caída en average precision):

'''
    for _, r in importance.head(8).iterrows():
        text += f"- {r.variable}: {r.caida_average_precision:.4f}.\n"
    text += '''
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
'''
    (out/'INFORME.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='resultados')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--step-minutes', type=int, choices=[15,30,60], default=60)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 48:
        parser.error('--horizon debe estar entre 1 y 48 horas')
    _, metrics = run(args.out, args.seed, args.horizon, args.step_minutes)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
