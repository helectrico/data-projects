"""Aplica el modelo a un histórico nuevo sin usar etiquetas futuras."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from overland import SENSORS, features, alert_policy


def predict(input_csv, model_file, output_csv):
    artifact = joblib.load(model_file)
    df = pd.read_csv(input_csv)
    required = ['timestamp', 'operando', *SENSORS]
    missing = sorted(set(required)-set(df.columns))
    if missing:
        raise ValueError(f'Columnas faltantes: {missing}')
    if df.empty:
        raise ValueError('El histórico está vacío')
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    df = df.set_index('timestamp')
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError('Timestamps deben ser únicos y estar ordenados')
    if len(df)>1 and not (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=artifact['step_minutes'])).all():
        raise ValueError('Frecuencia incorrecta o datos faltantes: revisar antes de predecir')
    if not df.operando.isin([0,1]).all():
        raise ValueError('operando debe ser 0 o 1 y provenir del estado de marcha')
    if not np.isfinite(df[SENSORS].to_numpy(dtype=float)).all() or (df[SENSORS]<0).any().any():
        raise ValueError('Sensores deben contener valores numéricos finitos y no negativos')
    df['max_vibracion'] = df[SENSORS].max(axis=1)
    df['trip'] = df.max_vibracion.ge(artifact['trip'])
    df['warning'] = df.max_vibracion.ge(artifact['warning']) & ~df.trip
    X = features(df, artifact['step_minutes']).reindex(columns=artifact['features'])
    usable = X.notna().all(axis=1) & df.operando.eq(1) & ~df.trip
    score = pd.Series(np.nan, index=df.index)
    if usable.any():
        score.loc[usable] = artifact['model'].predict_proba(X.loc[usable])[:,1]
    alerts = alert_policy(score, usable, artifact['threshold'], artifact['persistence_samples'])
    result = df[['operando', *SENSORS, 'max_vibracion', 'warning', 'trip']].copy()
    result['score_riesgo'] = score
    result['predictor_disponible'] = usable.astype(int)
    result['alerta_predictiva'] = alerts.astype(int)
    result['sensor_maximo'] = df[SENSORS].idxmax(axis=1)
    result['accion'] = np.select([df.trip, df.operando.eq(0), alerts, df.warning, ~usable],
                                ['DETENER', 'MANTENIMIENTO', 'INSPECCION_PREDICTIVA', 'WARNING', 'PREDICTOR_NO_DISPONIBLE'], default='NORMAL')
    result.to_csv(output_csv)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_csv', help='timestamp UTC, operando 0/1 y seis sensores; grilla regular')
    p.add_argument('--model', default='modelo.joblib')
    p.add_argument('--out', default='predicciones_nuevas.csv')
    args = p.parse_args()
    result = predict(args.input_csv, args.model, args.out)
    print(f'{len(result)} filas guardadas en {Path(args.out).resolve()}')
