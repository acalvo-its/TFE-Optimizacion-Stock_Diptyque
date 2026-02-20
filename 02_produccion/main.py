import os
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from google.cloud import bigquery
from category_encoders import TargetEncoder
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# CONFIGURACIÓN
PROJECT_ID = "automatizacion-boxnox"
DATASET_ID = "predicciones_diptyque"

# TABLAS
TABLA_VENTAS = f"{PROJECT_ID}.odoo_clean.ventas_tpv"
TABLA_STOCK = f"{PROJECT_ID}.odoo_clean.stock_completo"
TABLA_MAESTRO = f"{PROJECT_ID}.odoo_raw.maestro_productos"
DESTINATION_TABLE = f"{PROJECT_ID}.{DATASET_ID}.predicciones_finales"

puntos_venta_permitidos = ['AND', 'CAS', 'CC85', 'MAR', 'SAL', 'SER', 'ZGZ']

client = bigquery.Client()

def pipeline_prediccion():
    # 1. LOGICA DE VENTAS
    print("1/6 PROCESAMIENTO DE VENTAS")
    # Los filtros aplicados atienden a PFS activos de Diptyque, en los POS admitidos, evitando ventas corporativas
    query_ventas = f"""
    SELECT 
        CAST(v.fecha AS DATE) as fecha,
        v.punto_venta,
        v.vitalicio,
        v.sku,
        m.estado,
        SUM(CAST(v.cantidad AS FLOAT64)) as unidades_vendidas
    FROM `{TABLA_VENTAS}` v
    INNER JOIN `{TABLA_MAESTRO}` m ON v.sku = m.sku
    WHERE v.punto_venta IN UNNEST(@tiendas)
      AND v.PFS_PNFS = 'PFS'
      AND v.marca = 'DIPTYQUE'
      AND v.fecha IS NOT NULL
      AND m.estado = 'activo'
      AND CAST(v.cantidad AS FLOAT64) < 15 
    GROUP BY 1, 2, 3, 4, 5
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tiendas", "STRING", puntos_venta_permitidos)]
    )
    df_ventas = client.query(query_ventas, job_config=job_config).to_dataframe()
    
    df_ventas = df_ventas.drop(columns=['sku', 'estado'], errors='ignore')
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
    df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
    df_ventas = df_ventas.sort_values(['punto_venta', 'vitalicio', 'fecha'])

    # 2. GENERACIÓN DE VARIABLES
    print("2/6 GENERACIÓN DE VARIABLES Y SUMATORIOS")
    group_v = df_ventas.groupby(['punto_venta', 'vitalicio'])['unidades_vendidas']
    df_ventas['media_7dias'] = group_v.transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df_ventas['ventas_ayer'] = group_v.shift(1)
    df_ventas['ventas_7d_total'] = group_v.transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    
    años = df_ventas['fecha'].dt.year.unique().tolist()
    festivos_es = holidays.Spain(years=años)
    df_ventas['es_festivo'] = df_ventas['fecha'].apply(lambda x: 1 if x in festivos_es else 0)
    df_ventas = df_ventas.dropna(subset=['ventas_ayer']).fillna(0)

    # 3. LOGICA DE STOCK
    print("3/6 PROCESAMIENTO DE LA TABLA DE STOCK")
    query_stock = f"""
    SELECT 
        ubicacion as punto_venta,
        sku_vitalicio as vitalicio,
        MAX(CAST(gama AS STRING)) as gama,
        SUM(CAST(cantidad AS FLOAT64) - CAST(cantidad_reservada AS FLOAT64)) as stock_actual
    FROM `{TABLA_STOCK}`
    WHERE ubicacion IN UNNEST(@tiendas)
      AND PFS_PNFS = 'PFS'
      AND marca = 'DIPTYQUE'
    GROUP BY 1, 2
    """
    df_stock = client.query(query_stock, job_config=job_config).to_dataframe()

    # 4. ENTRENAMIENTO
    print("4/6 ENTRENAMIENTO DEL MODELO")
    features = ['dia_semana', 'es_festivo', 'ventas_ayer', 'media_7dias']
    encoder_tienda = TargetEncoder(cols=['punto_venta'], smoothing=10.0)
    df_ventas['punto_venta_enc'] = encoder_tienda.fit_transform(df_ventas['punto_venta'], df_ventas['unidades_vendidas'])
    df_ventas['vitalicio_enc'] = pd.factorize(df_ventas['vitalicio'])[0]
    
    X = df_ventas[['punto_venta_enc', 'vitalicio_enc'] + features].astype(float)
    y = df_ventas['unidades_vendidas'].astype(float)
    modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5)
    modelo.fit(X, y)

    # 5. PREDICCIONES
    print("5/6 GENERACIÓN DE PREDICCIONES Y FILTRO DE ENFRIAMIENTO")
    ultima_foto = df_ventas.groupby(['punto_venta', 'vitalicio']).last().reset_index()
    X_pred = ultima_foto[['punto_venta_enc', 'vitalicio_enc'] + features].astype(float)
    
    pred_diaria = modelo.predict(X_pred).clip(0)
    ultima_foto['demanda_7d'] = (pred_diaria * 7).round(2)
    fecha_limite = datetime.now() - timedelta(days=90)
    ultima_foto.loc[ultima_foto['fecha'] < fecha_limite, 'demanda_7d'] = 0

    # 6. CRUCE DE TABLAS
    print("6/6 CÁLCULO DE REPOSICIÓN OPTIMIZADA")
    final = pd.merge(
        ultima_foto[['punto_venta', 'vitalicio', 'demanda_7d', 'ventas_7d_total']], 
        df_stock, 
        on=['punto_venta', 'vitalicio'], 
        how='left'
    )

    final['gama'] = final['gama'].fillna('Sin Clasificar').astype(str)
    cols_numericas = ['stock_actual', 'demanda_7d', 'ventas_7d_total']
    final[cols_numericas] = final[cols_numericas].fillna(0)

    # 7. LOGICA DE STOCK PROYECTADO
    final['stock_proyectado'] = final['stock_actual'] + final['ventas_7d_total']
    final['stock_seguridad'] = (final['demanda_7d'] * 0.20).apply(np.ceil)
    final['unidades_a_reponer'] = (
        final['demanda_7d'] + final['stock_seguridad'] - final['stock_proyectado']
    ).clip(lower=0).apply(np.ceil)

    # Alertas
    condiciones = [
        (final['unidades_a_reponer'] > 0) & (final['stock_proyectado'] < (final['demanda_7d'] * 0.5)),
        (final['unidades_a_reponer'] > 0),
        (final['unidades_a_reponer'] == 0)
    ]
    etiquetas = ['Rotura', 'Sugerencia', 'Óptimo']
    final['alerta_stock'] = np.select(condiciones, etiquetas, default='Óptimo')

    # 8. CARGA
    final['ultima_actualizacion'] = datetime.now()
    job_config_bq = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    try:
        client.load_table_from_dataframe(final, DESTINATION_TABLE, job_config=job_config_bq).result()
        print(f"TABLA ACTUALIZADA CON ÉXITO: {DESTINATION_TABLE}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":

    pipeline_prediccion()
