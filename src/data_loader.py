import pandas as pd
import os
from src import config

def load_raw_data():
    """Carga el dataset original desde la ruta raw definida en config."""
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(f"El archivo no existe en: {config.RAW_DATA_PATH}")
    
    print(f"🔄 Cargando datos desde: {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"✅ Datos cargados. Shape: {df.shape}")
    return df

def basic_cleaning(df):
    """
    Realiza limpieza básica obligatoria:
    1. Eliminar duplicados.
    2. Eliminar columnas de Data Leakage.
    """
    # Copia para no modificar el original
    df_clean = df.copy()
    
    # 1. Eliminar duplicados
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"🧹 Eliminadas {initial_rows - len(df_clean)} filas duplicadas.")
    
    # 2. Eliminar Data Leakage (IMPORTANTE)
    # Verificamos si las columnas existen antes de borrarlas
    cols_drop = [c for c in config.COLS_TO_DROP if c in df_clean.columns]
    if cols_drop:
        df_clean = df_clean.drop(columns=cols_drop)
        print(f"🚫 Columnas eliminadas (Leakage/Irrelevantes): {cols_drop}")
        
    return df_clean