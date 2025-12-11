import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import config

def load_raw_data():
    """Carga el dataset original."""
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(f"El archivo no existe en: {config.RAW_DATA_PATH}")
    
    print(f"🔄 Cargando datos desde: {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"✅ Datos cargados. Shape inicial: {df.shape}")
    return df

def clean_data(df):
    """Limpieza integral del dataset."""
    df = df.copy()
    
    # 1. Duplicados
    df = df.drop_duplicates()
    
    # 2. Nulos
    if 'company' in df.columns: df = df.drop(columns=['company'])
    if 'agent' in df.columns: df['agent'] = df['agent'].fillna(0)
    if 'country' in df.columns: df['country'] = df['country'].fillna('Unknown')
    if 'children' in df.columns: df['children'] = df['children'].fillna(0)
        
    # 3. Leakage
    leakage_cols = ['reservation_status', 'reservation_status_date']
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    return df

def encode_data(df):
    """One-Hot Encoding para separar X e y."""
    print("⚙️ Transformando variables categóricas...")
    
    target = 'is_canceled'
    if target not in df.columns:
        raise ValueError("Falta la columna target")
        
    y = df[target]
    X = df.drop(columns=[target])
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"✅ Encoding completado. Features: {X_encoded.shape[1]}")
    return X_encoded, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    1. Divide en Train y Test (80% / 20%).
    2. Escala las variables numéricas para que tengan media 0 y desviación 1.
    IMPORTANTE: El scaler se entrena solo con X_train para evitar data leakage.
    """
    print(f"✂️ Dividiendo datos (Test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("⚖️ Escalando datos (StandardScaler)...")
    scaler = StandardScaler()
    
    # Aprendemos la escala solo del train y transformamos ambos
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertimos de nuevo a DataFrame para no perder los nombres de las columnas
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("✅ Datos listos para entrar al modelo.")
    print(f"   Train shape: {X_train_df.shape}")
    print(f"   Test shape:  {X_test_df.shape}")
    
    return X_train_df, X_test_df, y_train, y_test