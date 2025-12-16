import os
from pathlib import Path

# Directorio raíz del proyecto (2 niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset_practica_final.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "hotel_bookings_cleaned.csv"

# Rutas de modelos y resultados
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Artefactos auxiliares
FEATURE_DEFAULTS_PATH = ARTIFACTS_DIR / "feature_defaults.json"
BEST_MODEL_PATH = MODELS_DIR / "boost_lightgbm.joblib"

# Parámetros aleatorios (para reproducibilidad)
RANDOM_SEED = 42

# Columnas a eliminar por Data Leakage o irrelevancia (IDs)
COLS_TO_DROP = ['reservation_status', 'reservation_status_date', 'company', 'agent'] 
# Nota: 'company' y 'agent' tienen muchos nulos, a veces se eliminan o se imputan. 
# Para la carga inicial, podemos mantenerlos o quitarlos según decidas en el EDA.