import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import os
from src import data_loader

# Asegurar que existe el directorio de salida
os.makedirs("outputs", exist_ok=True)

# 1. Cargar datos
print("Cargando datos...")
df = data_loader.load_raw_data()
df = data_loader.clean_data(df)

# 2. Separar X e y (SIN ENCODING MANUAL)
# El modelo ya tiene un pipeline que hace el encoding por dentro
target = 'is_canceled'
X = df.drop(columns=[target])
y = df[target]

# 3. Split (Usamos random_state=42 para que sea el mismo split que en el entreno)
print("Dividiendo datos (Train/Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Cargar el modelo Pipeline
print("Cargando modelo...")
model_path = 'models/boost_lightgbm.joblib'
if not os.path.exists(model_path):
    raise FileNotFoundError("No se encuentra el modelo. Ejecuta 'python -m src.train_pipeline' primero.")

model = joblib.load(model_path)

# 5. Predicciones
print("Realizando predicciones...")
# Le pasamos X_test con texto, y el pipeline lo transforma solo
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Generar Matriz de Confusión
print("Generando Matriz de Confusión...")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - LightGBM')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.savefig('outputs/confusion_matrix.png')
plt.close() # Cerrar para liberar memoria
print("Guardado: outputs/confusion_matrix.png")

# 7. Generar Curva ROC
print("Generando Curva ROC...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('outputs/roc_curve.png')
plt.close()
print("Guardado: outputs/roc_curve.png")