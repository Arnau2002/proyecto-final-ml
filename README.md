# Proyecto Final – Machine Learning Supervisado


**Autores:** Jordi Gras y Arnau Climent


* **Jordi Gras (Ingeniería):** Encargado de transformar los notebooks experimentales en código de producción (`src`), crear el pipeline automatizado `train_pipeline.py` y desarrollar el bonus de la interfaz gráfica (`app.py`).
* **Arnau Climent (Ciencia de Datos):** Encargado del análisis estadístico inicial, la decisión de eliminar variables con fuga de información (Leakage) y la interpretación de la matriz de confusión.


## 1. Objetivo del proyecto
El objetivo de este proyecto es desarrollar un modelo de **clasificación supervisada** capaz de predecir si una reserva de hotel será cancelada o no (variable objetivo: `is_canceled`). Esto permite a la cadena hotelera optimizar la gestión de habitaciones y reducir pérdidas por overbooking o habitaciones vacías.

El proyecto aplica un pipeline completo de Machine Learning que incluye preprocesado, entrenamiento, evaluación y comparación de modelos, siguiendo las mejores prácticas de la industria.

-------

## 2. Dataset
- **Formato:** CSV  
- **Ubicación:** `data/raw/dataset_practica_final.csv`  

El dataset contiene variables numéricas y categóricas que han sido tratadas mediante técnicas de limpieza, codificación y escalado para su uso en modelos de Machine Learning.

-----

## 3. Estructura del proyecto

```
proyecto-final-ml/
│
├── data/
|   └── processed/
│   └── raw/
|
├── docs/
│   └── informe.md
|
├── models/
|   └── artifacts/
|       └── feature_defaults.json/
│   └── tests/
│   └── *nn_mlp.keras
|
├── notebooks/
│   └── exploracion/
│       └── eda_inicial.ipynb
│       └── preparacion_datos.ipynb
│   └── finales/
│
├── outputs/
│   └── baseline_report.txt
│   └── lightgbm_report.txt
│   └── model_comparison.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   └── predictor.py
│   └── preprocessing.py
│   └── train_pipeline.py
|
|
├── .gitignore
└── README.md
├── requirements.txt
└── app.py
└── setup.py
```

--------

## 4. Instalación y ejecución

### Requisitos
- Python 3.10 o superior
- Jupyter Lab / Notebook

### Instalación de dependencias
```bash
python -m pip install -r requirements.txt
```

### Entrenamiento reproducible (pipeline completo)
Genera el pipeline de LightGBM, el reporte actualizado y los valores por defecto para inferencia:
```bash
python -m src.train_pipeline
```

### Aplicación Streamlit
Con los artefactos anteriores creados, levanta la interfaz web para realizar predicciones reales:
```bash
streamlit run app.py
```

### Notebooks
Los notebooks de exploración siguen disponibles para revisar el proceso manual. Desde VS Code / Jupyter Lab abre `notebooks/exploracion/preparacion_datos.ipynb` y ejecuta `Run → Restart Kernel and Run All`.

-------

## 5. Metodología

### 5.1 Preprocesado de datos
- Limpieza del dataset
- Codificación de variables categóricas
- Separación de variables independientes (X) y variable objetivo (y)
- Escalado de variables
- División en conjunto de entrenamiento y test

### 5.2 Modelos entrenados
- **Modelo baseline:** Regresión Logística
- **Modelos avanzados:**
  - Árbol de Decisión
  - Random Forest
  - XGBoost
  - LightGBM (Mejor rendimiento)
  - Red Neuronal (MLP)

-------

## 6. Resultados
Los modelos se evaluaron utilizando métricas como **accuracy**, **F1-score** y **AUC**. La Regresión Logística se empleó como baseline, mientras que los modelos basados en árboles obtuvieron los mejores resultados.

El pipeline de **LightGBM** (preprocesamiento + modelo) alcanzó una accuracy del **85.4%** en el conjunto de test, superando a XGBoost, Random Forest, Decision Tree y la red neuronal. El reporte se encuentra en `outputs/lightgbm_report.txt` y la comparación histórica de modelos en `outputs/model_comparison.txt`.

------

## 7. Conclusiones
LightGBM ofrece el mejor equilibrio entre rendimiento y coste computacional para el problema de cancelaciones hoteleras, con una mejora clara sobre el baseline logístico. El pipeline entrenado es reproducible (`python -m src.train_pipeline`), serializa tanto el modelo como los valores por defecto necesarios para inferencia y se integra directamente con la aplicación Streamlit para exponer predicciones reales.

