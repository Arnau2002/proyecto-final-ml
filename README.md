# Proyecto Final – Machine Learning Supervisado

## 1. Objetivo del proyecto
El objetivo de este proyecto es desarrollar un modelo de **clasificación supervisada** capaz de predecir si una reserva de hotel será cancelada o no (variable objetivo: `is_canceled`). Esto permite a la cadena hotelera optimizar la gestión de habitaciones y reducir pérdidas por overbooking o habitaciones vacías.

El proyecto aplica un pipeline completo de Machine Learning que incluye preprocesado, entrenamiento, evaluación y comparación de modelos, siguiendo las mejores prácticas de la industria.

---

## 2. Dataset
- **Formato:** CSV  
- **Ubicación:** `data/raw/dataset_practica_final.csv`  

El dataset contiene variables numéricas y categóricas que han sido tratadas mediante técnicas de limpieza, codificación y escalado para su uso en modelos de Machine Learning.

---

## 3. Estructura del proyecto

```
proyecto-final-ml/
│
├── data/
│   └── raw/
│       └── dataset_practica_final.csv
│
├── notebooks/
│   └── exploracion/
│       └── preparacion_datos.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   └── predictor.py
│
├── models/
│   └── *.joblib
│
├── outputs/
│   └── *.txt
│
├── requirements.txt
└── README.md
```

---

## 4. Instalación y ejecución

### Requisitos
- Python 3.10 o superior
- Jupyter Lab / Notebook

### Instalación de dependencias
```bash
python -m pip install -r requirements.txt
```

### Ejecución
```bash
jupyter lab
```

Abrir el notebook:
```
notebooks/exploracion/preparacion_datos.ipynb
```

Y ejecutar:
```
Run → Restart Kernel and Run All
```

---

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

---

## 6. Resultados
Los modelos se evaluaron utilizando métricas como **accuracy**, **F1-score** y **AUC**.

- La Regresión Logística se utilizó como modelo de referencia.
- **LightGBM** obtuvo el mejor rendimiento global, superando ligeramente a XGBoost y Random Forest.
- Se seleccionó **LightGBM como modelo final** para la fase de evaluación.

---

## 7. Conclusiones
El modelo Random Forest ofrece el mejor equilibrio entre rendimiento y robustez para el problema planteado, superando al modelo baseline. El pipeline desarrollado es reproducible y fácilmente extensible a futuros experimentos.

