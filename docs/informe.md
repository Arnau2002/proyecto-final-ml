
## INFORME_FINAL


# Predicción de Cancelaciones Hoteleras

**Autores:** Jordi Gras y Arnau Climent
**Fecha:** Diciembre 2025



## 1. Definición de Roles y Colaboración
Siguiendo los requisitos de la práctica, hemos dividido el trabajo simulando un equipo de datos. 
* **Jordi Gras (Ingeniería):** Encargado de transformar los notebooks experimentales en código de producción (`src`), crear el pipeline automatizado `train_pipeline.py` y desarrollar el bonus de la interfaz gráfica (`app.py`).
* **Arnau Climent (Ciencia de Datos):** Encargado del análisis estadístico inicial, la decisión de eliminar variables con fuga de información (Leakage) y la interpretación de la matriz de confusión.

--------

## 2. Justificación del Problema
Las cancelaciones de último minuto suponen un problema financiero grave. El objetivo es predecir la probabilidad de que la variable `is_canceled` sea igual a 1. Se ha elegido la métrica **Accuracy** como principal, vigilando de cerca el **Recall**, ya que nos interesa detectar la mayor cantidad posible de cancelaciones reales.

-------

## 3. Análisis Exploratorio (EDA) y Limpieza
Del análisis realizado en `notebooks/exploracion`, destacamos tres decisiones críticas:

1.  **Data Leakage:** Detectamos que `reservation_status` tenía una correlación de 1.0 con el objetivo. Se eliminó para evitar un modelo inválido.
2.  **Gestión de Nulos:**
    * `company` (>94% nulos): Variable eliminada
    * `agent`: Se imputaron los nulos con 0 (reserva directa)
3.  **Ingeniería de Variables:** Se aplicó *One-Hot Encoding* a variables categóricas (como `market_segment` y `country`) y *StandardScaler* a las numéricas para facilitar la convergencia de la Red Neuronal y la Regresión Logística.

-------

## 4. Diseño del Sistema y Automatización
Para cumplir con el requisito de automatización, no dependemos de ejecución manual de celdas. Se ha implementado un pipeline en `src/train_pipeline.py` que realiza secuencialmente:
1.  Carga de datos (`data_loader`)
2.  Split Train/Test estratificado
3.  Entrenamiento de 5 modelos comparativos
4.  Serialización del mejor modelo en `.joblib`

Además, se ha implementado un **Bonus Técnico**: una interfaz en **Streamlit** que carga el modelo entrenado y permite al usuario introducir datos manualmente para obtener una predicción en tiempo real.

--------

## 5. Resultados y Comparativa
Se han entrenado y comparado los siguientes modelos:

| Modelo              | Accuracy  | Ventajas / Desventajas

| **LightGBM**        | **0.854** | **Entrenamiento rápido y alta precisión.**
| XGBoost             | 0.851     | Resultados sólidos, estándar de la industria
| Random Forest       | 0.849     | Buen rendimiento pero modelo muy pesado
| Red Neuronal (MLP)  | 0.835     | Requiere más datos y ajuste fino para superar a los árboles
| Regresión Logística | 0.805     | Modelo base. Útil para interpretabilidad lineal

**Conclusión:** Se selecciona **LightGBM** por ofrecer el mejor rendimiento y eficiencia computacional.

---------

## 6. Reflexión Final
El proyecto cumple con el flujo completo de un sistema de IA. Como mejora futura, proponemos implementar **MLflow** para un registro más detallado de experimentos y dockerizar la aplicación para su despliegue en nube (AWS/Azure).