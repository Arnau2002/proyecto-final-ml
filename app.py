import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config, predictor

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Cancelaciones Hoteleras",
    page_icon="🏨",
    layout="wide"
)

# Título y descripción
st.title("Sistema de Predicción de Cancelaciones")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning (**LightGBM**) para predecir la probabilidad de que una reserva de hotel sea cancelada.
""")

# Cargar modelo
@st.cache_resource
def load_artifacts():
    model = predictor.load(config.BEST_MODEL_PATH)
    defaults = predictor.load_feature_defaults()
    return model, defaults

try:
    model, feature_defaults = load_artifacts()
    st.success("Pipeline cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar los artefactos del modelo: {e}")
    st.stop()

# Sidebar para inputs
st.sidebar.header("Parámetros de la Reserva")

def user_input_features():
    lead_time = st.sidebar.slider("Días de antelación (Lead Time)", 0, 700, 30)
    total_of_special_requests = st.sidebar.slider("Peticiones especiales", 0, 5, 0)
    adr = st.sidebar.number_input("Tarifa diaria promedio (ADR)", 0.0, 500.0, 100.0)
    previous_cancellations = st.sidebar.number_input("Cancelaciones previas", 0, 10, 0)

    deposit_type = st.sidebar.selectbox("Tipo de Depósito", ["No Deposit", "Non Refund", "Refundable"])
    market_segment = st.sidebar.selectbox("Segmento de Mercado", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"])

    return {
        'lead_time': lead_time,
        'total_of_special_requests': total_of_special_requests,
        'adr': adr,
        'previous_cancellations': previous_cancellations,
        'deposit_type': deposit_type,
        'market_segment': market_segment
    }

user_inputs = user_input_features()
input_df = pd.DataFrame([user_inputs])

st.subheader("Datos de la Reserva")
st.write(input_df)

if st.button("Predecir Cancelación"):
    feature_frame = predictor.build_feature_frame(user_inputs, feature_defaults)
    probabilities = predictor.predict_proba(model, feature_frame)[0]
    cancel_prob = float(probabilities[1])

    st.subheader("Resultado de la Predicción")
    if cancel_prob > 0.5:
        st.error(f"Alta probabilidad de cancelación: {cancel_prob:.2%}")
    else:
        st.success(f"Baja probabilidad de cancelación: {cancel_prob:.2%}")

    st.progress(cancel_prob)

st.markdown("---")
st.markdown("Desarrollado para el Proyecto Final de ML.")
