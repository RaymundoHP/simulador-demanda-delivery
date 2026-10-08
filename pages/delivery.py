# pages/delivery.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")

# ================================
#  TITULO
# ================================
st.title("🚴‍♂️ Análisis de Delivery y Tiempos de Entrega")

# ================================
#  CARGA DEL DATASET
# ================================

DATA_PATH = Path("data/food_delivery_time.csv")

if not DATA_PATH.exists():
    st.error(
        f"❌ No se encontró el archivo `{DATA_PATH}`.\n\n"
        "Verifica que exista en la carpeta `data/`."
    )
    st.stop()

data = pd.read_csv(DATA_PATH)

st.subheader("📄 Vista de Datos de Delivery")
st.dataframe(data.head(), use_container_width=False)

# ================================
#  GRÁFICAS
# ================================

st.markdown("---")
st.subheader("📊 Análisis Exploratorio")

# --- Tiempo promedio por ciudad ---
if "City" in data.columns and "Delivery_Time" in data.columns:
    avg_time = (
        data.groupby("City")["Delivery_Time"]
        .mean()
        .reset_index()
        .sort_values("Delivery_Time", ascending=False)
    )

    fig = px.bar(
        avg_time,
        x="City",
        y="Delivery_Time",
        title="⏱ Tiempo promedio de entrega por ciudad",
        color="Delivery_Time",
    )
    fig.update_layout(width="stretch")

    st.plotly_chart(fig)

else:
    st.warning("⚠ No se encontraron las columnas 'City' y 'Delivery_Time'.")

# --- Comparación por tipo de vehículo ---
if "Vehicle_Type" in data.columns and "Delivery_Time" in data.columns:
    fig2 = px.box(
        data,
        x="Vehicle_Type",
        y="Delivery_Time",
        title="🚚 Distribución del tiempo de entrega por tipo de vehículo",
    )
    fig2.update_layout(width="stretch")

    st.plotly_chart(fig2)

else:
    st.warning("⚠ No se encontraron las columnas 'Vehicle_Type' y 'Delivery_Time'.")


# ================================
#  ANALISIS ADICIONAL (opcional)
# ================================
if "Weather" in data.columns:
    st.markdown("---")
    st.subheader("🌦 Efecto del clima en los tiempos de entrega")

    fig3 = px.box(
        data,
        x="Weather",
        y="Delivery_Time",
        title="Impacto del clima en los tiempos de entrega",
    )
    fig3.update_layout(width="stretch")
    st.plotly_chart(fig3)
