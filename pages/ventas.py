# pages/ventas.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")

# =====================================
#  TÍTULO
# =====================================
st.title("🛒 Análisis de Ventas y Consumo de Usuarios")

# =====================================
#  CARGA DEL DATASET
# =====================================

DATA_PATH = Path("data/online_food.csv")

if not DATA_PATH.exists():
    st.error(
        f"❌ No se encontró el archivo `{DATA_PATH}`.\n\n"
        "Verifica que esté en la carpeta `data/`."
    )
    st.stop()

data = pd.read_csv(DATA_PATH)

st.subheader("📄 Vista general del dataset")
st.dataframe(data.head())


# =====================================
#  KPIs (Indicadores clave)
# =====================================

st.markdown("---")
st.subheader("📌 Indicadores Generales")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de registros", f"{len(data):,}")

with col2:
    if "Age" in data.columns:
        st.metric("Edad promedio", round(data["Age"].mean(), 2))
    else:
        st.metric("Edad promedio", "N/A")

with col3:
    if "Output" in data.columns:
        st.metric("Compras realizadas (1 = Sí)", data["Output"].sum())
    else:
        st.metric("Compras realizadas", "N/A")


# =====================================
#  GRÁFICAS
# =====================================

st.markdown("---")
st.subheader("📊 Análisis Exploratorio de Ventas")


# --- Frecuencia de compra por género ---
if "Gender" in data.columns and "Output" in data.columns:
    purchase_by_gender = (
        data.groupby("Gender")["Output"]
        .mean()
        .reset_index()
        .sort_values("Output", ascending=False)
    )

    fig1 = px.bar(
        purchase_by_gender,
        x="Gender",
        y="Output",
        title="💬 Frecuencia de compra por género",
        color="Output",
    )
    fig1.update_layout(width="stretch")
    st.plotly_chart(fig1)
else:
    st.warning("⚠ No se encontraron las columnas 'Gender' y 'Output'.")


# --- Preferencia de restaurante por edad ---
if "Age" in data.columns and "Restaurant" in data.columns:
    fig2 = px.scatter(
        data,
        x="Age",
        y="Restaurant",
        title="👥 Preferencia de restaurante según edad",
        color="Restaurant",
    )
    fig2.update_layout(width="stretch")
    st.plotly_chart(fig2)
else:
    st.warning("⚠ No se encontraron las columnas 'Age' y 'Restaurant'.")


# --- Frecuencia de pedidos por ocupación ---
if "Occupation" in data.columns:
    occ_counts = data["Occupation"].value_counts().reset_index()
    occ_counts.columns = ["Occupation", "Count"]

    fig3 = px.bar(
        occ_counts,
        x="Occupation",
        y="Count",
        title="💼 Frecuencia de pedidos por ocupación",
    )
    fig3.update_layout(width="stretch")
    st.plotly_chart(fig3)
else:
    st.warning("⚠ No se encontró la columna 'Occupation'.")


# =====================================
#  GRÁFICA ADICIONAL (si aplica)
# =====================================

if "Marital_Status" in data.columns and "Output" in data.columns:
    st.markdown("---")
    st.subheader("❤️ Impacto del estado civil en compras")

    ms = (
        data.groupby("Marital_Status")["Output"]
        .mean()
        .reset_index()
        .sort_values("Output", ascending=False)
    )

    fig4 = px.bar(
        ms,
        x="Marital_Status",
        y="Output",
        title="Comparativa de compras por estado civil",
        color="Output",
    )
    fig4.update_layout(width="stretch")
    st.plotly_chart(fig4)
