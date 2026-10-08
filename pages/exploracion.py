# pages/exploracion.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔎 Exploración de Datos (EDA)")

BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data"

st.sidebar.header("Opciones EDA")
csv_files = sorted(DATA_PATH.glob("*.csv")) if DATA_PATH.exists() else []

if not csv_files:
    st.error("No se encontraron CSVs en la carpeta `data/`. Sube datasets en la página de carga.")
    st.stop()

file = st.sidebar.selectbox("Dataset", csv_files, format_func=lambda x: x.name)
st.write(f"Dataset seleccionado: `{file.name}`")

df = pd.read_csv(file)
st.subheader("Vista previa")
st.dataframe(df.head())

st.subheader("Descripción")
st.write(df.describe(include='all'))

# Column chooser for numeric scatter
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) >= 2:
    st.subheader("Scatter: seleccionar variables")
    col_x = st.selectbox("X", numeric_cols, index=0)
    col_y = st.selectbox("Y", numeric_cols, index=1)
    fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}")
    fig.update_layout(width="stretch")
    st.plotly_chart(fig)

# Histogram of a chosen numeric column
if numeric_cols:
    st.subheader("Histograma")
    col_h = st.selectbox("Histograma - columna", numeric_cols)
    fig2 = px.histogram(df, x=col_h, nbins=50, title=f"Histograma {col_h}")
    fig2.update_layout(width="stretch")
    st.plotly_chart(fig2)
