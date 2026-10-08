# pages/dataset_loader.py
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(layout="wide")
st.title("📥 Cargar nuevos datasets")

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

uploaded = st.file_uploader("Arrastra o selecciona un CSV para subir", type=["csv"])
if uploaded:
    save_path = DATA_DIR / uploaded.name
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Archivo guardado en `{save_path}`")
    # preview
    try:
        df = pd.read_csv(save_path, nrows=200)
        st.dataframe(df.head())
    except Exception as e:
        st.warning(f"No se pudo leer el CSV guardado: {e}")

# List current files
st.markdown("---")
st.subheader("Archivos en data/")
files = list(DATA_DIR.glob("*.csv"))
if files:
    st.write([f.name for f in files])
else:
    st.info("No hay archivos en data/")
