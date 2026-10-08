# pages/ipca.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🧱 IPCA por bloques (visualización)")

BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"

ipca_path = MODELS_PATH / "ipca_blocks.pkl"
if not ipca_path.exists():
    st.warning("No se encontró `ipca_blocks.pkl`. Ejecuta entrenamiento para generarlo.")
    st.stop()

ipca_blocks = joblib.load(ipca_path)

st.write(f"Bloques detectados: {list(ipca_blocks.keys())}")

for block, info in ipca_blocks.items():
    st.subheader(f"Bloque: {block}")
    try:
        ipca = info.get("ipca")
        cols = info.get("cols", [])
        comps = ipca.components_
        comp_df = pd.DataFrame(comps, columns=cols)
        # show small table and heatmap
        st.write("Componentes (primeras filas):")
        st.dataframe(comp_df.head())
        fig = px.imshow(comp_df, labels={"x":"features","y":"components"}, title=f"IPCA {block}")
        fig.update_layout(width="stretch")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Fallo visualizando bloque '{block}': {e}")
