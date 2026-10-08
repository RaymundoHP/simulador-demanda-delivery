# pages/comparativa_modelos.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔁 Comparativa de Modelos")

BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"
PROCESSED = BASE_PATH / "processed"

st.write("Cargando modelos desde `models/` (ignora artefactos)")

models = {}
for p in MODELS_PATH.glob("*.pkl"):
    name = p.stem
    if any(x in name for x in ["scaler", "pca", "ipca", "feature", "kmeans"]):
        continue
    try:
        models[name] = joblib.load(p)
    except Exception:
        pass

if not models:
    st.warning("No se encontraron modelos listos para comparación.")
    st.stop()

st.write(f"Modelos a comparar: {list(models.keys())}")

# If processed test sample exists, use it
test_path = PROCESSED / "metadataset_enriched.csv"
if test_path.exists():
    df_test = pd.read_csv(test_path).sample(n=min(500, 1000), random_state=42)
    st.info(f"Usando muestra de {test_path.name} para predicciones (n={len(df_test)})")
    # try to pick numeric features subset
    X = df_test.select_dtypes(include="number").fillna(0)
    res = {}
    for name, m in models.items():
        try:
            # match shape or adapt
            if hasattr(m, "n_features_in_") and X.shape[1] != m.n_features_in_:
                X_try = X.values[:, :m.n_features_in_]
            else:
                X_try = X.values
            preds = m.predict(X_try)
            res[name] = preds
        except Exception as e:
            st.warning(f"Modelo {name} falló: {e}")
    # build dataframe of first N rows
    if res:
        madef = pd.DataFrame({k: v[:100] for k, v in res.items()})
        madef["idx"] = madef.index
        melt = madef.melt(id_vars="idx", var_name="model", value_name="pred")
        fig = px.line(melt, x="idx", y="pred", color="model", title="Comparativa: primeras 100 muestras")
        fig.update_layout(width="stretch")
        st.plotly_chart(fig)
    else:
        st.warning("Ningún modelo pudo predecir la muestra.")
else:
    st.info("No se encontró processed/metadataset_enriched.csv. Puedes generar un sample manual en 'Dataset Loader'.")
