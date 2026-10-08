# pages/clusters.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🧩 Análisis de Clusters")

BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"
PROCESSED = BASE_PATH / "processed"

st.write(f"Buscando `kmeans_cluster.pkl` en `{MODELS_PATH}` y datos en `{PROCESSED}`")

kmeans_path = MODELS_PATH / "kmeans_cluster.pkl"
if not kmeans_path.exists():
    st.warning("No se encontró `kmeans_cluster.pkl`. Ejecuta el entrenamiento para generarlo.")
    st.stop()

kmeans = joblib.load(kmeans_path)

# If cluster centers exist, show them
if hasattr(kmeans, "cluster_centers_"):
    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers["cluster"] = centers.index
    st.subheader("Centroides de KMeans")
    if centers.shape[1] >= 3:
        fig = px.scatter(centers, x=0, y=1, color="cluster", title="Centroides (dim 0 vs 1)")
    else:
        fig = px.bar(centers, x="cluster", y=0, title="Centroides - componente 0")
    fig.update_layout(width="stretch")
    st.plotly_chart(fig)
else:
    st.info("El objeto kmeans cargado no tiene `cluster_centers_`.")

# If we have a processed dataset, show labeled sample
proc_file = PROCESSED / "metadataset_enriched.csv"
if proc_file.exists():
    df = pd.read_csv(proc_file)
    # try to predict clusters for a sample using same representation as training (first columns)
    try:
        sample = df.select_dtypes(include="number").iloc[:500, :min(12, df.select_dtypes("number").shape[1])]
        labels = kmeans.predict(sample.fillna(0).values)
        sample = sample.copy()
        sample["cluster"] = labels
        sample_agg = sample.groupby("cluster").mean().reset_index()
        st.subheader("Muestra agregada por cluster (medias)")
        st.dataframe(sample_agg)
        fig2 = px.scatter(sample.reset_index(), x=sample.columns[0], y=sample.columns[1] if sample.shape[1]>1 else sample.columns[0],
                          color="cluster", title="Muestra etiquetada por cluster")
        fig2.update_layout(width="stretch")
        st.plotly_chart(fig2)
    except Exception as e:
        st.warning(f"No se pudieron predecir clusters sobre la muestra: {e}")
else:
    st.info("No se encontró `processed/metadataset_enriched.csv` para mostrar ejemplos.")
