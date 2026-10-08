# pages/acercade.py
import streamlit as st

st.set_page_config(layout="wide")
st.title("ℹ️ Acerca del Proyecto")

st.markdown("""
**Dashboard de Predicción de Demanda y Delivery de Alimentos**  
Versión: 1.0  
Autor: Ray (adaptado por ChatGPT)

### Objetivo
Construir un sistema reproducible para:
- Preparación de datos (feature engineering y merges entre datasets)
- Reducción dimensional por bloques (Incremental PCA / IPCA)
- Segmentación (KMeans / otros)
- Entrenamiento y comparación de modelos (RF, LGBM, XGB, LinearSVR, Stacking/Voting)
- Integración en dashboard Streamlit con páginas para EDA, modelos y predicción

### Notas
- Antes de usar predicciones en producción, valida las columnas (`feature_columns.pkl`) y las transformaciones guardadas.
- Si falta algún `.pkl` o CSV, revisa la carpeta `models/` y `data/`.
""")
