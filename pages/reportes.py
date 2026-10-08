# pages/reportes.py
import streamlit as st
import pandas as pd
from pathlib import Path
import io
import datetime

st.set_page_config(layout="wide")
st.title("📑 Reportes y Exportación")

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "processed"
PROCESSED.mkdir(exist_ok=True)

st.markdown("Exporta resultados, metadatasets o predicciones.")

# Show processed files
files = list(PROCESSED.glob("*.csv"))
st.subheader("Metadatasets procesados")
if files:
    st.write([f.name for f in files])
    sel = st.selectbox("Selecciona para exportar", files, format_func=lambda x: x.name)
    df = pd.read_csv(sel)
    st.dataframe(df.head())

    if st.button("Exportar a CSV"):
        out = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", out, file_name=f"{sel.stem}_export_{datetime.date.today()}.csv", mime="text/csv")
else:
    st.info("No hay metadatasets procesados en processed/")

# Quick generate fake report (CSV)
st.markdown("---")
st.subheader("Generar reporte rápido de ejemplo")
df_example = pd.DataFrame({"metric": ["mae","rmse","r2"], "value": [1.2, 2.3, 0.85]})
st.dataframe(df_example)
csv = df_example.to_csv(index=False).encode()
st.download_button("Descargar reporte ejemplo (CSV)", csv, "report_example.csv", "text/csv")
