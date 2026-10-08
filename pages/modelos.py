# pages/modelos.py
import streamlit as st
import joblib
from pathlib import Path
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧠 Modelos y Artefactos")

BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"

st.write(f"Buscando artefactos en `{MODELS_PATH}`")

if not MODELS_PATH.exists():
    st.error("Carpeta `models/` no existe. Coloca tus modelos allí.")
    st.stop()

pkl_files = sorted(MODELS_PATH.glob("*.pkl"))
if not pkl_files:
    st.warning("No se encontraron archivos .pkl en models/")
else:
    st.success(f"Se detectaron {len(pkl_files)} artefactos (.pkl).")

# Show list and preview of each model/artifact
for p in pkl_files:
    st.markdown("---")
    st.subheader(f"📦 {p.name}")
    try:
        obj = joblib.load(p)
        # basic info
        st.write(type(obj))
        # For sklearn-like models show n_features_in_
        if hasattr(obj, "n_features_in_"):
            st.write(f"➡️ Espera features: {obj.n_features_in_}")
        # Show attributes for common artifacts
        if isinstance(obj, (list, dict, tuple)):
            st.write(f"Contenido (preview): {str(obj)[:1000]}")
        else:
            try:
                st.write(obj)
            except Exception:
                st.write("Objeto no representable con .write()")
    except Exception as e:
        st.error(f"Error cargando {p.name}: {e}")

# Quick CSV of model summary
summary = []
for p in pkl_files:
    try:
        obj = joblib.load(p)
        summary.append({
            "file": p.name,
            "type": str(type(obj)).split("'")[1] if type(obj) else "unknown",
            "has_n_features": hasattr(obj, "n_features_in_")
        })
    except:
        summary.append({"file": p.name, "type": "load_error", "has_n_features": False})

st.markdown("---")
st.subheader("Resumen rápido")
st.dataframe(pd.DataFrame(summary))
