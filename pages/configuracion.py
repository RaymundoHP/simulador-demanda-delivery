# pages/configuracion.py
import streamlit as st
from pathlib import Path
import json

st.set_page_config(layout="wide")
st.title("⚙️ Configuración del Proyecto")

BASE = Path(__file__).resolve().parents[1]
CFG = BASE / "config.json"

st.markdown("Ajusta parámetros globales del proyecto (solo para uso local).")

# Load or create default
default = {
    "fast_debug": False,
    "rf_n_estimators": 150,
    "xgb_n_estimators": 150,
    "lgb_n_estimators": 150,
    "sample_fraction": 1.0
}

if CFG.exists():
    cfg = json.loads(CFG.read_text())
else:
    cfg = default

st.checkbox("Modo debug (reduce datos y modelos)", value=cfg.get("fast_debug", False), key="fast_debug")
st.number_input("RF - n_estimators", min_value=10, max_value=1000, value=cfg.get("rf_n_estimators", 150), key="rf_n")
st.number_input("XGB - n_estimators", min_value=10, max_value=1000, value=cfg.get("xgb_n_estimators", 150), key="xgb_n")
st.number_input("LGB - n_estimators", min_value=10, max_value=1000, value=cfg.get("lgb_n_estimators", 150), key="lgb_n")

if st.button("Guardar configuración"):
    new_cfg = {
        "fast_debug": st.session_state["fast_debug"],
        "rf_n_estimators": st.session_state["rf_n"],
        "xgb_n_estimators": st.session_state["xgb_n"],
        "lgb_n_estimators": st.session_state["lgb_n"]
    }
    CFG.write_text(json.dumps(new_cfg, indent=2))
    st.success("Configuración guardada en config.json")
