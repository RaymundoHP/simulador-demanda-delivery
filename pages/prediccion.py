# pages/prediccion.py  — Versión PRO (Pipeline reconstruido)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import warnings
import datetime

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🤖 Predicción de Demanda — Pipeline reconstruido")

# Paths robustos (models/ + fallback /mnt/data)
BASE_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = BASE_PATH / "models"
FALLBACK_PATH = Path("/mnt/data")

st.info(f"Buscando artifacts en `{MODELS_PATH}` y fallback `{FALLBACK_PATH}`")

# Robust loader: intenta models/ primero y luego /mnt/data
def robust_load(name_list):
    if isinstance(name_list, str):
        names = [name_list]
    else:
        names = list(name_list)
    loaded = {}
    for nm in names:
        p1 = MODELS_PATH / nm
        p2 = FALLBACK_PATH / nm
        if p1.exists():
            try:
                loaded[nm] = joblib.load(p1)
                st.write(f"✅ Cargado {nm} desde models/")
                continue
            except Exception as e:
                st.warning(f"⚠ Error cargando {p1}: {e}")
        if p2.exists():
            try:
                loaded[nm] = joblib.load(p2)
                st.write(f"✅ Cargado {nm} desde fallback /mnt/data")
                continue
            except Exception as e:
                st.warning(f"⚠ Error cargando {p2}: {e}")
        loaded[nm] = None
        st.info(f"ℹ️ No encontrado: {nm}")
    return loaded

# artifacts requeridos
names = ["feature_columns.pkl", "final_scaler.pkl", "ipca_blocks.pkl", "kmeans_cluster.pkl"]
art = robust_load(names)

feature_columns = art.get("feature_columns.pkl")
final_scaler = art.get("final_scaler.pkl")
ipca_blocks = art.get("ipca_blocks.pkl")
kmeans_cluster = art.get("kmeans_cluster.pkl")

if feature_columns is None:
    st.error("❌ No se encontró 'feature_columns.pkl'. Necesario para reconstruir las 274 features. Sube el archivo o colócalo en models/ o /mnt/data.")
    st.stop()

# Cargar modelos (todos los .pkl que no sean artifacts)
models = {}
if MODELS_PATH.exists():
    for f in MODELS_PATH.glob("*.pkl"):
        fname = f.name.lower()
        if any(x in fname for x in ["feature_columns", "final_scaler", "ipca_blocks", "kmeans_cluster", "scaler", "ipca", "pca"]):
            continue
        try:
            models[f.stem] = joblib.load(f)
        except Exception as e:
            st.warning(f"⚠ No se pudo cargar {f.name}: {e}")

# fallback /mnt/data
for f in FALLBACK_PATH.glob("*.pkl"):
    fname = f.name.lower()
    if any(x in fname for x in ["feature_columns", "final_scaler", "ipca_blocks", "kmeans_cluster", "scaler", "ipca", "pca"]):
        continue
    if f.stem in models:
        continue
    try:
        models[f.stem] = joblib.load(f)
    except Exception:
        pass

if not models:
    st.warning("⚠ No se cargó ningún modelo predictivo (.pkl). Asegúrate de que existan en models/")
else:
    st.success(f"Modelos detectados: {list(models.keys())}")

# INPUTS
col1, col2, col3 = st.columns(3)
with col1:
    pedidos = st.number_input("Pedidos estimados", min_value=0, max_value=20000, value=200)
with col2:
    temperatura = st.slider("Temperatura (°C)", 0, 45, 25)
with col3:
    hora = st.slider("Hora del día", 0, 23, 12)

input_user = pd.DataFrame([{"orders": pedidos, "temperature": temperatura, "hour": hora}])
st.write("📝 Entrada:", input_user)

# PREPROCESS: reconstruir vector final
def preprocess_input(raw_df, feature_columns, ipca_blocks=None, kmeans_cluster=None, final_scaler=None):
    # 1) vector vacío con todas las columnas de entrenamiento
    X_full = pd.DataFrame(0.0, index=raw_df.index, columns=feature_columns)

    # 2) mapear columnas obvias
    for col in raw_df.columns:
        if col in X_full.columns:
            X_full[col] = raw_df[col].values
        else:
            matches = [c for c in X_full.columns if c.lower() == col.lower()]
            if matches:
                X_full[matches[0]] = raw_df[col].values

    # 3) aplicar IPCA por bloques si existe
    if ipca_blocks:
        for gname, obj in ipca_blocks.items():
            try:
                cols = obj.get("cols", []) if isinstance(obj, dict) else []
                ipca = obj.get("ipca") if isinstance(obj, dict) else None
                scaler_g = obj.get("scaler") if isinstance(obj, dict) else None

                if not cols or ipca is None:
                    continue

                # construir matriz de entrada para esos cols (rellenar con 0 si faltan)
                Xg = np.zeros((len(raw_df), len(cols)), dtype=float)
                for j, c in enumerate(cols):
                    if c in X_full.columns:
                        Xg[:, j] = X_full[c].astype(float).values
                    elif c in raw_df.columns:
                        Xg[:, j] = raw_df[c].astype(float).values
                    else:
                        Xg[:, j] = 0.0

                # escalar grupo si existe
                if scaler_g is not None:
                    try:
                        Xg = scaler_g.transform(Xg)
                    except Exception:
                        Xg = (Xg - np.mean(Xg, axis=0)) / (np.std(Xg, axis=0) + 1e-9)

                # transformar con IPCA
                try:
                    Xg_comp = ipca.transform(Xg)
                except Exception:
                    Xg_comp = ipca.transform(np.asarray(Xg))

                comp_cols = [f"{gname}_PC{i+1}" for i in range(Xg_comp.shape[1])]
                for k, cname in enumerate(comp_cols):
                    if cname in X_full.columns:
                        X_full[cname] = Xg_comp[:, k]
            except Exception as e:
                st.warning(f"⚠ IPCA block '{gname}' error: {e}")
                continue

    # 4) Añadir clusters KMeans (one-hot) si existe el cluster model
    if kmeans_cluster is not None:
        try:
            repr_cols = X_full.columns[: min(12, X_full.shape[1])]
            repr_mat = X_full[repr_cols].astype(float).values
            cl_labels = kmeans_cluster.predict(repr_mat)
            for lab in sorted(set(cl_labels)):
                cname = f"cluster_k_{lab}"
                onehot = (cl_labels == lab).astype(int)
                if cname in X_full.columns:
                    X_full[cname] = onehot
        except Exception as e:
            st.warning(f"⚠ KMeans cluster add failed: {e}")

    # 5) asegurar orden y tipos, rellenar faltantes
    for c in feature_columns:
        if c not in X_full.columns:
            X_full[c] = 0.0

    X_full = X_full[feature_columns].astype(float).fillna(0.0)

    # 6) final scaler (si existe)
    if final_scaler is not None:
        try:
            X_scaled = final_scaler.transform(X_full)
            X_full = pd.DataFrame(X_scaled, columns=feature_columns, index=X_full.index)
        except Exception as e:
            st.warning(f"⚠ final_scaler transform failed: {e}")

    return X_full

# build processed vector
try:
    X_proc = preprocess_input(input_user, feature_columns, ipca_blocks=ipca_blocks, kmeans_cluster=kmeans_cluster, final_scaler=final_scaler)
    st.write(f"✅ Vector final preparado (shape: {X_proc.shape})")
except Exception as e:
    st.error(f"❌ Error preparando vector final: {e}")
    st.stop()

# PREDICT con seguridad
results = {}
for name, model in models.items():
    try:
        X_input = X_proc.values
        expected = None
        if hasattr(model, "n_features_in_"):
            expected = int(model.n_features_in_)
        try:
            pred = model.predict(X_input)
        except Exception as e:
            if expected is not None:
                if X_input.shape[1] < expected:
                    pad = np.zeros((X_input.shape[0], expected - X_input.shape[1]))
                    X_try = np.hstack([X_input, pad])
                else:
                    X_try = X_input[:, :expected]
                pred = model.predict(X_try)
            else:
                raise e
        results[name] = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
    except Exception as e:
        results[name] = None
        st.warning(f"⚠ No se pudo predecir con {name}: {e}")

if not results:
    st.error("❌ No hubo predicciones. Revisa que existan modelos cargados y que feature_columns.pkl corresponda al entrenamiento.")
    st.stop()

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["prediction"])
st.subheader("📊 Predicciones por modelo")
st.dataframe(results_df, width="stretch")

# Ensemble & IC95%
valid_preds = [v for v in results.values() if v is not None]
if valid_preds:
    ensemble_mean = float(np.mean(valid_preds))
    ensemble_std = float(np.std(valid_preds))
    li = ensemble_mean - 1.96 * ensemble_std
    ui = ensemble_mean + 1.96 * ensemble_std
    st.success(f"🔮 Predicción ensemble (mean): {ensemble_mean:.2f} pedidos  —  IC95%: [{li:.2f}, {ui:.2f}]")
else:
    st.warning("⚠ No hay predicciones válidas para construir ensemble")

# Export button
if valid_preds:
    export_df = input_user.copy()
    for k, v in results.items():
        export_df[f"pred_{k}"] = v
    export_df["ensemble_mean"] = ensemble_mean
    export_df["ensemble_li95"] = li
    export_df["ensemble_ui95"] = ui
    export_df["timestamp"] = datetime.datetime.now().isoformat()
    to_csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exportar predicción (CSV)", data=to_csv, file_name="prediction_export.csv", mime="text/csv")

# Optional: SHAP
if st.checkbox("Calcular SHAP (si está disponible)"):
    try:
        import shap
        # pick first model that supports predict
        first_model = None
        first_name = None
        for name, m in models.items():
            if hasattr(m, "predict"):
                first_model = m
                first_name = name
                break
        if first_model is not None:
            explainer = shap.Explainer(first_model, X_proc)
            shap_values = explainer(X_proc)
            st.subheader(f"SHAP local - {first_name}")
            try:
                fig = shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
            except Exception:
                st.write("SHAP calculado pero no se puede graficar (modelo/entorno).")
        else:
            st.write("No hay modelo compatible para SHAP.")
    except Exception as e:
        st.warning(f"SHAP no disponible o falló: {e}")

st.info("✅ Pipeline de predicción finalizado (revise mensajes y warnings).")
