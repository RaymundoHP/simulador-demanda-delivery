# ==========================================================
# dashboard_streamlit_live.py
# Proyecto: Dashboard Ejecutivo - Predicción Demanda (Live Update)
# Autor: Raymundo Hernández
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# ==== Configuración de página ====
st.set_page_config(
    page_title="Dashboard Ejecutivo - Live Update",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Título principal ====
st.markdown("<h1 style='text-align:center; color:#2E86AB;'>📊 Dashboard Ejecutivo - Live Update</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>KPIs y gráficos interactivos en tiempo real</p>", unsafe_allow_html=True)
st.markdown("---")

# ==== Paths ====
DATA_FILE = "../processed/food_demand_with_clusters.csv"
MODELS_PATH = "../models/"
MAX_POINTS = 15000  # Limitar puntos para gráficos

# ==== Función para cargar datos con cache ====
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Cargar datos
if not os.path.exists(DATA_FILE):
    st.error(f"❌ No se encontró el archivo: {DATA_FILE}")
    st.stop()
df = load_data(DATA_FILE)
st.success(f"✅ Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

# ==== Barra lateral: filtros dinámicos ====
st.sidebar.header("Filtros ejecutivos")
cluster_options = ["cluster_kmeans", "cluster_dbscan", "cluster_gmm"]
selected_cluster_type = st.sidebar.selectbox("Tipo de cluster:", cluster_options)

cluster_values = df[selected_cluster_type].unique()
selected_cluster_value = st.sidebar.multiselect(f"Valor(es) de {selected_cluster_type}:", cluster_values, default=cluster_values)

week_values = df["week"].unique()
selected_weeks = st.sidebar.multiselect("Semana(s):", week_values, default=week_values)

center_values = df["center_id"].unique()
selected_centers = st.sidebar.multiselect("Centro(s):", center_values, default=center_values)

category_values = df["category"].unique()
selected_categories = st.sidebar.multiselect("Categoría(s):", category_values, default=category_values)

# ==== Filtrar datos ====
@st.cache_data
def filter_data(df, cluster_type, cluster_values, weeks, centers, categories):
    return df[
        df[cluster_type].isin(cluster_values) &
        df["week"].isin(weeks) &
        df["center_id"].isin(centers) &
        df["category"].isin(categories)
    ]

df_filtered = filter_data(df, selected_cluster_type, selected_cluster_value, selected_weeks, selected_centers, selected_categories)
st.markdown(f"### 🔹 Datos filtrados: {df_filtered.shape[0]:,} filas")
st.markdown("---")

# ==== Cargar modelos SIN cache ====
def load_models(path):
    model_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    models_dict = {}
    for mf in model_files:
        try:
            models_dict[mf.replace(".pkl","")] = joblib.load(os.path.join(path, mf))
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar el modelo {mf}: {e}")
    return models_dict

models = load_models(MODELS_PATH)
if not models:
    st.error("❌ No se cargó ningún modelo. Revisa la carpeta de modelos.")
    st.stop()

feature_cols = ["checkout_price", "base_price", "emailer_for_promotion", "homepage_featured",
                "PCA1", "PCA2", "cluster_kmeans", "cluster_gmm", "cluster_dbscan"]

# ==== Predicciones y métricas en tiempo real ====
def compute_metrics(df, models, features):
    metrics_list = []
    df_pred = df.copy()
    for model_name, model in models.items():
        try:
            df_pred[model_name+"_pred"] = model.predict(df_pred[features])
            mae = mean_absolute_error(df_pred["num_orders"], df_pred[model_name+"_pred"])
            rmse = np.sqrt(mean_squared_error(df_pred["num_orders"], df_pred[model_name+"_pred"]))
            r2 = r2_score(df_pred["num_orders"], df_pred[model_name+"_pred"])
            metrics_list.append({
                "Modelo": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2
            })
        except Exception as e:
            st.warning(f"⚠️ Error prediciendo con {model_name}: {e}")
            continue

    if not metrics_list:
        st.error("❌ Ningún modelo generó métricas. Revisa tus modelos y features.")
        st.stop()

    metrics_df = pd.DataFrame(metrics_list).sort_values("MAE")
    return df_pred, metrics_df

df_filtered, metrics_df = compute_metrics(df_filtered, models, feature_cols)

# ==== KPIs en vivo ====
st.markdown("## 🎯 KPIs en tiempo real")
for i, row in metrics_df.iterrows():
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown(f"""
        <div style='background-color:#EAF2F8; padding:20px; border-radius:15px; text-align:center; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color:#1B4F72'>{row['Modelo']}</h3>
            <p style='font-size:22px; margin:5px; color:#117864'>R²: {row['R²']:.3f}</p>
            <p style='font-size:20px; margin:5px; color:#B03A2E'>MAE: {row['MAE']:.1f}</p>
            <p style='font-size:20px; margin:5px; color:#2874A6'>RMSE: {row['RMSE']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        df_filtered["error"] = df_filtered["num_orders"] - df_filtered[row['Modelo']+"_pred"]
        fig = px.histogram(df_filtered.sample(MAX_POINTS) if df_filtered.shape[0]>MAX_POINTS else df_filtered,
                           x="error", nbins=30, title="Distribución de errores")
        fig.update_layout(height=200, margin=dict(t=30,l=10,r=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==== Gráficos Predicción vs Real ====
st.markdown("## 📈 Comparativa Predicción vs Real")
for model_name in models.keys():
    st.markdown(f"### {model_name}")
    df_plot = df_filtered.sample(MAX_POINTS) if df_filtered.shape[0] > MAX_POINTS else df_filtered
    fig = px.scatter(
        df_plot,
        x="num_orders",
        y=model_name+"_pred",
        color=selected_cluster_type,
        hover_data=["meal_id","center_id","week"],
        title=f"{model_name}: Predicción vs Real",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.add_shape(
        type="line",
        x0=df_plot["num_orders"].min(),
        y0=df_plot["num_orders"].min(),
        x1=df_plot["num_orders"].max(),
        y1=df_plot["num_orders"].max(),
        line=dict(color="black", dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# ==== Resumen clusters interactivo ====
st.markdown("## 📊 Resumen por Clusters")
cluster_summary = df_filtered.groupby(selected_cluster_type).agg(
    total_orders=("num_orders","sum"),
    avg_orders=("num_orders","mean"),
    meals_count=("meal_id","nunique")
).reset_index()
st.dataframe(cluster_summary)

fig_cluster = px.bar(cluster_summary, x=selected_cluster_type, y="total_orders", color="avg_orders",
                     hover_data=["meals_count"], text="total_orders",
                     title=f"Total y promedio de pedidos por {selected_cluster_type}",
                     color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig_cluster, use_container_width=True)

st.success("✅ Dashboard Live Update listo para usuarios no expertos")
# ==========================================================
# dashboard_streamlit_live.py
# Proyecto: Dashboard Ejecutivo - Predicción Demanda (Live Update)
# Autor: Raymundo Hernández
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# ==== Configuración de página ====
st.set_page_config(
    page_title="Dashboard Ejecutivo - Live Update",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Título principal ====
st.markdown("<h1 style='text-align:center; color:#2E86AB;'>📊 Dashboard Ejecutivo - Live Update</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>KPIs y gráficos interactivos en tiempo real</p>", unsafe_allow_html=True)
st.markdown("---")

# ==== Paths ====
DATA_FILE = "../processed/food_demand_with_clusters.csv"
MODELS_PATH = "../models/"
MAX_POINTS = 15000  # Limitar puntos para gráficos

# ==== Función para cargar datos con cache ====
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Cargar datos
if not os.path.exists(DATA_FILE):
    st.error(f"❌ No se encontró el archivo: {DATA_FILE}")
    st.stop()
df = load_data(DATA_FILE)
st.success(f"✅ Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

# ==== Barra lateral: filtros dinámicos ====
st.sidebar.header("Filtros ejecutivos")
cluster_options = ["cluster_kmeans", "cluster_dbscan", "cluster_gmm"]
selected_cluster_type = st.sidebar.selectbox("Tipo de cluster:", cluster_options)

cluster_values = df[selected_cluster_type].unique()
selected_cluster_value = st.sidebar.multiselect(f"Valor(es) de {selected_cluster_type}:", cluster_values, default=cluster_values)

week_values = df["week"].unique()
selected_weeks = st.sidebar.multiselect("Semana(s):", week_values, default=week_values)

center_values = df["center_id"].unique()
selected_centers = st.sidebar.multiselect("Centro(s):", center_values, default=center_values)

category_values = df["category"].unique()
selected_categories = st.sidebar.multiselect("Categoría(s):", category_values, default=category_values)

# ==== Filtrar datos ====
@st.cache_data
def filter_data(df, cluster_type, cluster_values, weeks, centers, categories):
    return df[
        df[cluster_type].isin(cluster_values) &
        df["week"].isin(weeks) &
        df["center_id"].isin(centers) &
        df["category"].isin(categories)
    ]

df_filtered = filter_data(df, selected_cluster_type, selected_cluster_value, selected_weeks, selected_centers, selected_categories)
st.markdown(f"### 🔹 Datos filtrados: {df_filtered.shape[0]:,} filas")
st.markdown("---")

# ==== Cargar modelos SIN cache ====
def load_models(path):
    model_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    models_dict = {}
    for mf in model_files:
        try:
            models_dict[mf.replace(".pkl","")] = joblib.load(os.path.join(path, mf))
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar el modelo {mf}: {e}")
    return models_dict

models = load_models(MODELS_PATH)
if not models:
    st.error("❌ No se cargó ningún modelo. Revisa la carpeta de modelos.")
    st.stop()

feature_cols = ["checkout_price", "base_price", "emailer_for_promotion", "homepage_featured",
                "PCA1", "PCA2", "cluster_kmeans", "cluster_gmm", "cluster_dbscan"]

# ==== Predicciones y métricas en tiempo real ====
def compute_metrics(df, models, features):
    metrics_list = []
    df_pred = df.copy()
    for model_name, model in models.items():
        try:
            df_pred[model_name+"_pred"] = model.predict(df_pred[features])
            mae = mean_absolute_error(df_pred["num_orders"], df_pred[model_name+"_pred"])
            rmse = np.sqrt(mean_squared_error(df_pred["num_orders"], df_pred[model_name+"_pred"]))
            r2 = r2_score(df_pred["num_orders"], df_pred[model_name+"_pred"])
            metrics_list.append({
                "Modelo": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2
            })
        except Exception as e:
            st.warning(f"⚠️ Error prediciendo con {model_name}: {e}")
            continue

    if not metrics_list:
        st.error("❌ Ningún modelo generó métricas. Revisa tus modelos y features.")
        st.stop()

    metrics_df = pd.DataFrame(metrics_list).sort_values("MAE")
    return df_pred, metrics_df

df_filtered, metrics_df = compute_metrics(df_filtered, models, feature_cols)

# ==== KPIs en vivo ====
st.markdown("## 🎯 KPIs en tiempo real")
for i, row in metrics_df.iterrows():
    col1, col2 = st.columns([2,3])
    with col1:
        st.markdown(f"""
        <div style='background-color:#EAF2F8; padding:20px; border-radius:15px; text-align:center; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color:#1B4F72'>{row['Modelo']}</h3>
            <p style='font-size:22px; margin:5px; color:#117864'>R²: {row['R²']:.3f}</p>
            <p style='font-size:20px; margin:5px; color:#B03A2E'>MAE: {row['MAE']:.1f}</p>
            <p style='font-size:20px; margin:5px; color:#2874A6'>RMSE: {row['RMSE']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        df_filtered["error"] = df_filtered["num_orders"] - df_filtered[row['Modelo']+"_pred"]
        fig = px.histogram(df_filtered.sample(MAX_POINTS) if df_filtered.shape[0]>MAX_POINTS else df_filtered,
                           x="error", nbins=30, title="Distribución de errores")
        fig.update_layout(height=200, margin=dict(t=30,l=10,r=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==== Gráficos Predicción vs Real ====
st.markdown("## 📈 Comparativa Predicción vs Real")
for model_name in models.keys():
    st.markdown(f"### {model_name}")
    df_plot = df_filtered.sample(MAX_POINTS) if df_filtered.shape[0] > MAX_POINTS else df_filtered
    fig = px.scatter(
        df_plot,
        x="num_orders",
        y=model_name+"_pred",
        color=selected_cluster_type,
        hover_data=["meal_id","center_id","week"],
        title=f"{model_name}: Predicción vs Real",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.add_shape(
        type="line",
        x0=df_plot["num_orders"].min(),
        y0=df_plot["num_orders"].min(),
        x1=df_plot["num_orders"].max(),
        y1=df_plot["num_orders"].max(),
        line=dict(color="black", dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# ==== Resumen clusters interactivo ====
st.markdown("## 📊 Resumen por Clusters")
cluster_summary = df_filtered.groupby(selected_cluster_type).agg(
    total_orders=("num_orders","sum"),
    avg_orders=("num_orders","mean"),
    meals_count=("meal_id","nunique")
).reset_index()
st.dataframe(cluster_summary)

fig_cluster = px.bar(cluster_summary, x=selected_cluster_type, y="total_orders", color="avg_orders",
                     hover_data=["meals_count"], text="total_orders",
                     title=f"Total y promedio de pedidos por {selected_cluster_type}",
                     color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig_cluster, use_container_width=True)

st.success("✅ Dashboard Live Update listo para usuarios no expertos")
