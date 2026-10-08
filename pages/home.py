import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Dashboard Ejecutivo",
    page_icon="📊",
    layout="wide"
)

# ---- SIDEBAR ----
st.sidebar.title("📍 Navegación")

# Modo avanzado
modo_avanzado = st.sidebar.toggle("🧪 Modo avanzado", value=False)

st.sidebar.markdown("---")

# ---- CONTENIDO PRINCIPAL DEL HOME ----
st.title("📊 Dashboard Ejecutivo de Demanda y Delivery")
st.subheader("Una plataforma intuitiva para análisis general y técnico")

st.markdown("""
Bienvenido al Dashboard Ejecutivo.  
Este panel integra **análisis de delivery**, **ventas**, **predicción de modelos** y módulos avanzados para usuarios técnicos.

### 🟦 Modo General (para todos)
Incluye:
- Tendencias de delivery  
- Análisis de ventas  
- Exploración simple de datos  
- Reportes automáticos  

### 🟥 Modo Avanzado (solo si lo activas)
Incluye análisis técnico:
- Modelos de Machine Learning  
- Comparativa y métricas  
- Clustering (KMeans)  
- Reducción dimensional (IPCA)  
- Pipeline completo de features  

Activa el modo avanzado desde la barra lateral 👉
""")

st.markdown("---")
st.header("🧭 ¿Qué deseas analizar hoy?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚴 Delivery")
    st.markdown("Análisis de tiempos de entrega, ciudades, vehículos y más.")
    st.button("Ir a Delivery")

with col2:
    st.markdown("### 🍽️ Ventas")
    st.markdown("Explora el comportamiento de compras y preferencias.")
    st.button("Ir a Ventas")

with col3:
    st.markdown("### 📄 Reportes")
    st.markdown("Genera reportes automáticos listos para descargar.")
    st.button("Ir a Reportes")

st.markdown("---")

st.info("""
💡 *Consejo:* Si eres usuario técnico, activa "Modo Avanzado" en la barra lateral para acceder a los módulos de Machine Learning.
""")

