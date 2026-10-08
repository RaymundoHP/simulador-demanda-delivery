import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import random
from inference_utils import PredictionPipeline

# --- INICIALIZACIÓN ---
pipeline = PredictionPipeline()

# Fuente profesional
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA, dbc.icons.BOOTSTRAP, FONT_URL], suppress_callback_exceptions=True)
app.title = "Prediccion Delivery"

# --- ESTILOS (CSS) ---
SIDEBAR_STYLE = {
    "background": "linear-gradient(180deg, #111827 0%, #0f172a 100%)",
    "color": "#E2E8F0",
    "height": "100vh",
    "padding": "2rem 1.5rem",
    "overflow-y": "auto",
    "display": "flex",
    "flex-direction": "column",
    "font-family": "'Inter', sans-serif"
}

CONTENT_STYLE = {
    "padding": "1.5rem",
    "background-color": "#F8FAFC",
    "height": "100vh",
    "overflow-y": "hidden",
    "font-family": "'Inter', sans-serif"
}

CARD_STYLE = {
    "box-shadow": "0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03)",
    "border": "1px solid #E2E8F0",
    "border-radius": "12px",
    "height": "100%",
    "overflow": "hidden",
    "background-color": "white"
}

# --- MENU LATERAL ---
sidebar_content = html.Div([
    html.Div([
        html.Div([
            html.I(className="bi bi-hexagon-fill fs-4 me-2", style={"color": "#3B82F6"}),
            html.Span("Prediccion Delivery", style={"fontSize": "1.25rem", "fontWeight": "600", "letterSpacing": "-0.5px", "color": "white"})
        ], className="d-flex align-items-center mb-5"),
        
        dbc.Nav([
            dbc.NavLink(
                [html.I(className="bi bi-cpu-fill me-3"), "Panel de simulación"], 
                href="/simulator", 
                active="exact", 
                className="text-white d-flex align-items-center",
                style={"borderRadius": "8px", "padding": "12px 15px", "backgroundColor": "rgba(255,255,255,0.1)", "fontWeight": "500"}
            ),
        ], vertical=True, pills=True),
    ]),
    
    html.Div([
        html.Div([
            html.Small("STATUS DEL SISTEMA", className="text-muted fw-bold", style={"fontSize": "0.6rem", "letterSpacing": "1px"}),
            html.Div([
                html.I(className="bi bi-circle-fill text-success me-2", style={"fontSize": "8px"}),
                html.Span("Motor Neural Activo", style={"fontSize": "0.85rem"})
            ], className="mt-1 d-flex align-items-center")
        ], className="mb-4 p-3", style={"backgroundColor": "rgba(255,255,255,0.05)", "borderRadius": "8px"}),

        dbc.Button([html.I(className="bi bi-arrows-angle-expand me-2"), "Expandir"], 
                   id="btn-toggle-sidebar", 
                   color="dark", 
                   className="w-100 btn-sm py-2",
                   style={"border": "1px solid rgba(255,255,255,0.2)", "borderRadius": "8px"})
    ], className="mt-auto")
], style={"height": "100%", "display": "flex", "flexDirection": "column"})

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = dbc.Container([
    dcc.Location(id="url"),
    dcc.Store(id='sidebar-state', data=True),
    
    dbc.Row([
        dbc.Col(html.Div(sidebar_content, style=SIDEBAR_STYLE), id="sidebar-col", width=12, lg=2, className="p-0", style={"transition": "all 0.4s ease-in-out"}),
        dbc.Col(content, id="content-col", width=12, lg=10, className="p-0", style={"transition": "all 0.4s ease-in-out"})
    ], className="g-0")
], fluid=True, className="p-0")

# --- CALLBACKS UI ---
@app.callback(
    [Output("sidebar-col", "lg"), Output("sidebar-col", "style"), Output("content-col", "lg"), Output("btn-toggle-sidebar", "children"), Output("sidebar-state", "data")],
    Input("btn-toggle-sidebar", "n_clicks"), State("sidebar-state", "data"), prevent_initial_call=True
)
def toggle_sidebar(n, is_open):
    if not n: return no_update
    if is_open: return 0, {"display": "none"}, 12, [html.I(className="bi bi-box-arrow-right me-2"), "Menú"], False
    else: return 2, SIDEBAR_STYLE, 10, [html.I(className="bi bi-arrows-angle-expand me-2"), "Expandir"], True

@app.callback(Output("btn-restore-sidebar", "className"), Input("sidebar-state", "data"))
def show_restore_button(is_open): return "d-inline-block ms-3 shadow-sm" if not is_open else "d-none"

@app.callback(
    [Output("sidebar-col", "lg", allow_duplicate=True), Output("sidebar-col", "style", allow_duplicate=True), Output("content-col", "lg", allow_duplicate=True), Output("btn-toggle-sidebar", "children", allow_duplicate=True), Output("sidebar-state", "data", allow_duplicate=True)],
    Input("btn-restore-sidebar", "n_clicks"), prevent_initial_call=True
)
def restore_sidebar(n): return 2, SIDEBAR_STYLE, 10, [html.I(className="bi bi-arrows-angle-expand me-2"), "Expandir"], True

# --- RENDERIZADO ---
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/" or pathname == "/dashboard": pass

    return dbc.Container([
        dbc.Row([dbc.Col([html.H4("Panel operativo", className="text-dark fw-bold mb-0 d-inline-block me-3", style={"letterSpacing": "-0.5px"}), dbc.Button([html.I(className="bi bi-layout-sidebar-inset me-2"), "Menú"], id="btn-restore-sidebar", color="dark", size="sm", className="d-none", n_clicks=0)], width=12)], className="mb-3", style={"height": "4vh"}),
        
        dbc.Row([
            # COLUMNA IZQUIERDA
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Parámetros de Entrada", className="bg-white fw-bold border-0 pt-3 pb-0 small text-uppercase text-muted"),
                    dbc.CardBody([
                        dbc.Form([
                            html.Label("Segmento", className="small fw-bold text-dark"),
                            dcc.Dropdown(id="in-cuisine", options=[{'label': '🍔 Fast Food', 'value': 'fast_food'}, {'label': '🥗 Saludable', 'value': 'healthy'}, {'label': '☕ Bebidas', 'value': 'beverages'}, {'label': '🍲 Del día', 'value': 'generic'}], value='generic', clearable=False, className="mb-2 small"),
                            dbc.Row([dbc.Col([html.Label("Pedidos", className="small text-muted"), dbc.Input(id="in-orders", type="number", value=300, step=10, size="sm")]), dbc.Col([html.Label("Tendencia", className="small text-muted"), dcc.Dropdown(id="in-trend", options=[{'label': '➡️ Estable', 'value': 'stable'}, {'label': '📈 Sube', 'value': 'growing'}, {'label': '📉 Baja', 'value': 'declining'}], value='stable', clearable=False, className="small")])], className="mb-2 g-1"),
                            dbc.Row([dbc.Col([html.Label("Clima", className="small text-muted"), dcc.Dropdown(id="in-weather", options=[{'label': '☀️ Soleado', 'value': 'sunny'}, {'label': '🌧️ Lluvia', 'value': 'rain'}, {'label': '⛈️ Tormenta', 'value': 'storm'}], value='sunny', clearable=False, className="small")]), dbc.Col([html.Label("Evento", className="small text-muted"), dcc.Dropdown(id="in-event", options=[{'label': '🟢 Normal', 'value': 'none'}, {'label': '⚽ Partido', 'value': 'match'}, {'label': '🚧 Tráfico', 'value': 'traffic'}], value='none', clearable=False, className="small")])], className="mb-2 g-1"),
                            html.Label("Flota/Repartidores & Radio(km)", className="small fw-bold text-dark"),
                            dbc.Row([dbc.Col(dbc.Input(id="in-fleet", type="number", value=20, step=1, placeholder="Flota", size="sm")), dbc.Col(dbc.Input(id="in-radius", type="number", value=2.5, step=0.5, placeholder="Km", size="sm"))], className="mb-2 g-1"),
                            html.Label("Temperatura", className="small text-muted"),
                            dcc.Slider(id="in-temp", min=0, max=40, step=1, value=24, marks={0:'0°', 40:'40°'}, className="mb-2"),
                            dbc.Row([dbc.Col([html.Label("Precio", className="small text-muted"), dbc.Input(id="in-base-price", type="number", value=150, size="sm")]), dbc.Col([html.Label("Dscto %", className="small text-muted"), dbc.Input(id="in-discount", type="number", value=0, size="sm")])], className="mb-2 g-1"),
                            dbc.Checklist(id="in-promo-channels", options=[{"label": "📧 Email", "value": "email"}, {"label": "📱 App", "value": "homepage"}], value=[], switch=True, inline=True, className="mb-2 small"),
                            dbc.Button("🚀 Simular Escenario", id="btn-predict", color="primary", className="w-100 fw-bold shadow-sm btn-sm mt-2")
                        ])
                    ], className="p-3", style={"overflowY": "auto"})
                ], style={**CARD_STYLE, "height": "55vh"}, className="mb-3"),
                dbc.Card(dbc.CardBody([html.H6("Carga Horaria", className="card-title text-muted small mb-0 text-uppercase fw-bold"), dcc.Graph(id="hourly-chart", config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'})], className="p-2 h-100"), style={**CARD_STYLE, "height": "30vh"})
            ], width=12, lg=3, className="d-flex flex-column"),

            # COLUMNA DERECHA
            dbc.Col([
                # Fila 1: Gauge + Alertas
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div(dcc.Graph(id="gauge-chart", config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'}), style={"height":"100%"}, id="gauge-wrapper"), dbc.Tooltip("Capacidad vs Demanda", target="gauge-wrapper")], className="h-100 p-0 overflow-hidden"), style=CARD_STYLE), width=5),
                    dbc.Col(html.Div(id="alerts-panel", className="h-100"), width=7)
                ], className="g-3 mb-4", style={"height": "21vh"}),
                # Fila 2: KPIs
                dbc.Row(id="kpi-row", className="g-3 mb-4", style={"height": "13vh"}),
                # Fila 3: Análisis
                dbc.Row(id="deep-analysis-row", className="g-3 mb-4", style={"height": "21vh"}),
                # Fila 4: Datos No Estructurados
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.H6([html.I(className="bi bi-chat-quote-fill me-2 text-primary"), "Tópicos Clave (WordCloud)"], className="card-title text-dark fw-bold small mb-0"), dcc.Graph(id="wordcloud-chart", config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'})], className="h-100 p-3 overflow-hidden"), style=CARD_STYLE), width=7),
                    dbc.Col(dbc.Card(dbc.CardBody([html.H6([html.I(className="bi bi-emoji-smile-fill me-2 text-primary"), "Grafica de sentimiento"], className="card-title text-dark fw-bold small mb-0"), dcc.Graph(id="sentiment-chart", config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'})], className="h-100 p-3 overflow-hidden"), style=CARD_STYLE), width=5),
                ], className="g-3", style={"height": "28vh"})
            ], width=12, lg=9, className="d-flex flex-column")
        ])
    ], fluid=True, className="vh-100 overflow-hidden")

@app.callback(
    [Output("alerts-panel", "children"), Output("kpi-row", "children"), Output("deep-analysis-row", "children"),
     Output("gauge-chart", "figure"), Output("hourly-chart", "figure"), Output("wordcloud-chart", "figure"), Output("sentiment-chart", "figure")],
    Input("btn-predict", "n_clicks"),
    [State("in-cuisine", "value"), State("in-fleet", "value"), State("in-orders", "value"), State("in-trend", "value"), State("in-weather", "value"), State("in-event", "value"), State("in-radius", "value"), State("in-temp", "value"), State("in-base-price", "value"), State("in-discount", "value"), State("in-promo-channels", "value")]
)
def update_dashboard(n_clicks, cuisine, fleet, orders, trend, weather, event, radius, temp, base_price, discount, channels):
    colors = {"success": "#10B981", "warning": "#F59E0B", "danger": "#EF4444", "info": "#3B82F6", "dark": "#111827", "secondary": "#9CA3AF"}
    
    def kpi(title, val, icon, status, sub=None):
        style = {**CARD_STYLE, "border-left": f"4px solid {colors[status]}"}
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([html.Small(title, className="fw-bold text-secondary text-uppercase", style={"fontSize":"0.65rem", "letterSpacing":"0.5px"}), html.H3(val, className="mb-0 fw-bold text-dark", style={"fontSize":"1.7rem"}), html.Small(sub, className=f"text-{status} fw-bold", style={"fontSize":"0.65rem"}) if sub else None]),
            html.I(className=f"bi {icon} fs-1 text-{status} opacity-25")
        ], className="d-flex justify-content-between align-items-center h-100 p-3"), style=style), width=3)

    # --- ESTADO INICIAL ---
    if not n_clicks:
        empty_alerts = html.Div([dbc.Card([dbc.CardHeader("IA Insights", className="bg-white border-0 fw-bold py-1 small text-secondary"), dbc.CardBody(html.Small("Inicia una simulación para ver predicciones.", className="text-muted"), className="py-1")], className="flex-grow-1 h-100 border-secondary", style={"borderStyle": "dashed"})], className="d-flex flex-column h-100")
        empty_kpis = [kpi("Ingresos", "$--", "bi-currency-dollar", "secondary"), kpi("Utilidad", "$--", "bi-wallet2", "secondary"), kpi("Saturación", "--%", "bi-activity", "secondary"), kpi("Clima", "--", "bi-thermometer-half", "secondary")]
        empty_analysis = [dbc.Col(dbc.Card(dbc.CardBody([html.H6("Eficiencia", className="card-title mb-3 text-muted small"), dbc.Progress(value=0, className="mb-2"), dbc.Progress(value=0)], className="h-100 p-3"), style=CARD_STYLE), width=6), dbc.Col(dbc.Card(dbc.CardBody([html.H6("Experiencia", className="card-title mb-3 text-muted small"), dbc.Progress(value=0, className="mb-2"), dbc.Progress(value=0)], className="h-100 p-3"), style=CARD_STYLE), width=6)]
        
        empty_fig = go.Figure()
        empty_fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
        
        # Gauge vacío (mismos márgenes que el lleno)
        gauge_empty = go.Figure(go.Indicator(mode="gauge+number", value=0, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#E5E7EB"}}))
        gauge_empty.update_layout(margin=dict(t=30, b=30, l=45, r=45), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#9CA3AF", 'size': 12})

        return empty_alerts, empty_kpis, empty_analysis, gauge_empty, empty_fig, empty_fig, empty_fig

    # --- CÁLCULOS ---
    if None in [fleet, orders, radius, base_price, temp]: return dbc.Alert("Error", color="danger"), [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure()

    promo_email, promo_home = "email" in channels, "homepage" in channels
    inputs = {'cuisine': cuisine, 'orders': orders, 'trend': trend, 'weather': weather, 'event': event, 'radius': radius, 'temp': temp, 'base_price': base_price, 'discount': discount, 'promo_email': promo_email, 'promo_homepage': promo_home}
    demanda = pipeline.predict(inputs)
    capacidad = fleet * 15
    saturacion = (demanda / capacidad) * 100 if capacidad > 0 else 100
    precio_final = base_price * (1 - (discount or 0)/100)
    
    retraso = 10 if saturacion > 85 else (25 if saturacion > 100 else 0)
    eta = 18 + (radius * 6) + retraso
    prob_canc = 5 if eta > 40 else (15 if eta > 50 else (30 if eta > 60 else 0))
    
    perdidos_eta = demanda * (prob_canc/100)
    total_perdidos = max(0, demanda - capacidad) + perdidos_eta
    real_orders = min(demanda, capacidad) - perdidos_eta
    ingresos = real_orders * precio_final
    dinero_perdido = total_perdidos * precio_final
    costo_op = fleet * 60 * 5
    utilidad = ingresos - costo_op
    prod = (real_orders / (fleet * 5)) if fleet > 0 else 0
    cpo = (costo_op / real_orders) if real_orders > 0 else 0

    st_sat = "success" if saturacion < 85 else "warning" if saturacion < 100 else "danger"
    st_util = "success" if utilidad > 0 else "danger"
    st_prod = "success" if 1.5 <= prod <= 3.5 else "danger" if prod < 1.5 else "warning"
    st_cpo = "success" if cpo < 30 else "warning" if cpo < 50 else "danger"
    st_eta = "success" if eta < 40 else "warning" if eta < 55 else "danger"

    # --- VISUALES ---
    alert_msg = dbc.Alert([html.I(className="bi bi-exclamation-circle-fill me-2"), f"PÉRDIDA CRÍTICA: ${dinero_perdido:,.0f}"], color="danger", className="mb-2 py-1 small flex-shrink-0 fw-bold") if dinero_perdido > 0 else None
    drivers_needed = int(np.ceil(total_perdidos / 15))
    rec_list = html.Ul([html.Li(f"Faltan {drivers_needed} repartidor(es).") if drivers_needed > 0 else html.Li("Capacidad óptima."), html.Li(f"Fuga clientes: {prob_canc}%")], className="mb-0 small ps-3 text-dark")
    ia_card = dbc.Card([dbc.CardHeader([html.I(className="bi bi-cpu-fill me-2 text-primary"), "Insights"], className="bg-white border-0 fw-bold py-1 small"), dbc.CardBody(rec_list, className="py-1")], className="flex-grow-1 border-primary", style={"borderStyle": "solid", "borderWidth": "1px"})
    alerts_panel_content = html.Div([alert_msg, ia_card] if alert_msg else [ia_card], className="d-flex flex-column h-100")

    kpi_row_content = [
        kpi("Ingresos Reales", f"${ingresos:,.0f}", "bi-currency-dollar", "dark"),
        kpi("Utilidad Neta", f"${utilidad:,.0f}", "bi-wallet2", st_util, f"Margen: {(utilidad/ingresos*100) if ingresos>0 else 0:.0f}%"),
        kpi("Saturación", f"{saturacion:.0f}%", "bi-activity", st_sat, "Sobrecarga" if saturacion > 100 else "Estable"),
        kpi("Clima / Temp", f"{temp}°C", "bi-thermometer-half", "info", weather.title())
    ]

    deep_analysis_content = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Eficiencia Operativa", className="card-title mb-3 text-dark fw-bold small text-uppercase"),
            dbc.Row([dbc.Col(html.Small("Velocidad", className="fw-bold text-muted")), dbc.Col([html.Span(f"{prod:.1f} p/h", className="fw-bold text-dark me-2"), dbc.Badge("Bajo" if prod<1.5 else "Alto" if prod>3.5 else "Óptimo", color=st_prod, className="p-1", style={"fontSize":"0.6em"})], className="text-end")]),
            dbc.Progress(value=min(prod*30, 100), color=st_prod, style={"height": "6px"}, className="mb-2"),
            dbc.Row([dbc.Col(html.Small("Costo Operativo", className="fw-bold text-muted")), dbc.Col([html.Span(f"${cpo:.0f} /u", className="fw-bold text-dark me-2"), dbc.Badge("Caro" if cpo>50 else "Bien", color=st_cpo, className="p-1", style={"fontSize":"0.6em"})], className="text-end")]),
            dbc.Progress(value=min(cpo, 100), color=st_cpo, style={"height": "6px"}, className="mb-0"),
        ], className="h-100 p-3"), style=CARD_STYLE), width=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Experiencia del Cliente", className="card-title mb-3 text-dark fw-bold small text-uppercase"),
            dbc.Row([dbc.Col(html.Small("Tiempo Entrega (ETA)", className="fw-bold text-muted")), dbc.Col([html.Span(f"{eta:.0f} min", className="fw-bold text-dark me-2"), dbc.Badge("Lento" if eta>45 else "Rápido", color=st_eta, className="p-1", style={"fontSize":"0.6em"})], className="text-end")]),
            dbc.Progress([dbc.Progress(value=18, color="primary", bar=True, className="opacity-50"), dbc.Progress(value=radius*6, color="info", bar=True, className="opacity-50"), dbc.Progress(value=retraso, color="danger", bar=True, striped=True)], style={"height": "6px"}, className="mb-1"),
            html.Div([html.Span("● Cocina ", className="text-primary me-2", style={"fontSize":"0.6rem"}), html.Span("● Viaje ", className="text-info me-2", style={"fontSize":"0.6rem"}), html.Span("● Retraso", className="text-danger", style={"fontSize":"0.6rem"})], className="d-flex mb-2"),
            dbc.Row([dbc.Col(html.Small("Prob. Cancelación", className="fw-bold text-muted")), dbc.Col(html.Span(f"{prob_canc}% Riesgo", className="float-end fw-bold text-danger" if prob_canc > 0 else "text-success"))]),
            dbc.Progress(value=prob_canc, color="danger", style={"height": "6px"}),
        ], className="h-100 p-3"), style=CARD_STYLE), width=6)
    ]

    # GAUGE CORREGIDO (MÁRGENES + DELTA)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", 
        value=demanda, 
        delta={'reference': capacidad, 'increasing': {'color': "red" if saturacion > 100 else "gray"}}, 
        gauge={
            'axis': {'range': [0, max(demanda*1.2, capacidad*1.2)]}, 
            'bar': {'color': colors[st_sat]}, 
            'threshold': {'line': {'color': "black", 'width': 3}, 'value': capacidad}
        }
    ))
    # Aumento márgenes (t=30, b=30, l=45, r=45) para asegurar que el delta y los números laterales entren
    fig_gauge.update_layout(margin=dict(t=30, b=30, l=45, r=45), paper_bgcolor="rgba(0,0,0,0)", font={'size': 14, 'family': "Inter"})

    fig_hourly = go.Figure()
    horas_x = [12,13,14,15,16,17,18,19,20,21,22]
    demanda_y = [demanda * p for p in [0.05, 0.10, 0.15, 0.08, 0.05, 0.05, 0.08, 0.15, 0.18, 0.08, 0.03]]
    cap_hora = fleet * 2.5
    cols_bar = [colors["danger"] if d > cap_hora else colors["info"] for d in demanda_y]
    fig_hourly.add_trace(go.Bar(x=horas_x, y=demanda_y, marker_color=cols_bar))
    fig_hourly.add_trace(go.Scatter(x=horas_x, y=[cap_hora]*11, mode='lines', line=dict(color='black', dash='dash')))
    fig_hourly.update_layout(margin=dict(l=25, r=10, t=10, b=20), showlegend=False, template="plotly_white", xaxis=dict(tickmode='array', tickvals=horas_x, ticktext=[f"{h}h" for h in horas_x], tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)))

    bad_mood = eta > 45 or saturacion > 100
    word_list = ["Tarde", "Frío", "Caro", "Lento", "Nunca", "Mal"] * 3 if bad_mood else ["Rápido", "Caliente", "Buen", "Amable", "Tiempo", "Rico"] * 3
    selected_words = random.choices(word_list, k=12)
    fig_wc = go.Figure()
    for word in selected_words:
        fig_wc.add_trace(go.Scatter(x=[random.random()], y=[random.random()], mode="text", text=[word], textfont=dict(size=random.randint(14, 24), color=random.choice([colors["danger"] if bad_mood else colors["success"]]), family="Inter")))
    fig_wc.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    neg_pct = 0.8 if bad_mood else 0.1
    pos_pct = 1 - neg_pct
    display_val, display_lbl, icon_sent, txt_col = (int(pos_pct * 100), "Pos", "😊", colors["success"]) if pos_pct >= neg_pct else (int(neg_pct * 100), "Neg", "😠", colors["danger"])
    fig_sent = go.Figure(data=[go.Pie(labels=['Pos', 'Neg'], values=[pos_pct, neg_pct], hole=.75, marker_colors=[colors["success"], colors["danger"]], textinfo='none')])
    fig_sent.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), annotations=[dict(text=f"<span style='font-size:30px'>{icon_sent}</span><br><span style='font-size:14px; font-weight:bold; color:{txt_col}'>{display_val}% {display_lbl}</span>", x=0.5, y=0.5, showarrow=False)], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return alerts_panel_content, kpi_row_content, deep_analysis_content, fig_gauge, fig_hourly, fig_wc, fig_sent

if __name__ == "__main__":
    app.run(debug=True)