import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Cargar datos a DF
df=pd.read_csv("incident_event_log_clean.csv")

# app.py
# Dash dashboard: Boxplot de resolution_time_log por factor categórico + Heatmap de correlaciones
# Ejecuta:  python app.py   y abre http://127.0.0.1:8070/


Y = "resolution_time_log"

# Sugerencias de variables categóricas candidatas (toma las que existan)
candidate_cat = [
    "incident_state", "active", "made_sla", "opened_by",
    "contact_type", "location", "category", "subcategory",
    "impact", "urgency", "priority","knowledge","close_code","resolved_by"
]
cat_vars = [c for c in candidate_cat if c in df.columns]

# Sugerencias de variables numéricas candidatas (toma las que existan)
candidate_num = ["reassignment_count", "reopen_count", "sys_mod_count"]
num_vars = [n for n in candidate_num if n in df.columns]

# Limpieza mínima: quitar NaN de Y
df = df.copy()
df = df[np.isfinite(df[Y])]

# ========= Funciones de figura =========
def make_boxplot(factor: str):
    """Boxplot de Y por factor categórico con orden por media ascendente."""
    if factor not in df.columns:
        return px.box(title=f"Variable {factor} no encontrada")

    # Ordenar categorías por media de Y (ayuda a comparar)
    order = (
        df.groupby(factor)[Y]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig = px.box(
        df,
        x=factor,
        y=Y,
        category_orders={factor: order},
        points="outliers",  # muestra outliers
        title=f"Distribución de {Y} por {factor}",
    )
    fig.update_layout(
        xaxis_title=factor,
        yaxis_title=Y,
        margin=dict(l=30, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def make_corr_heatmap():
    """Heatmap de correlación entre Y y variables numéricas disponibles."""
    cols = [Y] + num_vars
    data = df[cols].copy()

    # Calcular correlaciones (si hay al menos 2 columnas)
    if data.shape[1] < 2:
        return px.imshow(
            np.array([[1.0]]),
            x=[Y],
            y=[Y],
            title="No hay suficientes variables numéricas para correlacionar",
            text_auto=True
        )

    corr = data.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlaciones (incluye resolution_time_log)"
    )
    fig.update_layout(margin=dict(l=30, r=20, t=60, b=40))
    return fig

# ========= App =========
app = dash.Dash(__name__)
app.title = "Incidentes — Exploración de Tiempo de Resolución"

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, Arial", "padding": "18px"},
    children=[
        html.H2("Exploración preliminar: ¿Qué factores explican el tiempo de resolución?"),
        html.P(
            "Usa el selector para comparar la distribución de resolution_time_log por factor categórico. "
            "Abajo verás el heatmap de correlaciones con variables numéricas."
        ),

        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div("Factor categórico:", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="factor-dropdown",
                    options=[{"label": c, "value": c} for c in cat_vars] or [{"label": "(no hay factores)", "value": ""}],
                    value=(cat_vars[0] if len(cat_vars) else ""),
                    clearable=False,
                    style={"minWidth": "260px"},
                ),
            ],
        ),

        html.Div(
            children=[
                dcc.Graph(id="boxplot", figure=make_boxplot(cat_vars[0]) if cat_vars else {}),
            ],
            style={"marginTop": "8px"},
        ),

        html.Hr(style={"margin": "18px 0"}),

        html.H3("Heatmap de correlaciones (numéricas)"),
        html.P(
            "Revisa la relación entre resolution_time_log y métricas como reasignaciones, modificaciones, etc."
        ),
        dcc.Graph(id="heatmap", figure=make_corr_heatmap()),
    ],
)

# ========= Callbacks =========
@app.callback(
    Output("boxplot", "figure"),
    Input("factor-dropdown", "value"),
)
def update_boxplot(factor_value):
    if not factor_value:
        return px.box(title="Selecciona un factor categórico")
    return make_boxplot(factor_value)

# ========= Main =========
if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1", port=8055)


