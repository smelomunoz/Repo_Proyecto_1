import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Cargar datos a DF
df=pd.read_csv("incident_event_log_clean.csv")

# app.py
# Dash dashboard: Boxplot de resolution_time_log por factor categórico + Heatmap de correlaciones
# Ejecuta:  python app.py   y abre http://127.0.0.1:8055/


Y = "resolution_time_log"

candidate_cat = [
    "incident_state", "active", "made_sla", "opened_by",
    "contact_type", "location", "category", "subcategory",
    "impact", "urgency", "priority","knowledge","close_code","resolved_by"
]
cat_vars = [c for c in candidate_cat if c in df.columns]

candidate_num = ["reassignment_count", "reopen_count", "sys_mod_count"]
num_vars = [n for n in candidate_num if n in df.columns]

df = df.copy()
df = df[np.isfinite(df[Y])]

if "opened_at" in df.columns:
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")
else:
   
    raise ValueError("La columna 'opened_at' no existe en el CSV.")

if "number" not in df.columns:
    raise ValueError("La columna 'number' no existe en el CSV.")


df_time = (
    df.dropna(subset=["opened_at", "number"])
      .drop_duplicates(subset=["number", "opened_at"])
      .copy()
)
if not df_time.empty:
    df_time["date"] = df_time["opened_at"].dt.date
    df_time["day_of_month"] = df_time["opened_at"].dt.day
    DATE_MIN = df_time["opened_at"].min().date()
    DATE_MAX = df_time["opened_at"].max().date()
else:
    DATE_MIN = pd.Timestamp("2016-01-01").date()
    DATE_MAX = pd.Timestamp("2016-12-31").date()

# ========= Funciones de figura =========
def make_boxplot(factor: str):
    """Boxplot de Y por factor categórico con orden por media ascendente."""
    if factor not in df.columns:
        return px.box(title=f"Variable {factor} no encontrada")

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
        points="outliers",  
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

def compute_anova(cat_list):
    """Calcula ANOVA (F y p) para cada variable categórica seleccionada."""
    results = []
    for var in (cat_list or []):
        if var not in df.columns:
            continue
        try:
            # Modelo lineal: Y ~ C(var)
            model = smf.ols(f"{Y} ~ C({var})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # F y p están en la primera fila (C(var))
            row = anova_table.iloc[0]
            results.append({
                "Variable": var,
                "F_value": float(row["F"]),
                "p_value": float(row["PR(>F)"])
            })
        except Exception:
            pass
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("F_value", ascending=False)
    return res_df

def make_anova_figs(anova_df: pd.DataFrame):
    if anova_df.empty:
        return (
            px.bar(title="ANOVA: sin variables o sin resultados"),
            px.bar(title="ANOVA: sin variables o sin resultados")
        )
    fig_f = px.bar(
        anova_df, x="F_value", y="Variable", orientation="h",
        title="ANOVA — F-value (mayor = más explicativa)"
    )

    pvals = anova_df["p_value"].replace(0, np.nextafter(0, 1))
    anova_df_disp = anova_df.copy()
    anova_df_disp["neg_log10_p"] = -np.log10(pvals)

    fig_p = px.bar(
        anova_df_disp, x="neg_log10_p", y="Variable", orientation="h",
        title="ANOVA — −log10(p-value) (más alto = más significativo)"
    )
    for fig in (fig_f, fig_p):
        fig.update_layout(margin=dict(l=30, r=20, t=60, b=40))
    return fig_f, fig_p

def make_recurrence_figures(
    df_src: pd.DataFrame,
    start_date=None,
    end_date=None,
    agg: str = "D",          # "D" diaria, "W" semanal, "M" mensual
    number_col: str = "number",
    opened_col: str = "opened_at"
):
   
    if number_col not in df_src.columns or opened_col not in df_src.columns:
        msg = f"Faltan columnas requeridas: '{number_col}' y/o '{opened_col}'"
        return px.line(title=msg), px.histogram(title=msg)

    tmp = df_src.copy()
    tmp[opened_col] = pd.to_datetime(tmp[opened_col], errors="coerce")
    tmp = tmp.dropna(subset=[opened_col, number_col]).drop_duplicates(subset=[number_col, opened_col])

    if tmp.empty:
        return px.line(title="Sin datos válidos tras limpieza"), px.histogram(title="Sin datos válidos tras limpieza")

    if start_date is None:
        start_date = tmp[opened_col].min().date()
    else:
        start_date = pd.to_datetime(start_date).date()

    if end_date is None:
        end_date = tmp[opened_col].max().date()
    else:
        end_date = pd.to_datetime(end_date).date()

    mask = (tmp[opened_col].dt.date >= start_date) & (tmp[opened_col].dt.date <= end_date)
    sub = tmp.loc[mask].copy()

    if sub.empty:
        return (px.line(title="Incidentes por periodo (sin datos en el rango)"),
                px.histogram(title="Día del mes (sin datos en el rango)"))

    sub = sub.set_index(opened_col).sort_index()
    counts = sub.resample(agg)[number_col].nunique()
    ts_df = counts.reset_index()
    ts_df.columns = [opened_col, "count"]

    titulo_periodo = {"D": "Día", "W": "Semana", "M": "Mes"}.get(agg, agg)
    ts_fig = px.line(
        ts_df, x=opened_col, y="count",
        title=f"Incidentes únicos por periodo ({titulo_periodo})"
    )
    ts_fig.update_layout(
        xaxis_title="Fecha", yaxis_title="Incidentes únicos",
        margin=dict(l=30, r=20, t=60, b=40)
    )

    # Histograma por día del mes
    sub["day_of_month"] = sub.index.day
    dom_fig = px.histogram(
        sub.reset_index(),
        x="day_of_month", nbins=31,
        title="Distribución de incidentes por día del mes (únicos)"
    )
    dom_fig.update_layout(
        xaxis_title="Día del mes", yaxis_title="Cantidad de incidentes únicos",
        margin=dict(l=30, r=20, t=60, b=40)
    )
    dom_fig.update_xaxes(dtick=1)

    return ts_fig, dom_fig

def make_channel_constancy_figures(
    df_src: pd.DataFrame,
    channel_col: str = "contact_type",
    date_col: str = "opened_at",
    period: str = "W",           # "D" diaria, "W" semanal, "M" mensual
    channels_filter: list | None = None
):
    
    if channel_col not in df_src.columns or date_col not in df_src.columns:
        msg = f"Faltan columnas requeridas: '{channel_col}' y/o '{date_col}'"
        return px.line(title=msg), px.bar(title=msg)

    tmp = df_src.dropna(subset=[channel_col, date_col]).copy()
    if channels_filter:
        tmp = tmp[tmp[channel_col].isin(channels_filter)]
    if tmp.empty:
        return px.line(title="Sin datos para los canales seleccionados"), px.bar(title="Sin datos para los canales seleccionados")

    # Resample por periodo y contar incidentes únicos por canal
    tmp = tmp.set_index(date_col).sort_index()
    grp = (
        tmp.groupby(channel_col)
           .resample(period)["number"].nunique()
           .rename("count")
           .reset_index()
    )

    # --- Serie temporal por canal ---
    titulo_periodo = {"D": "diario", "W": "semanal", "M": "mensual"}.get(period, period)
    fig_ts = px.line(
        grp, x=date_col, y="count", color=channel_col,
        title=f"Incidentes únicos por {titulo_periodo} y canal ({channel_col})"
    )
    fig_ts.update_layout(
        xaxis_title="Fecha", yaxis_title="Incidentes únicos",
        margin=dict(l=30, r=20, t=60, b=40)
    )

    # --- CV por canal (std/mean en el vector temporal de cada canal) ---
    cv_df = (
        grp.groupby(channel_col)["count"]
           .agg(mean="mean", std="std")
           .reset_index()
    )
    # Evitar división por 0
    cv_df["CV"] = cv_df.apply(lambda r: (r["std"] / r["mean"]) if r["mean"] not in (0, np.nan) else np.nan, axis=1)
    # Ordenar por menor CV (más constante arriba)
    cv_df = cv_df.sort_values("CV", ascending=True)

    fig_cv = px.bar(
        cv_df, x="CV", y=channel_col, orientation="h",
        hover_data=["mean", "std"],
        title=f"Constancia por canal — Coeficiente de Variación (menor = más constante)"
    )
    fig_cv.update_layout(
        xaxis_title="CV (std/mean)",
        yaxis_title="Canal",
        margin=dict(l=30, r=20, t=60, b=40)
    )

    return fig_ts, fig_cv

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

        # ----- Boxplot selector -----
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
            children=[dcc.Graph(id="boxplot", figure=make_boxplot(cat_vars[0]) if cat_vars else {})],
            style={"marginTop": "8px"},
        ),

        html.Hr(style={"margin": "18px 0"}),

        html.H3("Heatmap de correlaciones (numéricas)"),
        html.P("Revisa la relación entre resolution_time_log y métricas como reasignaciones, modificaciones, etc."),
        dcc.Graph(id="heatmap", figure=make_corr_heatmap()),

        html.Hr(style={"margin": "18px 0"}),

        # ====== ANOVA ======
        html.H3("ANOVA por variables categóricas"),
        html.P("Selecciona las variables categóricas para evaluar su poder explicativo individual sobre Y."),
        dcc.Dropdown(
            id="anova-cats",
            options=[{"label": c, "value": c} for c in cat_vars],
            value=cat_vars[:5] if len(cat_vars) > 0 else [],
            multi=True,
            placeholder="Selecciona variables categóricas"
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[dcc.Graph(id="anova-f"), dcc.Graph(id="anova-p")]
        ),

        html.Hr(style={"margin": "18px 0"}),

        # ====== Recurrencia temporal ======
        html.H3("Patrones de recurrencia de incidentes (sin duplicados por number + opened_at)"),
        html.P("Ajusta el rango y la agregación temporal para detectar picos y momentos del mes con más incidentes."),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                html.Div([
                    html.Div("Rango de fechas:", style={"fontWeight": 600}),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=DATE_MIN,
                        end_date=DATE_MAX,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        display_format="YYYY-MM-DD"
                    ),
                ]),
                html.Div([
                    html.Div("Agregación temporal:", style={"fontWeight": 600}),
                    dcc.Dropdown(
                        id="time-agg",
                        options=[
                            {"label": "Diaria", "value": "D"},
                            {"label": "Semanal", "value": "W"},
                            {"label": "Mensual", "value": "M"},
                        ],
                        value="D",
                        clearable=False
                    ),
                ]),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "16px", "marginTop": "8px"},
            children=[dcc.Graph(id="ts-incidents"), dcc.Graph(id="hist-dom")]
        ),
        html.Hr(style={"margin": "18px 0"}),

        # ====== Constancia por canal ======
        html.H3("¿Por qué medio se reportan de forma más constante los incidentes?"),
        html.P("Explora la estabilidad por canal (menor CV = más constante). Usa el periodo y filtra canales."),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                # Dropdown de PERIODO  (id correcto: ch-period)
                html.Div([
                    html.Div("Periodo de agregación:", style={"fontWeight": 600}),
                    dcc.Dropdown(
                        id="ch-period",
                        options=[
                            {"label": "Diario", "value": "D"},
                            {"label": "Semanal", "value": "W"},
                            {"label": "Mensual", "value": "M"},
                        ],
                        value="W",   # semanal por defecto
                        clearable=False
                    ),
                ]),

                # Dropdown de CANALES (id único: ch-filter)
                html.Div([
                    html.Div("Canales (contact_type):", style={"fontWeight": 600}),
                    dcc.Dropdown(
                        id="ch-filter",
                        options=[
                            {"label": str(x), "value": x}
                            for x in sorted(
                                df_time["contact_type"].dropna().unique(),
                                key=lambda v: str(v).lower()
                            )
                        ] if "contact_type" in df_time.columns else [],
                        value=None,
                        multi=True,
                        placeholder="(Opcional) Selecciona canales"
                    )
                ])
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "16px", "marginTop": "8px"},
            children=[
                dcc.Graph(id="ch-ts"),   # serie temporal por canal
                dcc.Graph(id="ch-cv"),   # barras de CV
            ]
        ),
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


@app.callback(
    Output("anova-f", "figure"),
    Output("anova-p", "figure"),
    Input("anova-cats", "value"),
)
def update_anova(anova_cats):
    use_cats = [c for c in (anova_cats or []) if c in df.columns]
    anova_df = compute_anova(use_cats)
    fig_f, fig_p = make_anova_figs(anova_df)
    return fig_f, fig_p


@app.callback(
    Output("ts-incidents", "figure"),
    Output("hist-dom", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("time-agg", "value"),
)
def update_time_recurrence(start_date, end_date, agg):
    ts_fig, dom_fig = make_recurrence_figures(
        df_src=df_time,
        start_date=start_date,
        end_date=end_date,
        agg=agg,
        number_col="number",
        opened_col="opened_at"
    )
    return ts_fig, dom_fig
@app.callback(
    Output("ch-ts", "figure"),
    Output("ch-cv", "figure"),
    Input("ch-period", "value"),
    Input("ch-filter", "value"),
)

def update_channel_constancy(period, channels_selected):
    channels = channels_selected if channels_selected else None
    fig_ts, fig_cv = make_channel_constancy_figures(
        df_src=df_time,
        channel_col="contact_type",   # cambia si tu columna se llama distinto
        date_col="opened_at",
        period=period,
        channels_filter=channels
    )
    return fig_ts, fig_cv

# ========= Main =========
if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1", port=8055)


