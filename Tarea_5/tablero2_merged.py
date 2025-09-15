
import os, json, joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

import dash
from dash import Dash, dcc, html, Input, Output

# ================= App base =================
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Incidentes — Exploración y Modelos (robusto)"

# ================= Paths / artifacts (robustos) =================
CSV_PATH = "incident_event_log_clean.csv"

def _resolve_artdir():
    """Resuelve una carpeta de artifacts de forma robusta (var de entorno o rutas conocidas)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    cands = [
        os.environ.get("ARTIFACTS_DIR", "").strip(),
        os.path.join(repo_root, "Tarea_4", "artifacts"),
        os.path.join(repo_root, "Tarea_4 artifacts"),
        os.path.join(here, "Tarea_4", "artifacts"),
    ]
    for p in cands:
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    # fallback por defecto
    return os.path.abspath(os.path.join(repo_root, "Tarea_4", "artifacts"))

ART_DIR = _resolve_artdir()

def placeholder_fig(title: str):
    """Figura vacía amigable para mensajes/errores."""
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=30, r=20, t=60, b=40))
    return fig

def debug_check_artifacts(art_dir=ART_DIR):
    """Devuelve rutas requeridas y cuáles faltan, para mensajes claros en el dashboard."""
    req = {
        "model":   os.path.join(art_dir, "model_dnn2.keras"),
        "ohe":     os.path.join(art_dir, "ohe_pipe.joblib"),
        "te_maps": os.path.join(art_dir, "te_maps.json"),
        "flists":  os.path.join(art_dir, "feature_lists.json"),
        # opcionales para evaluación:
        "preds":   os.path.join(art_dir, "predictions_test.csv"),
        "metrics": os.path.join(art_dir, "metrics.json"),
    }
    missing = {k: v for k, v in req.items() if not os.path.exists(v)}
    return req, missing

def load_predict_artifacts(art_dir=ART_DIR):
    """Carga modelo + OHE + TE + listas. Mensajes claros si algo falta."""
    paths, missing = debug_check_artifacts(art_dir)
    # requeridos para predicción
    required = ["model","ohe","te_maps","flists"]
    if any(k in missing for k in required):
        print("[ARTIFACTS] Faltan archivos requeridos:")
        for k in required:
            if k in missing:
                print(f"  - {k}: {missing[k]}")
        print("[ARTIFACTS] Busca en:", art_dir)
        print("[ARTIFACTS] Opciones: 1) genera desde Tarea_4, 2) exporta ARTIFACTS_DIR con la ruta correcta.")
        return None
    try:
        model = tf.keras.models.load_model(paths["model"])
        ohe_pipe = joblib.load(paths["ohe"])
        with open(paths["te_maps"], "r") as f:
            te_maps = json.load(f)
        with open(paths["flists"], "r") as f:
            lists = json.load(f)
    except Exception as e:
        print(f"[ARTIFACTS] Error cargando:", e)
        return None
    return {
        "model": model, "ohe_pipe": ohe_pipe, "te_maps": te_maps,
        "LOW_CAT": lists.get("LOW_CAT", []), "HIGH_CAT": lists.get("HIGH_CAT", []),
        "paths": paths
    }

def load_model_artifacts_for_eval(art_dir=ART_DIR):
    preds_path = os.path.join(art_dir, "predictions_test.csv")
    metrics_path = os.path.join(art_dir, "metrics.json")
    if not (os.path.exists(preds_path) and os.path.exists(metrics_path)):
        return None, None
    preds = pd.read_csv(preds_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return preds, pd.DataFrame(metrics)

# ================= Datos base =================
df = pd.read_csv(CSV_PATH)

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
if Y not in df.columns:
    raise ValueError(f"No se encuentra la columna '{Y}' en el CSV.")
df = df[np.isfinite(df[Y])]

if "opened_at" not in df.columns:
    raise ValueError("La columna 'opened_at' no existe en el CSV.")
if "number" not in df.columns:
    raise ValueError("La columna 'number' no existe en el CSV.")

df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")

# df_time deduplicado por (number, opened_at)
df_time = (
    df.dropna(subset=["opened_at", "number"])
      .drop_duplicates(subset=["number", "opened_at"])
      .copy()
)
if "contact_type" in df_time.columns:
    df_time["contact_type"] = df_time["contact_type"].astype(str)

if not df_time.empty:
    df_time["date"] = df_time["opened_at"].dt.date
    df_time["day_of_month"] = df_time["opened_at"].dt.day
    DATE_MIN = df_time["opened_at"].min().date()
    DATE_MAX = df_time["opened_at"].max().date()
else:
    DATE_MIN = pd.Timestamp("2016-01-01").date()
    DATE_MAX = pd.Timestamp("2016-12-31").date()

# ================= Helpers de figuras / EDA =================
def make_boxplot(factor: str):
    if factor not in df.columns:
        return px.box(title=f"Variable {factor} no encontrada")
    order = (df.groupby(factor)[Y].mean().sort_values(ascending=True).index.tolist())
    fig = px.box(df, x=factor, y=Y, category_orders={factor: order},
                 points="outliers", title=f"Distribución de {Y} por {factor}")
    fig.update_layout(xaxis_title=factor, yaxis_title=Y,
                      margin=dict(l=30, r=20, t=60, b=80))
    fig.update_xaxes(tickangle=45)
    return fig

def make_corr_heatmap():
    cols = [Y] + num_vars
    data = df[cols].copy()
    if data.shape[1] < 2:
        return px.imshow(np.array([[1.0]]), x=[Y], y=[Y],
                         title="No hay suficientes variables numéricas",
                         text_auto=True)
    corr = data.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, title="Correlaciones (incluye resolution_time_log)")
    fig.update_layout(margin=dict(l=30, r=20, t=60, b=40))
    return fig

def compute_anova(cat_list):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    results = []
    for var in (cat_list or []):
        if var not in df.columns:
            continue
        try:
            model = smf.ols(f"{Y} ~ C({var})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            row = anova_table.iloc[0]
            results.append({"Variable": var,
                            "F_value": float(row["F"]),
                            "p_value": float(row["PR(>F)"])})
        except Exception:
            pass
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values("F_value", ascending=False)
    return res_df

def make_anova_figs(anova_df: pd.DataFrame):
    if anova_df.empty:
        msg = "ANOVA: sin variables o sin resultados"
        return placeholder_fig(msg), placeholder_fig(msg)
    fig_f = px.bar(anova_df, x="F_value", y="Variable", orientation="h",
                   title="ANOVA — F-value (mayor = más explicativa)")
    pvals = anova_df["p_value"].replace(0, np.nextafter(0, 1))
    anova_df_disp = anova_df.copy()
    anova_df_disp["neg_log10_p"] = -np.log10(pvals)
    fig_p = px.bar(anova_df_disp, x="neg_log10_p", y="Variable", orientation="h",
                   title="ANOVA — −log10(p-value)")
    for fig in (fig_f, fig_p):
        fig.update_layout(margin=dict(l=30, r=20, t=60, b=40))
    return fig_f, fig_p

def make_recurrence_figures(
    df_src: pd.DataFrame, start_date=None, end_date=None,
    agg: str = "D", number_col: str = "number", opened_col: str = "opened_at"
):
    if number_col not in df_src.columns or opened_col not in df_src.columns:
        msg = f"Faltan columnas '{number_col}' y/o '{opened_col}'"
        return placeholder_fig(msg), placeholder_fig(msg)

    tmp = df_src.copy()
    tmp[opened_col] = pd.to_datetime(tmp[opened_col], errors="coerce")
    tmp = tmp.dropna(subset=[opened_col, number_col]).drop_duplicates(subset=[number_col, opened_col])
    if tmp.empty:
        return placeholder_fig("Sin datos tras limpieza"), placeholder_fig("Sin datos tras limpieza")

    start_date = (tmp[opened_col].min().date() if start_date is None
                  else pd.to_datetime(start_date).date())
    end_date = (tmp[opened_col].max().date() if end_date is None
                else pd.to_datetime(end_date).date())

    mask = (tmp[opened_col].dt.date >= start_date) & (tmp[opened_col].dt.date <= end_date)
    sub = tmp.loc[mask].copy()
    if sub.empty:
        return placeholder_fig("Sin datos en el rango"), placeholder_fig("Sin datos en el rango")

    sub = sub.set_index(opened_col).sort_index()
    counts = sub.resample(agg)[number_col].nunique()
    ts_df = counts.reset_index()
    ts_df.columns = [opened_col, "count"]

    titulo_periodo = {"D": "Día", "W": "Semana", "M": "Mes"}.get(agg, agg)
    ts_fig = px.line(ts_df, x=opened_col, y="count",
                     title=f"Incidentes únicos por periodo ({titulo_periodo})")
    ts_fig.update_layout(xaxis_title="Fecha", yaxis_title="Incidentes únicos",
                         margin=dict(l=30, r=20, t=60, b=40))

    sub["day_of_month"] = sub.index.day
    dom_fig = px.histogram(sub.reset_index(), x="day_of_month", nbins=31,
                           title="Distribución de incidentes por día del mes (únicos)")
    dom_fig.update_layout(xaxis_title="Día del mes", yaxis_title="Cantidad de incidentes únicos",
                          margin=dict(l=30, r=20, t=60, b=40))
    dom_fig.update_xaxes(dtick=1)
    return ts_fig, dom_fig

def make_channel_constancy_figures(
    df_src: pd.DataFrame, channel_col: str = "contact_type",
    date_col: str = "opened_at", period: str = "W", channels_filter=None
):
    if channel_col not in df_src.columns or date_col not in df_src.columns:
        msg = f"Faltan columnas '{channel_col}' y/o '{date_col}'"
        return placeholder_fig(msg), placeholder_fig(msg)

    tmp = df_src.dropna(subset=[channel_col, date_col]).copy()
    if channels_filter:
        tmp = tmp[tmp[channel_col].isin(channels_filter)]
    if tmp.empty:
        return placeholder_fig("Sin datos para los canales seleccionados"), placeholder_fig("Sin datos para los canales seleccionados")

    tmp = tmp.set_index(date_col).sort_index()
    grp = (tmp.groupby(channel_col).resample(period)["number"].nunique()
              .rename("count").reset_index())

    titulo_periodo = {"D": "diario", "W": "semanal", "M": "mensual"}.get(period, period)
    fig_ts = px.line(grp, x=date_col, y="count", color=channel_col,
                     title=f"Incidentes únicos por {titulo_periodo} y canal ({channel_col})")
    fig_ts.update_layout(xaxis_title="Fecha", yaxis_title="Incidentes únicos",
                         margin=dict(l=30, r=20, t=60, b=40))

    cv_df = (grp.groupby(channel_col)["count"].agg(mean="mean", std="std").reset_index())
    cv_df["CV"] = cv_df.apply(lambda r: (r["std"] / r["mean"]) if pd.notnull(r["mean"]) and r["mean"] != 0 else np.nan, axis=1)
    cv_df = cv_df.sort_values("CV", ascending=True)

    fig_cv = px.bar(cv_df, x="CV", y=channel_col, orientation="h",
                    hover_data=["mean","std"],
                    title="Constancia por canal — Coeficiente de Variación (menor = más constante)")
    fig_cv.update_layout(xaxis_title="CV (std/mean)", yaxis_title="Canal",
                         margin=dict(l=30, r=20, t=60, b=40))
    return fig_ts, fig_cv

# ====== Frecuencia & criticidad (se conserva del tablero2 original) ======
def make_freq_critical_figure(
    df: pd.DataFrame,
    type_col: str | None = None,   # "category" o "subcategory"
    top_n: int = 15,
    chart: str = "bar"             # "bar" o "scatter"
):
    if type_col is None:
        if "category" in df.columns:
            type_col = "category"
        elif "subcategory" in df.columns:
            type_col = "subcategory"
        else:
            return px.bar(title="No existe 'category' ni 'subcategory' en el dataset.")
    if type_col not in df.columns:
        return px.bar(title=f"La columna '{type_col}' no existe en el dataset.")
    if "number" not in df.columns:
        return px.bar(title="No existe la columna 'number' para deduplicar incidentes.")

    d0 = df.drop_duplicates(subset=["number"]).copy()
    is_critical = pd.Series(False, index=d0.index)

    # priority
    if "priority" in d0.columns:
        pr = d0["priority"].astype(str).str.lower()
        is_critical |= pr.str.contains(r"\bp1\b|critical|high|1\s*-\s*critical|2\s*-\s*high", regex=True)
    # impact
    if "impact" in d0.columns:
        imp = d0["impact"].astype(str).str.lower()
        is_critical |= imp.eq("1") | imp.str.contains(r"\bhigh\b|1\s*-\s*high|1\s*-\s*critical")
        with np.errstate(invalid="ignore"):
            is_critical |= pd.to_numeric(d0["impact"], errors="coerce").eq(1)
    # urgency
    if "urgency" in d0.columns:
        urg = d0["urgency"].astype(str).str.lower()
        is_critical |= urg.eq("1") | urg.str.contains(r"\bhigh\b|1\s*-\s*high|1\s*-\s*critical")
        with np.errstate(invalid="ignore"):
            is_critical |= pd.to_numeric(d0["urgency"], errors="coerce").eq(1)

    d0["is_critical"] = is_critical.fillna(False)
    g = (d0.groupby(type_col, dropna=False)
           .agg(incidents=("number", "nunique"),
                critical=("is_critical", "sum"))
           .reset_index())
    g["critical_rate"] = np.where(g["incidents"] > 0, g["critical"] / g["incidents"], np.nan)
    g = g.sort_values("incidents", ascending=False).head(top_n)
    g["critical_pct"] = (g["critical_rate"] * 100).round(1)

    if chart == "scatter":
        fig = px.scatter(g, x="incidents", y="critical_rate",
                         size="critical", color="critical_rate",
                         color_continuous_scale="Reds",
                         hover_data={type_col: True, "incidents": True, "critical": True, "critical_pct": True},
                         text=type_col, title=f"Frecuencia vs. % crítico por {type_col} (top {top_n})")
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="Incidentes (frecuencia)",
                          yaxis_title="% crítico",
                          yaxis_tickformat=".0%",
                          margin=dict(l=30, r=20, t=60, b=60))
        return fig

    fig = px.bar(g, x=type_col, y="incidents", color="critical_rate",
                 color_continuous_scale="Reds",
                 hover_data={"critical": True, "critical_pct": True},
                 title=f"Tipos más frecuentes y su % crítico — por {type_col} (top {top_n})")
    fig.update_layout(xaxis_title=type_col, yaxis_title="Incidentes (frecuencia)",
                      margin=dict(l=30, r=20, t=60, b=80))
    fig.update_xaxes(tickangle=45)
    fig.update_traces(text=g["critical_pct"].astype(str) + "% crítico", textposition="outside")
    return fig

# ================= Artifacts (modelo / evaluación) =================
ART = load_predict_artifacts(ART_DIR)
REG_PREDS, REG_METRICS = load_model_artifacts_for_eval(ART_DIR)

def apply_te_value(x_value, mapping: dict, global_mean: float) -> float:
    if x_value is None or (isinstance(x_value, float) and np.isnan(x_value)):
        return float(global_mean)
    return float(mapping.get(str(x_value), global_mean))

def score_single_example(inputs_dict: dict, ART):
    LOW_CAT = ART["LOW_CAT"]; HIGH_CAT = ART["HIGH_CAT"]
    ohe_pipe = ART["ohe_pipe"]; te_maps = ART["te_maps"]; model = ART["model"]

    te_feats = []
    for col in HIGH_CAT:
        te_info = te_maps.get(col, None)
        if te_info is None:
            te_feats.append([0.0])
        else:
            te_val = apply_te_value(inputs_dict.get(col), te_info["map"], te_info["global"])
            te_feats.append([te_val])
    X_te = np.hstack(te_feats).astype("float32") if te_feats else np.empty((1,0), dtype="float32")

    if LOW_CAT:
        df_low = pd.DataFrame({c: [inputs_dict.get(c)] for c in LOW_CAT})
        X_ohe = ohe_pipe.transform(df_low)
        X_ohe = np.asarray(X_ohe, dtype="float32")
    else:
        X_ohe = np.empty((1,0), dtype="float32")

    X = np.hstack([X_te, X_ohe]).astype("float32")  # Normalization está dentro del modelo
    y_pred = model.predict(X, verbose=0).reshape(-1)[0]
    return float(y_pred)

# ================= Controles para predicción interactiva =================
if ART is not None:
    PRED_LOW = [c for c in ART["LOW_CAT"] if c in df.columns]
    PRED_HIGH = [c for c in ART["HIGH_CAT"] if c in df.columns]
    PRED_COLS = PRED_LOW + PRED_HIGH

    def col_dropdown(col):
        opts = sorted(df[col].dropna().astype(str).unique(), key=lambda v: v.lower()) if col in df.columns else []
        return html.Div([
            html.Div(col, style={"fontWeight":600, "marginBottom":"4px"}),
            dcc.Dropdown(
                id=f"pred-{col}",
                options=[{"label": o, "value": o} for o in opts],
                value=None, placeholder="Selecciona o escribe...", searchable=True, clearable=True
            )
        ])
    PRED_CONTROLS = [col_dropdown(c) for c in PRED_COLS]
else:
    PRED_COLS = []
    PRED_CONTROLS = [html.Div(f"No hay artifacts. Genera en: {ART_DIR}",
                              style={"color":"#aa0000", "fontWeight":600})]

# ================= Layout =================
app.layout = html.Div(
    style={"fontFamily":"Inter, system-ui, Arial", "padding":"18px"},
    children=[
        html.H2("Exploración preliminar: ¿Qué factores explican el tiempo de resolución?"),
        html.P("Comparar distribución por factor categórico y revisar correlaciones numéricas."),

        # Boxplot
        html.Div(style={"display":"flex","gap":"14px","flexWrap":"wrap","alignItems":"center"}, children=[
            html.Div("Factor categórico:", style={"fontWeight":600}),
            dcc.Dropdown(
                id="factor-dropdown",
                options=[{"label": c, "value": c} for c in cat_vars] or [{"label":"(no hay factores)","value":""}],
                value=(cat_vars[0] if len(cat_vars) else ""), clearable=False, style={"minWidth":"260px"}
            ),
        ]),
        html.Div(style={"marginTop":"8px"}, children=[dcc.Graph(id="boxplot",
            figure=make_boxplot(cat_vars[0]) if cat_vars else {})]),

        html.Hr(style={"margin":"18px 0"}),

        # Heatmap
        html.H3("Heatmap de correlaciones"),
        dcc.Graph(id="heatmap", figure=make_corr_heatmap()),

        html.Hr(style={"margin":"18px 0"}),

        # ANOVA
        html.H3("ANOVA por variables categóricas"),
        dcc.Dropdown(
            id="anova-cats",
            options=[{"label": c, "value": c} for c in cat_vars],
            value=cat_vars[:5] if len(cat_vars)>0 else [],
            multi=True, placeholder="Selecciona variables"
        ),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"},
                 children=[dcc.Graph(id="anova-f"), dcc.Graph(id="anova-p")]),

        html.Hr(style={"margin":"18px 0"}),

        # Recurrencia
        html.H3("Patrones de recurrencia (sin duplicados por number + opened_at)"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            html.Div([
                html.Div("Rango de fechas:", style={"fontWeight":600}),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=DATE_MIN, end_date=DATE_MAX,
                    min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                    display_format="YYYY-MM-DD"
                ),
            ]),
            html.Div([
                html.Div("Agregación temporal:", style={"fontWeight":600}),
                dcc.Dropdown(
                    id="time-agg",
                    options=[{"label":"Diaria","value":"D"},
                             {"label":"Semanal","value":"W"},
                             {"label":"Mensual","value":"M"}],
                    value="D", clearable=False
                ),
            ]),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1.5fr 1fr","gap":"16px","marginTop":"8px"},
                 children=[dcc.Graph(id="ts-incidents"), dcc.Graph(id="hist-dom")]),

        html.Hr(style={"margin":"18px 0"}),

        # Constancia por canal
        html.H3("¿Por qué medio se reportan de forma más constante los incidentes?"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
            html.Div([
                html.Div("Periodo de agregación:", style={"fontWeight":600}),
                dcc.Dropdown(
                    id="ch-period",
                    options=[{"label":"Diario","value":"D"},
                             {"label":"Semanal","value":"W"},
                             {"label":"Mensual","value":"M"}],
                    value="W", clearable=False
                ),
            ]),
            html.Div([
                html.Div("Canales (contact_type):", style={"fontWeight":600}),
                dcc.Dropdown(
                    id="ch-filter",
                    options=[
                        {"label": str(x), "value": x}
                        for x in (sorted(df_time["contact_type"].dropna().unique(), key=lambda v: str(v).lower())
                                  if "contact_type" in df_time.columns else [])
                    ],
                    value=None, multi=True, placeholder="(Opcional) Selecciona canales"
                )
            ]),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1.5fr 1fr","gap":"16px","marginTop":"8px"},
                 children=[dcc.Graph(id="ch-ts"), dcc.Graph(id="ch-cv")]),

        html.Hr(style={"margin":"18px 0"}),

        # ===== Frecuencia y criticidad =====
        html.H3("Tipos más frecuentes y críticos"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                dcc.Dropdown(
                    id="fc-type",
                    options=[opt for opt in [{"label":"category","value":"category"},{"label":"subcategory","value":"subcategory"}] if opt["value"] in df.columns],
                    value="category" if "category" in df.columns else ("subcategory" if "subcategory" in df.columns else None),
                    clearable=False
                ),
                dcc.Dropdown(
                    id="fc-chart",
                    options=[{"label":"Barras","value":"bar"},{"label":"Dispersión (bubble)","value":"scatter"}],
                    value="bar", clearable=False
                ),
                dcc.Slider(id="fc-topn", min=5, max=30, step=1, value=15, marks=None, tooltip={"placement":"bottom","always_visible":True}),
            ]
        ),
        dcc.Graph(id="freq-critical"),

        html.Hr(style={"margin":"18px 0"}),

        # Evaluación de modelo
        html.H3("Evaluación del modelo de regresión (tiempo de resolución)"),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"12px"}, children=[
            dcc.Dropdown(
                id="reg-model",
                options=[{"label":"DNN (1 oculta)", "value":"DNN_1_hidden"},
                         {"label":"DNN (2 ocultas)","value":"DNN_2_hidden"}],
                value="DNN_2_hidden", clearable=False
            ),
            dcc.RadioItems(id="reg-show-metrics",
                           options=[{"label":"Mostrar métricas","value":"yes"}],
                           value="yes",
                           labelStyle={"display":"inline-block","marginRight":"12px"}),
            html.Div(id="reg-metrics")
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1.1fr 0.9fr","gap":"16px","marginTop":"8px"},
                 children=[dcc.Graph(id="reg-pred-vs-true"), dcc.Graph(id="reg-resid-vs-pred")]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr","gap":"16px","marginTop":"8px"},
                 children=[dcc.Graph(id="reg-resid-hist")]),

        html.Hr(style={"margin":"18px 0"}),

        # Predicción interactiva
        html.H3("Predicción interactiva del tiempo de resolución"),
        html.P(f"Artifacts buscados en: {ART_DIR} (o variable de entorno ARTIFACTS_DIR)"),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(3, minmax(220px, 1fr))","gap":"12px"},
                 children=PRED_CONTROLS),
        html.Div(style={"display":"flex","gap":"12px","alignItems":"center","marginTop":"12px"},
                 children=[
                     html.Button("Calcular predicción", id="pred-btn", n_clicks=0, className="button"),
                     html.Div(id="pred-result", style={"fontSize":"18px","fontWeight":600})
                 ]),
    ],
)

# ================= Callbacks =================
@app.callback(Output("boxplot","figure"), Input("factor-dropdown","value"))
def update_boxplot(factor_value):
    if not factor_value:
        return px.box(title="Selecciona un factor")
    return make_boxplot(factor_value)

@app.callback(Output("anova-f","figure"), Output("anova-p","figure"),
              Input("anova-cats","value"))
def update_anova(anova_cats):
    use_cats = [c for c in (anova_cats or []) if c in df.columns]
    anova_df = compute_anova(use_cats)
    fig_f, fig_p = make_anova_figs(anova_df)
    return fig_f, fig_p

@app.callback(Output("ts-incidents","figure"), Output("hist-dom","figure"),
              Input("date-range","start_date"), Input("date-range","end_date"),
              Input("time-agg","value"))
def update_time_recurrence(start_date, end_date, agg):
    ts_fig, dom_fig = make_recurrence_figures(
        df_src=df_time, start_date=start_date, end_date=end_date,
        agg=agg, number_col="number", opened_col="opened_at"
    )
    return ts_fig, dom_fig

@app.callback(Output("ch-ts","figure"), Output("ch-cv","figure"),
              Input("ch-period","value"), Input("ch-filter","value"))
def update_channel_constancy(period, channels_selected):
    channels = channels_selected if channels_selected else None
    fig_ts, fig_cv = make_channel_constancy_figures(
        df_src=df_time, channel_col="contact_type", date_col="opened_at",
        period=period, channels_filter=channels
    )
    return fig_ts, fig_cv

@app.callback(Output("freq-critical","figure"),
              Input("fc-type","value"), Input("fc-chart","value"), Input("fc-topn","value"))
def update_freq_critical(tcol, chart_kind, topn):
    return make_freq_critical_figure(df, type_col=tcol, top_n=int(topn or 15), chart=chart_kind or "bar")

@app.callback(Output("reg-pred-vs-true","figure"),
              Output("reg-resid-vs-pred","figure"),
              Output("reg-resid-hist","figure"),
              Input("reg-model","value"))
def update_regression_figs(model_key):
    if REG_PREDS is None:
        msg = f"No se encontraron predicciones en {os.path.join(ART_DIR,'predictions_test.csv')}. Genera desde Tarea_4."
        return placeholder_fig(msg), placeholder_fig(msg), placeholder_fig(msg)
    return make_regression_figures(REG_PREDS, model_key)

@app.callback(Output("reg-metrics","children"),
              Input("reg-model","value"), Input("reg-show-metrics","value"))
def show_regression_metrics(model_key, show):
    if show != "yes":
        return html.Div()
    if REG_METRICS is None:
        return html.Div(f"Métricas no encontradas. Genera {os.path.join(ART_DIR,'metrics.json')}",
                        style={"color":"#aa0000"})
    row = REG_METRICS[REG_METRICS["model"] == model_key]
    if row.empty:
        return html.Div(f"Sin métricas para {model_key}", style={"color":"#aa0000"})
    r = row.iloc[0]
    return html.Div(style={"display":"flex","gap":"16px","alignItems":"stretch"}, children=[
        html.Div([html.Div("MAE", style={"fontWeight":600}), html.Div(f"{r['MAE']:.3f} h")],
                 style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
        html.Div([html.Div("MSE", style={"fontWeight":600}), html.Div(f"{r['MSE']:.3f}")],
                 style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
        html.Div([html.Div("R²", style={"fontWeight":600}), html.Div(f"{r['R2']:.4f}")],
                 style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
    ])

# Predicción interactiva
if ART is not None:
    from dash.dependencies import Input as DInput
    pred_inputs = [DInput(f"pred-{c}", "value") for c in PRED_COLS]

    @app.callback(Output("pred-result","children"),
                  [Input("pred-btn","n_clicks")] + pred_inputs,
                  prevent_initial_call=True)
    def on_predict(n_clicks, *values):
        if ART is None:
            return f"Modelo no disponible. Genera artifacts en: {ART_DIR}"
        inputs_dict = {}
        for c, v in zip(PRED_COLS, values):
            inputs_dict[c] = None if v in (None, "", "None", "nan") else v
        try:
            y_hat = score_single_example(inputs_dict, ART)
            return f"Tiempo estimado: {y_hat:.2f} horas"
        except Exception as e:
            return f"Error al predecir: {e}"
else:
    @app.callback(Output("pred-result","children"), Input("pred-btn","n_clicks"), prevent_initial_call=True)
    def _no_artifacts(_):
        return f"Modelo no disponible. Genera artifacts en: {ART_DIR}"

# ================= Main =================
if __name__ == "__main__":
    # usa el puerto 8070 de tablerofinalcopia para no chocar con otros servicios
    app.run(debug=True, host="127.0.0.1", port=8090)
