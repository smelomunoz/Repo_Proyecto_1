import dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os, json, joblib, tensorflow as tf
import plotly.graph_objects as go 

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

ART_DIR = "Tarea_4/artifacts" 

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

def load_predict_artifacts(art_dir=ART_DIR):
    paths = {
        "model1": os.path.join(art_dir, "model_dnn1.keras"),
        "model2": os.path.join(art_dir, "model_dnn2.keras"),
        "temaps": os.path.join(art_dir, "te_maps.json"),
        "ohe":    os.path.join(art_dir, "ohe_pipe.joblib"),
        "flists": os.path.join(art_dir, "feature_lists.json"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            return None

    with open(paths["temaps"], "r") as f:
        te_maps = json.load(f)
    with open(paths["flists"], "r") as f:
        lists = json.load(f)

    ohe_pipe = joblib.load(paths["ohe"])
    model2 = tf.keras.models.load_model(paths["model2"])  # usamos el de 2 capas por defecto

    LOW_CAT = lists["LOW_CAT"]
    HIGH_CAT = lists["HIGH_CAT"]
    return {"model": model2, "ohe_pipe": ohe_pipe, "te_maps": te_maps, "LOW_CAT": LOW_CAT, "HIGH_CAT": HIGH_CAT}

ART = load_predict_artifacts()

if ART is not None:
    PRED_LOW = [c for c in ART["LOW_CAT"] if c in df.columns]
    PRED_HIGH = [c for c in ART["HIGH_CAT"] if c in df.columns]
    PRED_COLS = PRED_LOW + PRED_HIGH

    def col_dropdown(col):
        # opciones desde los datos (puedes escribir un valor no listado)
        opts = sorted(df[col].dropna().astype(str).unique(), key=lambda v: v.lower()) if col in df.columns else []
        return html.Div([
            html.Div(col, style={"fontWeight":600, "marginBottom":"4px"}),
            dcc.Dropdown(
                id=f"pred-{col}",
                options=[{"label": o, "value": o} for o in opts],
                value=None,
                placeholder="Selecciona o escribe...",
                searchable=True,
                clearable=True
            )
        ])

    PRED_CONTROLS = [col_dropdown(c) for c in PRED_COLS]
else:
    PRED_LOW = PRED_HIGH = PRED_COLS = []
    PRED_CONTROLS = [html.Div("Aún no hay artefactos del modelo. Genera Tarea_4/artifacts.", style={"color":"#aa0000"})]

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

def make_freq_critical_figure(
    df: pd.DataFrame,
    type_col: str | None = None,   # "category" o "subcategory"
    top_n: int = 15,
    chart: str = "bar"             # "bar" o "scatter"
):

    # --------- escoger columna de tipo ---------
    if type_col is None:
        if "category" in df.columns:
            type_col = "category"
        elif "subcategory" in df.columns:
            type_col = "subcategory"
        else:
            return px.bar(title="No existe 'category' ni 'subcategory' en el dataset.")

    if type_col not in df.columns:
        return px.bar(title=f"La columna '{type_col}' no existe en el dataset.")

    # --------- deduplicar por incidente ---------
    if "number" not in df.columns:
        return px.bar(title="No existe la columna 'number' para deduplicar incidentes.")
    d0 = df.drop_duplicates(subset=["number"]).copy()

    # --------- construir bandera de criticidad ---------
    is_critical = pd.Series(False, index=d0.index)

    # priority como fuente principal
    if "priority" in d0.columns:
        pr = d0["priority"].astype(str).str.lower()
        # match P1, 1 - Critical, Critical, High, P2 High, etc.
        is_critical |= pr.str.contains(r"\bp1\b|critical|high|1\s*-\s*critical|2\s*-\s*high", regex=True)

    # si no hay priority fuerte, usar impact
    if "impact" in d0.columns:
        imp = d0["impact"].astype(str).str.lower()
        # numérico 1 o texto "high" / "1 - high"
        is_critical |= imp.eq("1") | imp.str.contains(r"\bhigh\b|1\s*-\s*high|1\s*-\s*critical")

        # si impact es numérico
        with np.errstate(invalid="ignore"):
            is_critical |= pd.to_numeric(d0["impact"], errors="coerce").eq(1)

    # como fallback, urgency
    if "urgency" in d0.columns:
        urg = d0["urgency"].astype(str).str.lower()
        is_critical |= urg.eq("1") | urg.str.contains(r"\bhigh\b|1\s*-\s*high|1\s*-\s*critical")
        with np.errstate(invalid="ignore"):
            is_critical |= pd.to_numeric(d0["urgency"], errors="coerce").eq(1)

    d0["is_critical"] = is_critical.fillna(False)

    # --------- agregación por tipo ---------
    g = (
        d0.groupby(type_col, dropna=False)
          .agg(incidents=("number", "nunique"),
               critical=("is_critical", "sum"))
          .reset_index()
    )
    g["critical_rate"] = np.where(g["incidents"] > 0, g["critical"] / g["incidents"], np.nan)

    # ordenar por frecuencia y tomar top_n
    g = g.sort_values("incidents", ascending=False).head(top_n)

    # etiquetas bonitas
    g["critical_pct"] = (g["critical_rate"] * 100).round(1)

    # --------- graficar ---------
    if chart == "scatter":
        fig = px.scatter(
            g,
            x="incidents",
            y="critical_rate",
            size="critical",
            color="critical_rate",
            color_continuous_scale="Reds",
            hover_data={type_col: True, "incidents": True, "critical": True, "critical_pct": True},
            text=type_col,
            title=f"Frecuencia vs. % crítico por {type_col} (top {top_n})"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Incidentes (frecuencia)",
            yaxis_title="% crítico",
            yaxis_tickformat=".0%",
            margin=dict(l=30, r=20, t=60, b=60)
        )
        return fig

    # barras coloreadas por % crítico
    fig = px.bar(
        g,
        x=type_col, y="incidents",
        color="critical_rate",
        color_continuous_scale="Reds",
        hover_data={"critical": True, "critical_pct": True},
        title=f"Tipos más frecuentes y su % crítico — por {type_col} (top {top_n})"
    )
    fig.update_layout(
        xaxis_title=type_col,
        yaxis_title="Incidentes (frecuencia)",
        margin=dict(l=30, r=20, t=60, b=80)
    )
    fig.update_xaxes(tickangle=45)
    # añadir % crítico como texto sobre la barra
    fig.update_traces(
        text=g["critical_pct"].astype(str) + "% crítico",
        textposition="outside"
    )
    return fig

def load_model_artifacts(art_dir=ART_DIR):
    """Carga predicciones y métricas guardadas por el script de Tarea_4."""
    preds_path = os.path.join(art_dir, "predictions_test.csv")
    metrics_path = os.path.join(art_dir, "metrics.json")
    if not (os.path.exists(preds_path) and os.path.exists(metrics_path)):
        return None, None
    preds = pd.read_csv(preds_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return preds, pd.DataFrame(metrics)


def make_regression_figures(preds_df: pd.DataFrame, model_key: str):
    """
    Devuelve: fig_pred_vs_true, fig_resid_vs_pred, fig_resid_hist
    """
    use = preds_df[preds_df["model"] == model_key].copy()
    if use.empty:
        msg = f"Sin predicciones para '{model_key}'."
        return px.scatter(title=msg), px.scatter(title=msg), px.histogram(title=msg)

    use["residual"] = use["y_true"] - use["y_pred"]

    # 1) Predicho vs Real
    fig1 = px.scatter(
        use, x="y_true", y="y_pred",
        opacity=0.6,
        title=f"Predicho vs Real — {model_key}",
        labels={"y_true":"Real (h)", "y_pred":"Predicho (h)"}
    )
    # línea y=x
    minv = float(np.nanmin([use["y_true"].min(), use["y_pred"].min()]))
    maxv = float(np.nanmax([use["y_true"].max(), use["y_pred"].max()]))
    fig1.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv,
                   line=dict(dash="dash"))
    fig1.update_layout(margin=dict(l=30, r=20, t=60, b=40))

    # 2) Residuos vs Predicho
    fig2 = px.scatter(
        use, x="y_pred", y="residual",
        opacity=0.6,
        title=f"Residuos vs Predicho — {model_key}",
        labels={"y_pred":"Predicho (h)", "residual":"Residuo (h)"},
    )
    fig2.add_hline(y=0, line_dash="dash")
    fig2.update_layout(margin=dict(l=30, r=20, t=60, b=40))

    # 3) Distribución de residuos
    fig3 = px.histogram(
        use, x="residual", nbins=40,
        title=f"Distribución de residuos — {model_key}",
        labels={"residual":"Residuo (h)"}
    )
    fig3.add_vline(x=0, line_dash="dash")
    fig3.update_layout(margin=dict(l=30, r=20, t=60, b=40))

    return fig1, fig2, fig3
def load_predict_artifacts(art_dir=ART_DIR):
    paths = {
        "model1": os.path.join(art_dir, "model_dnn1.keras"),
        "model2": os.path.join(art_dir, "model_dnn2.keras"),
        "temaps": os.path.join(art_dir, "te_maps.json"),
        "ohe":    os.path.join(art_dir, "ohe_pipe.joblib"),
        "flists": os.path.join(art_dir, "feature_lists.json"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            return None

    with open(paths["temaps"], "r") as f:
        te_maps = json.load(f)
    with open(paths["flists"], "r") as f:
        lists = json.load(f)

    ohe_pipe = joblib.load(paths["ohe"])
    model2 = tf.keras.models.load_model(paths["model2"])  # usamos el de 2 capas por defecto

    LOW_CAT = lists["LOW_CAT"]
    HIGH_CAT = lists["HIGH_CAT"]
    return {"model": model2, "ohe_pipe": ohe_pipe, "te_maps": te_maps, "LOW_CAT": LOW_CAT, "HIGH_CAT": HIGH_CAT}

ART = load_predict_artifacts()

def apply_te_series(x_value, mapping: dict, global_mean: float):
    """Aplica TE a un valor (o NaN) devolviendo float."""
    if x_value is None or (isinstance(x_value, float) and np.isnan(x_value)):
        return float(global_mean)
    return float(mapping.get(str(x_value), global_mean))

def score_single_example(inputs_dict: dict, ART):
    """
    inputs_dict: {col: valor} SOLO con columnas LOW_CAT+HIGH_CAT
    Devuelve y_pred (horas) usando el modelo y los artefactos.
    """
    LOW_CAT = ART["LOW_CAT"]; HIGH_CAT = ART["HIGH_CAT"]
    ohe_pipe = ART["ohe_pipe"]; te_maps = ART["te_maps"]; model = ART["model"]

    # --- Target Encoding en HIGH_CAT ---
    te_feats = []
    for col in HIGH_CAT:
        te_info = te_maps.get(col, None)
        if te_info is None:
            # si faltara el mapeo, usa 0 como respaldo
            te_feats.append([0.0])
        else:
            te_val = apply_te_series(inputs_dict.get(col), te_info["map"], te_info["global"])
            te_feats.append([te_val])
    X_te = np.hstack(te_feats).astype("float32") if te_feats else np.empty((1,0), dtype="float32")

    # --- OHE + Imputer en LOW_CAT ---
    if LOW_CAT:
        df_low = pd.DataFrame({c: [inputs_dict.get(c)] for c in LOW_CAT})
        X_ohe = ohe_pipe.transform(df_low)
        X_ohe = np.asarray(X_ohe, dtype="float32")
    else:
        X_ohe = np.empty((1,0), dtype="float32")

    # --- Ensamble y predicción ---
    X = np.hstack([X_te, X_ohe]).astype("float32")  # el modelo ya contiene Normalization
    y_pred = model.predict(X, verbose=0).reshape(-1)[0]
    return float(y_pred)
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
        html.Hr(),
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
        html.Hr(style={"margin": "18px 0"}),

        html.H3("Evaluación del modelo de regresión (tiempo de resolución)"),
        html.P("Selecciona el modelo para ver Predicho vs Real, Residuos vs Predicho y Distribución de residuos."),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                dcc.Dropdown(
                    id="reg-model",
                    options=[
                        {"label": "DNN (1 oculta)", "value": "DNN_1_hidden"},
                        {"label": "DNN (2 ocultas)", "value": "DNN_2_hidden"},
                    ],
                    value="DNN_2_hidden",
                    clearable=False
                ),
                dcc.RadioItems(
                    id="reg-show-metrics",
                    options=[{"label":"Mostrar métricas","value":"yes"}],
                    value="yes",
                    labelStyle={"display":"inline-block", "marginRight":"12px"}
                ),
                html.Div(id="reg-metrics")
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.1fr 0.9fr", "gap": "16px", "marginTop": "8px"},
            children=[
                dcc.Graph(id="reg-pred-vs-true"),
                dcc.Graph(id="reg-resid-vs-pred"),
            ]
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px", "marginTop": "8px"},
            children=[
                dcc.Graph(id="reg-resid-hist"),
            ]
        ),
        html.Hr(style={"margin": "18px 0"}),

        html.H3("Predicción interactiva del tiempo de resolución"),
        html.P("Modifica las variables predictoras para obtener una estimación (horas)."),

        html.Div(
            style={"display":"grid","gridTemplateColumns":"repeat(3, minmax(220px, 1fr))","gap":"12px"},
            children=PRED_CONTROLS
        ),

        html.Div(
            style={"display":"flex","gap":"12px","alignItems":"center","marginTop":"12px"},
            children=[
                html.Button("Calcular predicción", id="pred-btn", n_clicks=0, className="button"),
                html.Div(id="pred-result", style={"fontSize":"18px","fontWeight":600})
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
@app.callback(
    Output("freq-critical","figure"),
    Input("fc-type","value"),
    Input("fc-chart","value"),
    Input("fc-topn","value"),
)
def update_freq_critical(tcol, chart_kind, topn):
    return make_freq_critical_figure(df, type_col=tcol, top_n=int(topn or 15), chart=chart_kind or "bar")

# Carga artefactos una vez (si existen)
REG_PREDS, REG_METRICS = load_model_artifacts()

@app.callback(
    Output("reg-pred-vs-true", "figure"),
    Output("reg-resid-vs-pred", "figure"),
    Output("reg-resid-hist", "figure"),
    Input("reg-model", "value"),
)
def update_regression_figs(model_key):
    if REG_PREDS is None:
        msg = "No se encontraron artefactos. Guarda predictions_test.csv y metrics.json desde Tarea_4."
        f1 = go.Figure(); f1.update_layout(title=msg)
        f2 = go.Figure(); f2.update_layout(title=msg)
        f3 = go.Figure(); f3.update_layout(title=msg)
        return f1, f2, f3

    return make_regression_figures(REG_PREDS, model_key)


@app.callback(
    Output("reg-metrics", "children"),
    Input("reg-model", "value"),
    Input("reg-show-metrics", "value"),
)
def show_regression_metrics(model_key, show):
    if show != "yes":
        return html.Div()
    if REG_METRICS is None:
        return html.Div("Métricas no encontradas. Genera Tarea_4/artifacts/metrics.json.", style={"color":"#aa0000"})

    row = REG_METRICS[REG_METRICS["model"] == model_key]
    if row.empty:
        return html.Div(f"Sin métricas para {model_key}", style={"color":"#aa0000"})
    r = row.iloc[0]
    # Tarjetitas simples con MAE/MSE/R2
    return html.Div(
        style={"display":"flex","gap":"16px","alignItems":"stretch"},
        children=[
            html.Div([
                html.Div("MAE", style={"fontWeight":600}),
                html.Div(f"{r['MAE']:.3f} h")
            ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
            html.Div([
                html.Div("MSE", style={"fontWeight":600}),
                html.Div(f"{r['MSE']:.3f}")
            ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
            html.Div([
                html.Div("R²", style={"fontWeight":600}),
                html.Div(f"{r['R2']:.4f}")
            ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","minWidth":"120px"}),
        ]
    )
# === Callback de predicción interactiva ===
if ART is not None:
    from dash.dependencies import Input as DInput

    pred_inputs = [DInput(f"pred-{c}", "value") for c in PRED_COLS]

    @app.callback(
        Output("pred-result","children"),
        [Input("pred-btn","n_clicks")] + pred_inputs,
        prevent_initial_call=True
    )
    def on_predict(n_clicks, *values):
        if ART is None:
            return "Modelo no disponible. Genera artefactos en Tarea_4."
        # arma diccionario {col: valor}
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
        return "Modelo no disponible. Genera artefactos en Tarea_4."

# ========= Main =========
if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1", port=8055)


