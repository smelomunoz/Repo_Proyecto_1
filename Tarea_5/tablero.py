import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# Funciones auxiliares
# ----------------------------
def load_dictionary(dict_path):
    field_help = {}
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, desc = line.split(":", 1)
                    field_help[key.strip()] = desc.strip()
    except Exception:
        pass
    return field_help

def prepare_df(csv_path):
    df = pd.read_csv(csv_path)

    # parse datetimes
    for col in ["opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # durations
    df["t_resolve_h"] = (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600.0
    df["t_close_h"]   = (df["closed_at"]   - df["opened_at"]).dt.total_seconds() / 3600.0
    for tcol in ["t_resolve_h","t_close_h"]:
        if tcol in df.columns:
            df.loc[(df[tcol] < 0) | (df[tcol] > 24*90), tcol] = np.nan

    # time helpers
    if "opened_at" in df.columns:
        df["date"] = df["opened_at"].dt.date
        df["week"] = df["opened_at"].dt.to_period("W").astype(str)
        df["dow"] = df["opened_at"].dt.day_name()
        df["hour"] = df["opened_at"].dt.hour
        df["day"] = df["opened_at"].dt.day
    return df

# ----------------------------
# Construcción del Dash App
# ----------------------------
def build_app(df, field_help):
    app = Dash(__name__)
    app.title = "Incidentes TI – Tablero"

    cat_cols = [c for c in ["contact_type","category","subcategory","impact","urgency","priority","assignment_group"] if c in df.columns]
    num_cols = [c for c in ["reassignment_count","reopen_count","sys_mod_count","made_sla"] if c in df.columns]
    if "made_sla" in df.columns:
        if df["made_sla"].dtype == bool:
            df["made_sla_num"] = df["made_sla"].astype(int)
        else:
            df["made_sla_num"] = df["made_sla"].map({True:1, False:0, "true":1,"false":0,"True":1,"False":0}).fillna(0).astype(int)
        num_cols.append("made_sla_num")

    def options_from_series(s):
        vals = sorted([v for v in s.dropna().unique()])
        return [{"label": str(v), "value": v} for v in vals]

    # ----------------------------
    # Layout
    # ----------------------------
    controls = html.Div([
        html.Div([
            html.Label("Rango de fechas"),
            dcc.DatePickerRange(
                id="date_range",
                min_date_allowed=df["opened_at"].min().date() if "opened_at" in df.columns else None,
                max_date_allowed=df["opened_at"].max().date() if "opened_at" in df.columns else None,
                start_date=df["opened_at"].min().date() if "opened_at" in df.columns else None,
                end_date=df["opened_at"].max().date() if "opened_at" in df.columns else None,
                display_format="YYYY-MM-DD"
            ),
        ], style={"marginRight":"12px"}),

        html.Div([
            html.Label("Medio de reporte"),
            dcc.Dropdown(
                id="contact_type_dd",
                options=options_from_series(df["contact_type"]) if "contact_type" in df.columns else [],
                multi=True,
                placeholder="Todos"
            )
        ], style={"minWidth":"240px","marginRight":"12px"}),

        html.Div([
            html.Label("Categoría"),
            dcc.Dropdown(
                id="category_dd",
                options=options_from_series(df["category"]) if "category" in df.columns else [],
                multi=True,
                placeholder="Todas"
            )
        ], style={"minWidth":"240px","marginRight":"12px"}),

        html.Div([
            html.Label("Grupo asignado"),
            dcc.Dropdown(
                id="group_dd",
                options=options_from_series(df["assignment_group"]) if "assignment_group" in df.columns else [],
                multi=True,
                placeholder="Todos"
            )
        ], style={"minWidth":"240px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"8px","alignItems":"end","marginBottom":"8px"})

    kpis = html.Div([
        html.Div([html.H4("Tiempo mediano a resolver (h)"), html.H2(id="kpi_med_resolve")], className="card"),
        html.Div([html.H4("Tiempo mediano a cerrar (h)"), html.H2(id="kpi_med_close")], className="card"),
        html.Div([html.H4("# Incidentes"), html.H2(id="kpi_count")], className="card"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(3, 1fr)","gap":"12px","marginTop":"8px"})

    app.layout = html.Div([
        html.H1("Tablero de Incidentes TI"),
        controls,
        kpis,
        html.Hr(),
        dcc.Tabs([
            dcc.Tab(label="1) Factores de tiempo a resolver", children=[
                html.Button("Entrenar modelo (RandomForest)", id="fit_btn", n_clicks=0),
                html.Div(id="model_status"),
                dcc.Graph(id="feat_importance_fig"),
            ]),
            dcc.Tab(label="2) Tiempo hasta 'close'", children=[
                dcc.Graph(id="close_distribution"),
                html.Div(id="close_summary")
            ]),
            dcc.Tab(label="3) Patrones de recurrencia", children=[
                dcc.Graph(id="heatmap_dow_hour"),
                dcc.Graph(id="incidents_by_day")
            ]),
            dcc.Tab(label="4) Medio de reporte", children=[
                dcc.Graph(id="contact_counts"),
                html.Div(id="contact_cv_table")
            ]),
            dcc.Tab(label="5) Incidentes frecuentes y críticos", children=[
                dcc.Graph(id="top_categories"),
                dcc.Graph(id="critical_breakdown")
            ]),
            dcc.Tab(label="Datos filtrados", children=[
                dash_table.DataTable(
                    id="table_sample",
                    page_size=10,
                    style_table={"overflowX":"auto"},
                    style_cell={"fontFamily":"monospace","fontSize":"12px"},
                )
            ]),
        ])
    ], style={"maxWidth":"1300px","margin":"0 auto","fontFamily":"sans-serif"})

    # ----------------------------
    # Función de filtrado
    # ----------------------------
    def apply_filters(_df, start_date, end_date, contact_types, categories, groups):
        df2 = _df.copy()
        if "opened_at" in df2.columns:
            if start_date is not None:
                df2 = df2[df2["opened_at"] >= pd.to_datetime(start_date)]
            if end_date is not None:
                df2 = df2[df2["opened_at"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
        if contact_types:
            df2 = df2[df2["contact_type"].isin(contact_types)]
        if categories:
            df2 = df2[df2["category"].isin(categories)]
        if groups:
            df2 = df2[df2["assignment_group"].isin(groups)]
        return df2

    # ----------------------------
    # Callbacks
    # ----------------------------
    @app.callback(
        Output("kpi_med_resolve","children"),
        Output("kpi_med_close","children"),
        Output("kpi_count","children"),
        Output("table_sample","data"),
        Output("table_sample","columns"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("contact_type_dd","value"),
        Input("category_dd","value"),
        Input("group_dd","value"),
    )
    def update_kpis(sd, ed, ct, cat, grp):
        dff = apply_filters(df, sd, ed, ct, cat, grp)
        med_res = np.nanmedian(dff["t_resolve_h"]) if "t_resolve_h" in dff.columns else np.nan
        med_close = np.nanmedian(dff["t_close_h"]) if "t_close_h" in dff.columns else np.nan
        sample = dff.head(10)
        cols = [{"name": c, "id": c} for c in sample.columns]
        return (
            f"{med_res:.1f}" if pd.notnull(med_res) else "NA",
            f"{med_close:.1f}" if pd.notnull(med_close) else "NA",
            f"{len(dff):,}",
            sample.to_dict("records"),
            cols
        )

    @app.callback(
        Output("model_status","children"),
        Output("feat_importance_fig","figure"),
        Input("fit_btn","n_clicks"),
        prevent_initial_call=True
    )
    def fit_model(n):
        dff = df.dropna(subset=["t_resolve_h"]).copy()
        y = dff["t_resolve_h"]
        use_num = [c for c in ["reassignment_count","reopen_count","sys_mod_count","made_sla_num"] if c in dff.columns]
        use_cat = [c for c in ["contact_type","category","subcategory","impact","urgency","priority","assignment_group"] if c in dff.columns]
        X = dff[use_num + use_cat]
        if len(X) < 100:
            return "Muy pocos datos para entrenar", go.Figure()
        ctg = ColumnTransformer([
            ("num","passthrough", use_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), use_cat)
        ])
        pipe = Pipeline([
            ("prep", ctg),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=17, n_jobs=-1))
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        importances = pipe.named_steps["rf"].feature_importances_
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(use_cat)) if len(use_cat)>0 else []
        feature_names = use_num + cat_names
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
        fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Importancia de variables (RandomForest)")
        return f"Modelo entrenado. R²: {score:.3f}", fig

    @app.callback(
        Output("close_distribution","figure"),
        Output("close_summary","children"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("contact_type_dd","value"),
        Input("category_dd","value"),
        Input("group_dd","value"),
    )
    def close_time_view(sd, ed, ct, cat, grp):
        dff = apply_filters(df, sd, ed, ct, cat, grp)
        series = dff["t_close_h"].dropna()
        if series.empty:
            return go.Figure(), "Sin datos"
        fig = px.histogram(series, nbins=40, title="Distribución del tiempo a cerrar (horas)")
        summ = {
            "Mediana (h)": np.nanmedian(series),
            "Promedio (h)": np.nanmean(series),
            "P90 (h)": np.nanpercentile(series, 90),
            "N": len(series)
        }
        return fig, str(summ)

    @app.callback(
        Output("heatmap_dow_hour","figure"),
        Output("incidents_by_day","figure"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("contact_type_dd","value"),
        Input("category_dd","value"),
        Input("group_dd","value"),
    )
    def recurrence(sd, ed, ct, cat, grp):
        dff = apply_filters(df, sd, ed, ct, cat, grp)
        tmp = dff.dropna(subset=["dow","hour"]).groupby(["dow","hour"]).size().reset_index(name="count")
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        if "dow" in tmp.columns:
            tmp["dow"] = pd.Categorical(tmp["dow"], categories=order, ordered=True)
        fig1 = px.density_heatmap(tmp, x="hour", y="dow", z="count", title="Incidentes por día y hora")
        byday = dff.groupby("day").size().reset_index(name="count")
        fig2 = px.bar(byday, x="day", y="count", title="Incidentes por día del mes")
        return fig1, fig2

    @app.callback(
        Output("contact_counts","figure"),
        Output("contact_cv_table","children"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("contact_type_dd","value"),
        Input("category_dd","value"),
        Input("group_dd","value"),
    )
    def contact_constancy(sd, ed, ct, cat, grp):
        dff = apply_filters(df, sd, ed, ct, cat, grp)
        if "contact_type" not in dff.columns:
            return go.Figure(), "Columna no encontrada"
        counts = dff.groupby("contact_type").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(counts, x="contact_type", y="count", title="Incidentes por medio de reporte")
        return fig, counts.to_dict("records")

    @app.callback(
        Output("top_categories","figure"),
        Output("critical_breakdown","figure"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("contact_type_dd","value"),
        Input("category_dd","value"),
        Input("group_dd","value"),
    )
    def frequent_critical(sd, ed, ct, cat, grp):
        dff = apply_filters(df, sd, ed, ct, cat, grp)
        if "category" in dff.columns:
            topc = dff.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False).head(15)
            fig1 = px.bar(topc, x="category", y="count", title="Top categorías")
        else:
            fig1 = go.Figure()
        crit_mask = pd.Series(False, index=dff.index)
        if "impact" in dff.columns:
            crit_mask |= dff["impact"].astype(str).str.startswith("1", na=False)
        if "urgency" in dff.columns:
            crit_mask |= dff["urgency"].astype(str).str.startswith("1", na=False)
        if "priority" in dff.columns:
            crit_mask |= dff["priority"].astype(str).str.contains("1|High", na=False)
        dff["is_critical"] = np.where(crit_mask, "Crítico", "No crítico")
        bycat = dff.groupby(["category","is_critical"]).size().reset_index(name="count")
        fig2 = px.bar(bycat, x="category", y="count", color="is_critical", barmode="group",
                      title="Críticos vs No críticos")
        return fig1, fig2

    return app

# ----------------------------
# MAIN
# ----------------------------
def main():
    csv_path = "incident_event_log_clean.csv"
    dict_path = "diccionario.txt"

    print(f"[INFO] Cargando CSV: {csv_path}")
    print(f"[INFO] Cargando diccionario: {dict_path}")

    df = prepare_df(csv_path)
    field_help = load_dictionary(dict_path)
    app = build_app(df, field_help)
    app.run(debug=True, port=8060, use_reloader=False)

if __name__ == "__main__":
    main()
