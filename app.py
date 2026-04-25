import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Inpatient Bed Usage Predictor", layout="wide")
SEED = 42

@st.cache_resource(show_spinner="Training model on full dataset...")
def load_model_and_data():
    df = pd.read_csv("data/clean/final_featuresv2.csv")
    exclude_cols = [
        "state", "date", "inpatient_beds", "inpatient_beds_used",
        "inpatient_beds_utilization", "inpatient_beds_utilization_coverage",
        "hospital_inpatient_beds", "coverage_per_state",
    ]
    cols = [c for c in df.columns if c not in exclude_cols]
    df = df[cols].dropna().copy()

    X = df.drop(columns="hospital_inpatient_beds_used")
    y = df["hospital_inpatient_beds_used"]

    feature_medians = X.median().to_dict()

    model = RandomForestRegressor(random_state=SEED, max_features="sqrt", n_jobs=-1)
    model.fit(X, y)

    return model, X.columns.tolist(), feature_medians, float(y.mean()), float(y.min()), float(y.max()), df

model, feature_cols, feature_medians, y_mean, y_min, y_max, df = load_model_and_data()

important_features = ['beds_used_lag1', 'beds_used_lag2', 'beds_used_rolling4',
       'state_staffing_shortage_ratio_lag2',
       'previous_day_admission_adult_covid_confirmed_80_normalized',
       'previous_day_admission_adult_covid_confirmed_50_59_normalized',
       'previous_day_admission_adult_covid_suspected_60_69_normalized',
       'previous_day_admission_influenza_confirmed_normalized']

SLIDER_FEATURES = {
    important_features[0]: {
        "label": important_features[0],
        "min": min(df[important_features[0]]),
        "max": max(df[important_features[0]]),
        "step": max(df[important_features[0]])/100,
    },
    important_features[1]: {
        "label": important_features[1],
        "min": min(df[important_features[1]]), 
        "max": max(df[important_features[1]]), 
        "step": max(df[important_features[1]])/100,
    },
    important_features[2]: {
        "label": important_features[2],
        "min": min(df[important_features[2]]), 
        "max": max(df[important_features[2]]), 
        "step": max(df[important_features[2]])/100,
    },
    important_features[3]: {
        "label": important_features[3],
        "min": min(df[important_features[3]]), 
        "max": max(df[important_features[3]]), 
        "step": max(df[important_features[3]])/100,
    },
    important_features[4]: {
        "label": important_features[4],
        "min": min(df[important_features[4]]), 
        "max": max(df[important_features[4]]), 
        "step": max(df[important_features[4]])/100,
    },
    important_features[5]: {
        "label": important_features[5],
        "min": min(df[important_features[5]]), 
        "max": max(df[important_features[5]]), 
        "step": max(df[important_features[5]])/100,
    },
    important_features[6]: {
        "label": important_features[6],
        "min": min(df[important_features[6]]), 
        "max": max(df[important_features[6]]), 
        "step": max(df[important_features[6]])/100,
    },
    important_features[7]: {
        "label": important_features[7],
        "min": min(df[important_features[7]]), 
        "max": max(df[important_features[7]]), 
        "step": max(df[important_features[7]])/100,
    },
}



def build_input_row(slider_vals: dict, feature_cols: list, feature_medians: dict) -> pd.DataFrame:
    row = {col: feature_medians.get(col, 0.0) for col in feature_cols}
    row.update(slider_vals)
    return pd.DataFrame([row])[feature_cols]


def sensitivity_chart(model, slider_vals, feature_cols, feature_medians, steps=60):
    baseline_row = build_input_row(slider_vals, feature_cols, feature_medians)
    impacts = {}

    for feat, cfg in SLIDER_FEATURES.items():
        vals = np.linspace(cfg["min"], cfg["max"], steps)
        preds = []
        for v in vals:
            row = baseline_row.copy()
            row.iloc[0, row.columns.get_loc(feat)] = v
            preds.append(model.predict(row)[0])
        impacts[feat] = {"x": vals, "y": preds, "label": cfg["label"]}

    return impacts


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Inpatient Bed Usage Predictor")
st.caption("Adjust the sliders to explore how each factor affects predicted hospital inpatient bed usage.")


col_sliders, col_charts = st.columns([1, 2], gap="large")

with col_sliders:
    st.subheader("Feature Controls")
    slider_vals = {}
    for feat, cfg in SLIDER_FEATURES.items():
        default = feature_medians.get(feat, (cfg["min"] + cfg["max"]) / 2)
        slider_vals[feat] = st.slider(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(default) if isinstance(default, float) else int(default),
            step=float(cfg["step"]),
        )

input_row = build_input_row(slider_vals, feature_cols, feature_medians)
prediction = model.predict(input_row)[0]

with col_charts:
    # ── Prediction gauge ──────────────────────────────────────────────────────
    st.subheader("Predicted Inpatient Beds Used (% of capacity)")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        delta={"reference": y_mean, "valueformat": ".1f"},
        number={"valueformat": ".1f", "suffix": "%"},
        gauge={
            "axis": {"range": [y_min, y_max], "ticksuffix": "%"},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [y_min, y_mean * 0.85], "color": "#d4edda"},
                {"range": [y_mean * 0.85, y_mean * 1.15], "color": "#fff3cd"},
                {"range": [y_mean * 1.15, y_max], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": y_mean,
            },
        },
        title={"text": f"Mean baseline: {y_mean:.1f}%"},
    ))
    gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

    # ── Delta bar chart ───────────────────────────────────────────────────────
    # st.subheader("Per-Feature Contribution vs. Baseline")
    # st.caption("How much each slider — moved from its median to its current value — shifts the prediction.")

    # # ── Sensitivity lines ─────────────────────────────────────────────────────
    # st.subheader("Feature Impact — How Each Slider Shifts the Prediction")
    # st.caption("Each line shows the predicted bed usage when that feature varies across its full range, with all others held at current slider values.")

    # impacts = sensitivity_chart(model, slider_vals, feature_cols, feature_medians)

    # fig = go.Figure()
    # for feat, data in impacts.items():
    #     current_x = slider_vals[feat]
    #     fig.add_trace(go.Scatter(
    #         x=data["x"], y=data["y"],
    #         mode="lines",
    #         name=data["label"],
    #         hovertemplate=f"{data['label']}<br>Value: %{{x:.3f}}<br>Predicted: %{{y:.1f}}%<extra></extra>",
    #     ))
    #     current_pred = model.predict(build_input_row({**slider_vals, feat: current_x}, feature_cols, feature_medians))[0]
    #     fig.add_trace(go.Scatter(
    #         x=[current_x], y=[current_pred],
    #         mode="markers",
    #         marker=dict(size=9, symbol="circle"),
    #         showlegend=False,
    #         hoverinfo="skip",
    #     ))

    # fig.add_hline(y=y_mean, line_dash="dot", line_color="gray",
    #               annotation_text=f"Mean ({y_mean:.1f}%)", annotation_position="bottom right")
    # fig.update_layout(
    #     xaxis_title="Feature Value",
    #     yaxis_title="Predicted Beds Used (%)",
    #     height=420,
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     margin=dict(t=60, b=40, l=40, r=20),
    #     hovermode="x unified",
    # )
    # st.plotly_chart(fig, use_container_width=True)

    # ── Delta bar chart ───────────────────────────────────────────────────────
    st.subheader("Per-Feature Contribution vs. Baseline")
    st.caption("How much each slider — moved from its median to its current value — shifts the prediction.")

    baseline_pred = model.predict(build_input_row({}, feature_cols, feature_medians))[0]
    deltas = {}
    for feat in SLIDER_FEATURES:
        row_single = build_input_row({feat: slider_vals[feat]}, feature_cols, feature_medians)
        deltas[SLIDER_FEATURES[feat]["label"]] = model.predict(row_single)[0] - baseline_pred

    labels = list(deltas.keys())
    values = list(deltas.values())
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in values]

    bar_fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
    ))
    bar_fig.add_vline(x=0, line_color="black", line_width=1)
    bar_fig.update_layout(
        xaxis_title="Change in Predicted Beds Used (pp)",
        height=320,
        margin=dict(t=20, b=40, l=20, r=60),
        showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True)
