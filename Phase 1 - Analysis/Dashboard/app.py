from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = os.getenv("ALG1_CSV_PATH", str(REPO_ROOT / "alg1_lab_dataset.csv"))

SENSOR_COLUMNS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]
SCENARIO_NAMES = {
    1: "Normal Operation",
    2: "Occupied Laboratory",
    3: "Hazard Development",
    4: "Security Breach",
}
SCENARIO_COLORS = {
    "Normal Operation": "#2f7d5c",
    "Occupied Laboratory": "#3d6fb6",
    "Hazard Development": "#c27a18",
    "Security Breach": "#b33a3a",
}
ACTION_BY_SCENARIO = {
    1: "Monitor",
    2: "OptimizeVentilation",
    3: "VentilateAndAlert",
    4: "LockdownAndNotify",
}


st.set_page_config(
    page_title="ALG-1 Lab Guardian",
    page_icon="ALG",
    layout="wide",
)

px.defaults.template = "plotly_white"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        background: rgba(250, 250, 250, 0.72);
    }
    .small-note {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "timestamp": "Timestamp",
        "temp_c": "Temp_C",
        "humidity_pct": "Humidity_pct",
        "gas_aqi": "Gas_AQI",
        "light_lux": "Light_Lux",
        "motion_detected": "Motion_Detected",
        "true_scenario": "True_Scenario",
    }
    renamed = {}
    for column in df.columns:
        key = str(column).strip().lower()
        if key in aliases:
            renamed[column] = aliases[key]
    return df.rename(columns=renamed)


@st.cache_data(show_spinner=False)
def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(raw_df.copy())
    required = {"Timestamp", "Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected", "True_Scenario"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required ALG-1 columns: {missing}")

    df = df[list(required)].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    numeric_columns = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected", "True_Scenario"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)
    df["Motion_Detected"] = df["Motion_Detected"].astype(int)
    df["True_Scenario"] = df["True_Scenario"].astype(int)
    df["Scenario_Name"] = df["True_Scenario"].map(SCENARIO_NAMES).fillna("Unknown")
    df["Recommended_Action"] = df["True_Scenario"].map(ACTION_BY_SCENARIO).fillna("InspectLab")

    df["Date"] = df["Timestamp"].dt.date.astype(str)
    df["Hour"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["Day_Name"] = df["Timestamp"].dt.day_name()
    df["Is_Night"] = ((df["Hour"] < 6.0) | (df["Hour"] > 22.0)).astype(int)
    df["Is_Work_Window"] = ((df["Hour"] >= 9.0) & (df["Hour"] <= 16.5)).astype(int)

    df["Gas_Slope_30m"] = df["Gas_AQI"].diff(30).fillna(0.0)
    df["Motion_5m"] = df["Motion_Detected"].rolling(5, min_periods=1).sum()
    df["Temp_Delta_1m"] = df["Temp_C"].diff().fillna(0.0)
    df["Gas_Delta_1m"] = df["Gas_AQI"].diff().fillna(0.0)

    gas_pressure = np.clip((df["Gas_AQI"] - 80.0) / 120.0, 0.0, 1.0)
    gas_trend = np.clip(df["Gas_Slope_30m"] / 25.0, 0.0, 1.0)
    temp_pressure = np.clip((df["Temp_C"] - 30.0) / 10.0, 0.0, 1.0)
    night_motion = df["Is_Night"] * np.clip(df["Motion_5m"] / 5.0, 0.0, 1.0)
    df["Hazard_Risk"] = np.clip(0.55 * gas_pressure + 0.30 * gas_trend + 0.15 * temp_pressure, 0.0, 1.0)
    df["Security_Risk"] = np.clip(night_motion, 0.0, 1.0)
    df["Fuzzy_Risk"] = np.maximum(df["Hazard_Risk"], df["Security_Risk"])

    target_action_reward = {
        "Monitor": 0.96,
        "OptimizeVentilation": 0.88,
        "VentilateAndAlert": 0.82,
        "LockdownAndNotify": 0.78,
        "InspectLab": 0.70,
    }
    df["RL_Baseline_Reward"] = df["Recommended_Action"].map(target_action_reward).astype(float)
    df.loc[df["Fuzzy_Risk"] > 0.8, "RL_Baseline_Reward"] -= 0.08
    df["RL_Baseline_Reward"] = df["RL_Baseline_Reward"].clip(0.0, 1.0)

    return df


def scenario_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Scenario_Name"].value_counts().rename_axis("Scenario").reset_index(name="Rows")
    counts["Ratio"] = counts["Rows"] / max(len(df), 1)
    order = list(SCENARIO_NAMES.values())
    counts["Scenario"] = pd.Categorical(counts["Scenario"], categories=order, ordered=True)
    return counts.sort_values("Scenario")


def pca_projection(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    x = df[columns].astype(float).to_numpy()
    x = np.nan_to_num(x, nan=np.nanmedian(x, axis=0))
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:2]
    scores = x @ components.T
    variances = singular_values**2 / max(len(x) - 1, 1)
    explained = variances / variances.sum()
    projected = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "Scenario_Name": df["Scenario_Name"].to_numpy(),
            "Timestamp": df["Timestamp"].astype(str).to_numpy(),
        }
    )
    return projected, explained[:2]


def strongest_edges(corr: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    records = []
    cols = corr.columns.tolist()
    for i, source in enumerate(cols):
        for target in cols[i + 1 :]:
            value = float(corr.loc[source, target])
            records.append(
                {
                    "Source": source,
                    "Target": target,
                    "Weight": abs(value),
                    "Correlation": value,
                    "Relation_Type": "positive" if value >= 0 else "negative",
                }
            )
    return pd.DataFrame(records).sort_values("Weight", ascending=False).head(limit)


def feature_graph_figure(edges: pd.DataFrame, columns: list[str]) -> go.Figure:
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
    positions = {
        column: (float(np.cos(angle)), float(np.sin(angle)))
        for column, angle in zip(columns, angles, strict=False)
    }

    fig = go.Figure()
    for _, row in edges.iterrows():
        x0, y0 = positions[row["Source"]]
        x1, y1 = positions[row["Target"]]
        line_color = "#2f7d5c" if row["Correlation"] >= 0 else "#b33a3a"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=1.5 + 5.5 * row["Weight"], color=line_color),
                hovertemplate=(
                    f"{row['Source']} -> {row['Target']}<br>"
                    f"corr={row['Correlation']:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[positions[column][0] for column in columns],
            y=[positions[column][1] for column in columns],
            mode="markers+text",
            marker=dict(size=28, color="#3d6fb6", line=dict(width=1, color="#1f2937")),
            text=columns,
            textposition="bottom center",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Sensor-Modality Graph",
        xaxis=dict(visible=False, range=[-1.35, 1.35]),
        yaxis=dict(visible=False, range=[-1.25, 1.35]),
        plot_bgcolor="white",
    )
    return fig


def line_chart(df: pd.DataFrame, selected_sensors: list[str], rolling_window: int) -> go.Figure:
    fig = make_subplots(
        rows=len(selected_sensors),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=selected_sensors,
    )
    for row_index, sensor in enumerate(selected_sensors, start=1):
        smoothed = df[sensor].rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[sensor],
                mode="lines",
                name=f"{sensor} raw",
                line=dict(width=1, color="#8a8f98"),
                opacity=0.45,
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=smoothed,
                mode="lines",
                name=f"{sensor} SMA",
                line=dict(width=2.4, color="#b33a3a"),
            ),
            row=row_index,
            col=1,
        )
    fig.update_layout(
        height=max(360, 245 * len(selected_sensors)),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


st.title("Adaptive Lab Guardian ALG-1")
st.caption("University laboratory monitoring dataset for Edge-AI anomaly intelligence.")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    path_input = st.text_input("CSV path", value=DEFAULT_CSV_PATH)

try:
    if uploaded_file is not None:
        raw_df = load_data_from_bytes(uploaded_file.getvalue())
        source_label = uploaded_file.name
    elif path_input and Path(path_input).exists():
        raw_df = load_data_from_path(path_input)
        source_label = path_input
    else:
        st.info("Provide `alg1_lab_dataset.csv` in the sidebar.")
        st.stop()
    df = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Unable to load ALG-1 dataset: {exc}")
    st.stop()

with st.sidebar:
    st.caption(source_label)
    scenario_options = list(SCENARIO_NAMES.values())
    selected_scenarios = st.multiselect("Scenarios", scenario_options, default=scenario_options)
    min_ts = df["Timestamp"].min().to_pydatetime()
    max_ts = df["Timestamp"].max().to_pydatetime()
    selected_range = st.slider("Time window", min_ts, max_ts, (min_ts, max_ts), format="MM/DD HH:mm")
    rolling_window = st.slider("Trend SMA window", 5, 120, 20, 5)

df_filtered = df[
    df["Scenario_Name"].isin(selected_scenarios)
    & (df["Timestamp"] >= pd.Timestamp(selected_range[0]))
    & (df["Timestamp"] <= pd.Timestamp(selected_range[1]))
].copy()

if df_filtered.empty:
    st.warning("No records match the active filters.")
    st.stop()

scenario_counts = scenario_distribution(df_filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(df_filtered):,}")
k2.metric("Time Span", f"{df_filtered['Timestamp'].min():%b %d} - {df_filtered['Timestamp'].max():%b %d}")
k3.metric("Mean Risk", f"{df_filtered['Fuzzy_Risk'].mean():.2f}")
k4.metric("Scenario Coverage", f"{df_filtered['True_Scenario'].nunique()} / 4")

tab_ops, tab_sensors, tab_intel, tab_graph, tab_policy = st.tabs(
    ["Operations", "Sensors", "Intelligence", "Feature Graph", "Policy"]
)

with tab_ops:
    left, right = st.columns([1, 1], gap="large")
    with left:
        fig_dist = px.bar(
            scenario_counts,
            x="Scenario",
            y="Rows",
            color="Scenario",
            color_discrete_map=SCENARIO_COLORS,
            text=scenario_counts["Ratio"].map(lambda value: f"{value:.1%}"),
            title="Scenario Distribution",
        )
        fig_dist.update_layout(showlegend=False, height=420, margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig_dist, width="stretch")
    with right:
        action_counts = (
            df_filtered["Recommended_Action"]
            .value_counts()
            .rename_axis("Action")
            .reset_index(name="Rows")
        )
        fig_actions = px.pie(
            action_counts,
            names="Action",
            values="Rows",
            hole=0.45,
            title="Baseline Action Mix",
        )
        fig_actions.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig_actions, width="stretch")

    fig_timeline = px.scatter(
        df_filtered,
        x="Timestamp",
        y="True_Scenario",
        color="Scenario_Name",
        color_discrete_map=SCENARIO_COLORS,
        title="Scenario Timeline",
        height=360,
    )
    fig_timeline.update_traces(marker=dict(size=4, opacity=0.75))
    fig_timeline.update_yaxes(tickmode="array", tickvals=[1, 2, 3, 4], ticktext=list(SCENARIO_NAMES.values()))
    fig_timeline.update_layout(margin=dict(l=20, r=20, t=50, b=30), legend_title_text="")
    st.plotly_chart(fig_timeline, width="stretch")

with tab_sensors:
    default_sensors = ["Gas_AQI", "Temp_C", "Humidity_pct"]
    selected_sensors = st.multiselect(
        "Sensor channels",
        SENSOR_COLUMNS,
        default=[sensor for sensor in default_sensors if sensor in SENSOR_COLUMNS],
    )
    if selected_sensors:
        st.plotly_chart(line_chart(df_filtered, selected_sensors, rolling_window), width="stretch")
    else:
        st.info("Select at least one sensor channel.")

    detail_left, detail_right = st.columns([1, 1], gap="large")
    with detail_left:
        fig_box = px.box(
            df_filtered,
            x="Scenario_Name",
            y="Gas_AQI",
            color="Scenario_Name",
            color_discrete_map=SCENARIO_COLORS,
            title="Gas AQI by Scenario",
        )
        fig_box.update_layout(showlegend=False, height=390, margin=dict(l=20, r=20, t=50, b=80))
        st.plotly_chart(fig_box, width="stretch")
    with detail_right:
        fig_motion = px.histogram(
            df_filtered,
            x="Hour",
            color="Scenario_Name",
            color_discrete_map=SCENARIO_COLORS,
            nbins=24,
            title="Hourly Scenario Density",
        )
        fig_motion.update_layout(height=390, margin=dict(l=20, r=20, t=50, b=40), legend_title_text="")
        st.plotly_chart(fig_motion, width="stretch")

with tab_intel:
    projected, explained = pca_projection(df_filtered, SENSOR_COLUMNS)
    left, right = st.columns([1, 1], gap="large")
    with left:
        fig_pca = px.scatter(
            projected,
            x="PC1",
            y="PC2",
            color="Scenario_Name",
            color_discrete_map=SCENARIO_COLORS,
            hover_data=["Timestamp"],
            title=f"PCA State Map ({explained[0]:.1%}, {explained[1]:.1%})",
        )
        fig_pca.update_traces(marker=dict(size=5, opacity=0.65))
        fig_pca.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=40), legend_title_text="")
        st.plotly_chart(fig_pca, width="stretch")
    with right:
        anomaly_energy = (
            np.abs((df_filtered[SENSOR_COLUMNS] - df_filtered[SENSOR_COLUMNS].mean()) / df_filtered[SENSOR_COLUMNS].std(ddof=0))
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .max(axis=1)
        )
        energy_df = df_filtered.assign(Anomaly_Energy=anomaly_energy)
        fig_energy = px.histogram(
            energy_df,
            x="Anomaly_Energy",
            color="Scenario_Name",
            color_discrete_map=SCENARIO_COLORS,
            nbins=50,
            title="ART-Style State Novelty Proxy",
        )
        fig_energy.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=40), legend_title_text="")
        st.plotly_chart(fig_energy, width="stretch")

    intelligence_table = pd.DataFrame(
        {
            "Block": ["PCA", "ART", "RBF", "SOM", "Fuzzy", "GNN", "RL", "Genetic Algorithm"],
            "Dataset Signal": [
                "standardized sensor channels",
                "state novelty from sensor energy",
                "rolling trend features",
                "scenario-state clusters",
                "Hazard_Risk and Security_Risk",
                "feature graph edge weights",
                "Recommended_Action and RL_Baseline_Reward",
                "threshold optimization targets",
            ],
            "Status": ["Ready"] * 8,
        }
    )
    st.dataframe(intelligence_table, width="stretch", hide_index=True)

with tab_graph:
    corr = df_filtered[SENSOR_COLUMNS].corr()
    left, right = st.columns([1, 1], gap="large")
    with left:
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Sensor Correlation Matrix",
        )
        fig_corr.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_corr, width="stretch")
    with right:
        edges = strongest_edges(corr, limit=8)
        st.plotly_chart(feature_graph_figure(edges, SENSOR_COLUMNS), width="stretch")
    st.dataframe(edges, width="stretch", hide_index=True)

with tab_policy:
    left, right = st.columns([1, 1], gap="large")
    with left:
        fig_risk = px.line(
            df_filtered,
            x="Timestamp",
            y=["Hazard_Risk", "Security_Risk", "Fuzzy_Risk"],
            title="Fuzzy Risk Channels",
        )
        fig_risk.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40), legend_title_text="")
        st.plotly_chart(fig_risk, width="stretch")
    with right:
        reward_by_action = (
            df_filtered.groupby("Recommended_Action", as_index=False)["RL_Baseline_Reward"]
            .mean()
            .sort_values("RL_Baseline_Reward", ascending=False)
        )
        fig_reward = px.bar(
            reward_by_action,
            x="Recommended_Action",
            y="RL_Baseline_Reward",
            title="Baseline Reward by Action",
        )
        fig_reward.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=80), xaxis_title="")
        st.plotly_chart(fig_reward, width="stretch")

    export_cols = [
        "Timestamp",
        "Temp_C",
        "Humidity_pct",
        "Gas_AQI",
        "Light_Lux",
        "Motion_Detected",
        "Gas_Slope_30m",
        "Motion_5m",
        "Hazard_Risk",
        "Security_Risk",
        "Fuzzy_Risk",
        "Recommended_Action",
        "RL_Baseline_Reward",
        "True_Scenario",
        "Scenario_Name",
    ]
    st.dataframe(df_filtered[export_cols].head(500), width="stretch", height=360)
