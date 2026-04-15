from __future__ import annotations

import io
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


DEFAULT_CSV_PATH = os.getenv("METRO_CSV_PATH", "")
LOCAL_FALLBACK_CSV_PATH = (
    r"D:\Courses\DEPI R4 - Microsoft ML\Graduation Project\Phase 0 - About Dataset\Dataset\Metro.csv"
)

NOTEBOOK_SENSORS = ["tp2", "tp3", "h1", "oil_temperature", "motor_current"]

COLUMN_DEFINITIONS = {
    "tp2": "Pressure sensor reading at stage TP2.",
    "tp3": "Pressure sensor reading at stage TP3.",
    "h1": "Humidity-related sensor reading H1.",
    "oil_temperature": "Main oil temperature of the compressor system.",
    "motor_current": "Electrical current drawn by compressor motor.",
    "reservoirs": "Reservoir/load indicator from source data.",
    "comp": "Compressor operating state code.",
    "pressure_delta": "Pressure difference between TP3 and TP2.",
    "pressure_change": "Step-to-step change in TP3 pressure.",
    "power_indicator": "Proxy power indicator (motor_current x reservoirs).",
    "oil_temp_rolling": "Smoothed oil temperature (rolling mean window=60).",
    "current_per_pressure": "Motor current normalized by TP2 pressure.",
    "hour": "Hour of day extracted from timestamp.",
    "day_of_week": "Day index extracted from timestamp (0=Monday).",
}

THEME = {
    "primary": "#ff4b4b",
    "background": st.get_option("theme.backgroundColor") or "#ffffff",
    "secondary_bg": st.get_option("theme.secondaryBackgroundColor") or "#f0f2f6",
    "text": st.get_option("theme.textColor") or "#262730",
}

def blend(hex_a: str, hex_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    a = hex_a.lstrip("#")
    b = hex_b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    rr = int(ar * (1 - ratio) + br * ratio)
    rg = int(ag * (1 - ratio) + bg * ratio)
    rb = int(ab * (1 - ratio) + bb * ratio)
    return f"#{rr:02x}{rg:02x}{rb:02x}"


BRAND = {
    "primary": THEME["primary"],
    "secondary": blend(THEME["primary"], THEME["text"], 0.25),
    "accent": blend(THEME["primary"], "#ffffff", 0.35),
    "ink": THEME["text"],
    "muted": blend(THEME["text"], THEME["background"], 0.45),
    "grid": blend(THEME["text"], THEME["background"], 0.82),
    "axis": blend(THEME["text"], THEME["background"], 0.70),
    "bg": THEME["background"],
    "surface": THEME["secondary_bg"],
}

UNIFIED_DISCRETE_COLORS = [
    BRAND["secondary"],
    BRAND["accent"],
    blend(BRAND["primary"], "#ffffff", 0.25),
    blend(BRAND["primary"], "#ffffff", 0.40),
    blend(BRAND["primary"], "#ffffff", 0.55),
]

UNIFIED_SEQUENTIAL_SCALE = [
    [0.0, blend(BRAND["primary"], "#ffffff", 0.90)],
    [0.25, blend(BRAND["primary"], "#ffffff", 0.70)],
    [0.5, blend(BRAND["primary"], "#ffffff", 0.50)],
    [0.75, blend(BRAND["primary"], "#ffffff", 0.25)],
    [1.0, BRAND["primary"]],
]

UNIFIED_DIVERGING_SCALE = [
    [0.0, blend(BRAND["primary"], "#0f172a", 0.35)],
    [0.5, blend(BRAND["bg"], "#ffffff", 0.15)],
    [1.0, BRAND["primary"]],
]

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = UNIFIED_DISCRETE_COLORS


st.set_page_config(
    page_title="Metro Compressor Dashboard",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BRAND["bg"]};
    }}
    .block-container {{padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1400px;}}
    .hero {{
        background: linear-gradient(135deg, {BRAND["primary"]} 0%, {BRAND["secondary"]} 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.12);
        margin-bottom: 0.9rem;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }}
    .hero p {{
        margin: 0.3rem 0 0;
        font-size: 0.98rem;
        color: rgba(255, 255, 255, 0.92);
    }}
    div[data-testid="stMetric"] {{
        border: 1px solid {BRAND["grid"]};
        border-left: 4px solid {BRAND["primary"]};
        background: {BRAND["surface"]};
        border-radius: 12px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.65rem;
    }}
    div[data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}
    div[data-testid="stContainer"] {{
        margin-bottom: 0.75rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        border: 1px solid {BRAND["grid"]};
        background: {BRAND["surface"]};
        color: {BRAND["ink"]};
        padding-left: 14px;
        padding-right: 14px;
    }}
    .obs-box {{
        border: 1px solid {BRAND["grid"]};
        background: {BRAND["surface"]};
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
    }}
    .obs-box p {{
        margin: 0.2rem 0;
        color: {BRAND["ink"]};
        font-size: 0.95rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("How To Read This Dashboard", expanded=False):
    st.markdown(
        """
        - Use **Filters** in the sidebar to focus on a period or compressor state.
        - In **Correlations**, values near `+1` move together, near `-1` move opposite.
        - In **Sensor Trends**, compare raw signal and smoothed trend to spot drift/anomalies.
        - In **Metric Explorer**, pick any metric to see definition, stats, spread, and timeline.
        """
    )


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BRAND["surface"],
        plot_bgcolor=BRAND["bg"],
        font={"family": "Segoe UI, sans-serif", "color": BRAND["ink"], "size": 12},
        margin={"l": 16, "r": 150, "t": 58, "b": 16},
        legend={
            "orientation": "v",
            "y": 1.0,
            "yanchor": "top",
            "x": 1.02,
            "xanchor": "left",
            "title": None,
            "bgcolor": BRAND["surface"],
            "bordercolor": BRAND["grid"],
            "borderwidth": 1,
        },
        hovermode="x unified",
        title={"x": 0.01, "xanchor": "left", "font": {"size": 18, "color": BRAND["ink"]}},
    )
    fig.update_xaxes(showgrid=True, gridcolor=BRAND["grid"], showline=True, linecolor=BRAND["axis"])
    fig.update_yaxes(showgrid=True, gridcolor=BRAND["grid"], showline=True, linecolor=BRAND["axis"])
    if height is not None:
        fig.update_layout(height=height)
    return fig


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def prepare_data(raw_df: pd.DataFrame) -> dict:
    df = raw_df.copy()
    duplicate_count = int(df.duplicated().sum())
    nulls = df.isnull().sum().sort_values(ascending=False)

    unnamed_cols = [col for col in df.columns if str(col).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    df.columns = [str(col).lower().replace(" ", "_").strip() for col in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("Column 'timestamp' is required to build this dashboard.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_numeric(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    engineered_cols: list[str] = []

    if {"tp2", "tp3"}.issubset(df.columns):
        df["pressure_delta"] = df["tp3"] - df["tp2"]
        df["pressure_change"] = df["tp3"].diff()
        engineered_cols.extend(["pressure_delta", "pressure_change"])

    if {"motor_current", "reservoirs"}.issubset(df.columns):
        df["power_indicator"] = df["motor_current"] * df["reservoirs"]
        engineered_cols.append("power_indicator")

    if "oil_temperature" in df.columns:
        df["oil_temp_rolling"] = df["oil_temperature"].rolling(window=60).mean()
        engineered_cols.append("oil_temp_rolling")

    if {"motor_current", "tp2"}.issubset(df.columns):
        df["current_per_pressure"] = df["motor_current"] / (df["tp2"] + 0.1)
        engineered_cols.append("current_per_pressure")

    if engineered_cols:
        df = df.dropna(subset=engineered_cols)
    else:
        df = df.dropna()

    gap_series = df.index.to_series().diff().dt.total_seconds().fillna(0)
    gap_series.name = "gap_seconds"
    top_gaps = gap_series.sort_values(ascending=False).head(5)
    positive_gaps = gap_series[gap_series > 0]
    median_step_seconds = float(positive_gaps.median()) if not positive_gaps.empty else 10.0

    return {
        "raw_rows": len(raw_df),
        "duplicates": duplicate_count,
        "nulls": nulls,
        "df": df,
        "gap_series": gap_series,
        "top_gaps": top_gaps,
        "median_step_seconds": median_step_seconds,
    }


def format_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.1f}h"


def pretty_name(column: str) -> str:
    return column.replace("_", " ").title()


def column_description(column: str) -> str:
    return COLUMN_DEFINITIONS.get(column, "No custom description. Metric available from source dataset.")


def render_observations(title: str, observations: list[str]) -> None:
    body = "".join(f"<p>- {item}</p>" for item in observations if item)
    st.markdown(f"#### {title}", unsafe_allow_html=False)
    st.markdown(f"<div class='obs-box'>{body}</div>", unsafe_allow_html=True)


def strongest_correlation(corr_matrix: pd.DataFrame) -> tuple[str, float] | None:
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return None
    upper = corr_matrix.where(~pd.DataFrame(
        [[i >= j for j in range(len(corr_matrix.columns))] for i in range(len(corr_matrix.columns))],
        index=corr_matrix.index,
        columns=corr_matrix.columns,
    ))
    stacked = upper.stack()
    if stacked.empty:
        return None
    pair = stacked.abs().idxmax()
    value = stacked.loc[pair]
    return f"{pair[0]} <-> {pair[1]}", float(value)


st.markdown(
    """
    <div class="hero">
      <h1>Metro Compressor Monitoring Dashboard</h1>
      <p>Professional analytics view from your notebook: data quality, sensor behavior, correlations, and compressor operations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    path_input = st.text_input("Or CSV path", value=DEFAULT_CSV_PATH)
    st.caption("For deployment, upload a CSV file or set `METRO_CSV_PATH` in environment variables.")

raw_df: pd.DataFrame
source_label: str
try:
    if uploaded_file is not None:
        raw_df = load_data_from_bytes(uploaded_file.getvalue())
        source_label = f"Uploaded file: {uploaded_file.name}"
    elif path_input and Path(path_input).exists():
        raw_df = load_data_from_path(path_input)
        source_label = f"Local path: {path_input}"
    elif Path(LOCAL_FALLBACK_CSV_PATH).exists():
        raw_df = load_data_from_path(LOCAL_FALLBACK_CSV_PATH)
        source_label = f"Local fallback path: {LOCAL_FALLBACK_CSV_PATH}"
    else:
        st.info("Upload a CSV file or set a valid local path in the sidebar.")
        st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

prepared = prepare_data(raw_df)
df = prepared["df"].copy()

if df.empty:
    st.warning("No rows remain after cleaning and feature engineering.")
    st.stop()

with st.sidebar:
    st.caption(source_label)
    st.header("Filters")
    min_dt = df.index.min().to_pydatetime()
    max_dt = df.index.max().to_pydatetime()
    total_span = max_dt - min_dt
    base_seconds = max(1, int(prepared["median_step_seconds"]))

    if total_span > timedelta(days=365):
        slider_step = timedelta(days=7)
    elif total_span > timedelta(days=90):
        slider_step = timedelta(days=1)
    elif total_span > timedelta(days=14):
        slider_step = timedelta(hours=1)
    elif total_span > timedelta(days=2):
        slider_step = timedelta(minutes=5)
    else:
        slider_step = timedelta(seconds=base_seconds)

    selected_range = st.slider(
        "Time range",
        min_value=min_dt,
        max_value=max_dt,
        value=(min_dt, max_dt),
        format="YYYY-MM-DD HH:mm:ss",
        step=slider_step,
    )
    st.caption(f"Time step: {slider_step}")

start_time, end_time = selected_range
if start_time > end_time:
    start_time, end_time = end_time, start_time

df_filtered = df.loc[start_time:end_time].copy()

if "comp" in df_filtered.columns:
    with st.sidebar:
        comp_options = sorted(df_filtered["comp"].dropna().unique().tolist())
        selected_comp = st.multiselect("Compressor state (`comp`)", comp_options, default=comp_options)
    if selected_comp:
        df_filtered = df_filtered[df_filtered["comp"].isin(selected_comp)]
    else:
        df_filtered = df_filtered.iloc[0:0]

if df_filtered.empty:
    st.warning("No records match the active filters.")
    st.stop()

filtered_gap_series = df_filtered.index.to_series().diff().dt.total_seconds().fillna(0)
filtered_positive_gaps = filtered_gap_series[filtered_gap_series > 0]
filtered_median_step_seconds = (
    float(filtered_positive_gaps.median())
    if not filtered_positive_gaps.empty
    else float(prepared["median_step_seconds"])
)

with st.sidebar:
    st.header("Trend Settings")
    total_rows = len(df_filtered)
    default_start = max(0, total_rows - min(1000, total_rows))
    trend_row_range = st.slider(
        "Row range across filtered rows",
        min_value=0,
        max_value=max(0, total_rows - 1),
        value=(default_start, max(0, total_rows - 1)),
        step=1,
    )
    rolling_window = st.slider("Smoothing window (SMA)", 5, 120, 20, 5)

time_span_seconds = (df_filtered.index.max() - df_filtered.index.min()).total_seconds()
null_total = int(prepared["nulls"].sum())

st.markdown("### Executive Snapshot")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records (filtered)", f"{len(df_filtered):,}")
k2.metric("Time Span", format_seconds(time_span_seconds))
k3.metric("Duplicates (raw)", f"{prepared['duplicates']:,}")
k4.metric("Missing Values (raw)", f"{null_total:,}")

if "oil_temperature" in df_filtered.columns:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Oil Temp", f"{df_filtered['oil_temperature'].mean():.2f}")
    c2.metric("Max Oil Temp", f"{df_filtered['oil_temperature'].max():.2f}")
    if "motor_current" in df_filtered.columns:
        c3.metric("Avg Motor Current", f"{df_filtered['motor_current'].mean():.2f}")

top_gap_seconds = float(filtered_gap_series.max()) if not filtered_gap_series.empty else 0.0
render_observations(
    "Executive Observations",
    [
        f"Current filter contains {len(df_filtered):,} records over {format_seconds(time_span_seconds)}.",
        f"Largest observed sampling gap is {format_seconds(top_gap_seconds)}.",
        "Use Metric Explorer for detailed, per-metric interpretation and trend context.",
    ],
)

tab_overview, tab_corr, tab_trends, tab_ops, tab_explorer = st.tabs(
    ["Overview", "Correlations", "Sensor Trends", "Compressor Operations", "Metric Explorer"]
)

with tab_overview:
    st.markdown("### Data Quality Overview")
    st.subheader("Data Quality and Continuity")
    gaps_filtered = filtered_gap_series
    top_gaps = gaps_filtered.sort_values(ascending=False).head(5).rename("gap_seconds").reset_index()
    top_gaps = top_gaps.rename(columns={"index": "timestamp"})
    chart_left, chart_right = st.columns(2, gap="medium")

    with chart_left:
        with st.container(border=True):
            nulls_df = df_filtered.isnull().sum()
            nulls_df = nulls_df[nulls_df > 0].sort_values(ascending=False).reset_index()
            nulls_df.columns = ["feature", "missing_count"]

            if nulls_df.empty:
                st.success("No missing values in the selected time range.")
            else:
                fig_nulls = px.bar(
                    nulls_df.head(20),
                    x="missing_count",
                    y="feature",
                    orientation="h",
                    color="missing_count",
                    color_continuous_scale=UNIFIED_SEQUENTIAL_SCALE,
                    labels={
                        "missing_count": "Missing Count",
                        "feature": "Feature",
                    },
                    title="Top Missing Features",
                )
                fig_nulls.update_layout(yaxis_title="")
                style_figure(fig_nulls, height=420)
                st.plotly_chart(fig_nulls, width="stretch")
                st.caption("Higher bars indicate features with more missing values in the selected time range.")

    with chart_right:
        with st.container(border=True):
            fig_gap = px.histogram(
                x=gaps_filtered,
                nbins=50,
                log_y=True,
                color_discrete_sequence=[BRAND["secondary"]],
                labels={"x": "Gap Duration (seconds)", "count": "Frequency"},
                title="Time Gap Distribution (Log Frequency)",
            )
            fig_gap.update_layout(xaxis_title="Gap Duration (seconds)", yaxis_title="Frequency (log)")
            style_figure(fig_gap, height=420)
            st.plotly_chart(fig_gap, width="stretch")
            st.caption("Right-tail gaps suggest interruptions or irregular sampling intervals.")

    st.subheader("Detail Tables")
    table_left, table_right = st.columns(2, gap="medium")

    with table_left:
        with st.container(border=True):
            st.caption("Top 5 Time Gaps")
            st.dataframe(top_gaps, width="stretch", hide_index=True, height=250)

    with table_right:
        with st.container(border=True):
            st.caption("Processed Data Preview")
            st.dataframe(df_filtered.head(20), width="stretch", height=250)

    with st.container(border=True):
        dictionary = pd.DataFrame(
            {
                "column": df_filtered.columns,
                "type": [str(df_filtered[col].dtype) for col in df_filtered.columns],
                "description": [column_description(col) for col in df_filtered.columns],
            }
        )
        st.caption("Data Dictionary")
        st.dataframe(dictionary, width="stretch", height=260, hide_index=True)
    missing_feature_count = int((df_filtered.isnull().sum() > 0).sum())
    render_observations(
        "Overview Observations",
        [
            f"{missing_feature_count} feature(s) currently contain missing values in the selected range.",
            f"Top data-gap timestamp is {top_gaps['timestamp'].iloc[0] if not top_gaps.empty else 'N/A'}.",
            "Data dictionary below helps explain each field before deeper analysis.",
        ],
    )

with tab_corr:
    st.markdown("### Correlation Analysis")
    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to compute correlation.")
    else:
        corr_matrix = df_filtered[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale=UNIFIED_DIVERGING_SCALE,
            zmin=-1,
            zmax=1,
            aspect="auto",
            text_auto=".2f",
            labels={"x": "Feature", "y": "Feature", "color": "Correlation"},
            title="Correlation Heatmap - All Numeric Features",
        )
        style_figure(fig_corr, height=650)
        st.plotly_chart(fig_corr, width="stretch")
        st.caption("Use this map to quickly identify strongly related sensor signals.")
        strongest = strongest_correlation(corr_matrix)
        corr_obs = (
            f"Strongest relationship: {strongest[0]} with correlation {strongest[1]:.2f}."
            if strongest
            else "Not enough data to calculate a strongest pair."
        )
        render_observations(
            "Correlation Observations",
            [
                corr_obs,
                "Correlations show linear relationship strength, not causation.",
            ],
        )

    available_important = [col for col in NOTEBOOK_SENSORS if col in df_filtered.columns]
    if len(available_important) >= 2:
        important_corr = df_filtered[available_important].corr(numeric_only=True)
        fig_important = px.imshow(
            important_corr,
            color_continuous_scale=UNIFIED_DIVERGING_SCALE,
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            labels={"x": "Feature", "y": "Feature", "color": "Correlation"},
            title="Mechanical Sensors Correlation Map",
        )
        style_figure(fig_important, height=520)
        st.plotly_chart(fig_important, width="stretch")
        st.caption("Focused view of core mechanical sensors from your notebook.")

with tab_trends:
    st.markdown("### Sensor Trend Monitoring")
    st.subheader("Sensor Behavior with Moving Average")

    candidate_sensors = [
        col for col in (NOTEBOOK_SENSORS + ["oil_temp_rolling", "pressure_delta", "power_indicator"]) if col in df_filtered.columns
    ]
    candidate_sensors = list(dict.fromkeys(candidate_sensors))
    defaults = [col for col in NOTEBOOK_SENSORS if col in candidate_sensors][:3]
    sensor_display = {f"{pretty_name(col)} ({col})": col for col in candidate_sensors}
    default_display = [
        f"{pretty_name(col)} ({col})"
        for col in (defaults if defaults else candidate_sensors[:3])
    ]
    selected_display = st.multiselect(
        "Sensors to plot",
        options=list(sensor_display.keys()),
        default=default_display,
    )
    selected_sensors = [sensor_display[item] for item in selected_display]

    row_start, row_end = trend_row_range
    trend_df = df_filtered.iloc[row_start : row_end + 1]
    if not selected_sensors:
        st.info("Select at least one sensor to render trend charts.")
    elif trend_df.empty:
        st.info("Selected row range has no records.")
    else:
        fig = make_subplots(
            rows=len(selected_sensors),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f"{pretty_name(col)} ({col})" for col in selected_sensors],
        )
        for i, col in enumerate(selected_sensors, start=1):
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df[col],
                    mode="lines",
                    name=f"{col} raw",
                    line={"color": BRAND["secondary"], "width": 1.2},
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df[col].rolling(window=rolling_window).mean(),
                    mode="lines",
                    name=f"{col} SMA({rolling_window})",
                    line={"color": BRAND["accent"], "width": 2.2},
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
            fig.update_yaxes(title_text=pretty_name(col), row=i, col=1)
        fig.update_layout(title="Compressor Sensors Behavior Analysis")
        style_figure(fig, height=max(380, 260 * len(selected_sensors)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dark red line is raw signal; lighter red line is smoothed trend for easier interpretation.")
        render_observations(
            "Trend Observations",
            [
                f"Trend view covers rows {row_start} to {row_end} within current time filter.",
                f"{len(selected_sensors)} sensor(s) plotted with SMA window {rolling_window}.",
                "Use row range to zoom on operational events without changing global time filter.",
            ],
        )

with tab_ops:
    st.markdown("### Operations Intelligence")
    left, right = st.columns(2)

    with left:
        st.subheader("Average Oil Temperature by Compressor State")
        if {"comp", "oil_temperature"}.issubset(df_filtered.columns):
            comp_temp = (
                df_filtered.groupby("comp", dropna=False)["oil_temperature"]
                .mean()
                .reset_index(name="avg_oil_temperature")
            )
            fig_comp = px.bar(
                comp_temp,
                x="comp",
                y="avg_oil_temperature",
                text_auto=".2f",
                color="avg_oil_temperature",
                color_continuous_scale=UNIFIED_SEQUENTIAL_SCALE,
                labels={
                    "comp": "Compressor State",
                    "avg_oil_temperature": "Average Oil Temperature",
                },
                title="Average Oil Temperature by Compressor State",
            )
            fig_comp.update_layout(xaxis_title="Compressor State", yaxis_title="Average Oil Temperature")
            style_figure(fig_comp, height=430)
            st.plotly_chart(fig_comp, width="stretch")
            st.caption("Compares thermal behavior across compressor operating states.")
        else:
            st.info("Columns `comp` and `oil_temperature` are needed for this chart.")

    with right:
        st.subheader("Average State Duration")
        if "comp" in df_filtered.columns:
            state_change = df_filtered["comp"].ne(df_filtered["comp"].shift()).cumsum()
            durations = (
                df_filtered.assign(state_change=state_change)
                .groupby(["state_change", "comp"], dropna=False)
                .size()
                .reset_index(name="duration_steps")
            )
            avg_duration = durations.groupby("comp", dropna=False)["duration_steps"].mean().reset_index()
            avg_duration["avg_duration_seconds"] = (
                avg_duration["duration_steps"] * filtered_median_step_seconds
            )
            fig_duration = px.bar(
                avg_duration,
                x="comp",
                y="avg_duration_seconds",
                text_auto=".1f",
                color="avg_duration_seconds",
                color_continuous_scale=UNIFIED_SEQUENTIAL_SCALE,
                labels={
                    "comp": "Compressor State",
                    "avg_duration_seconds": "Average Duration (seconds)",
                },
                title="Average State Duration by Compressor State",
            )
            fig_duration.update_layout(
                xaxis_title="Compressor State",
                yaxis_title="Average Duration (seconds)",
            )
            style_figure(fig_duration, height=430)
            st.plotly_chart(fig_duration, width="stretch")
            st.caption(
                "Duration is estimated from state segment length multiplied by median sampling interval."
            )
        else:
            st.info("Column `comp` is needed for duration analysis.")
    if "comp" in df_filtered.columns and {"comp", "oil_temperature"}.issubset(df_filtered.columns):
        hottest = df_filtered.groupby("comp")["oil_temperature"].mean().sort_values(ascending=False)
        longest = None
        if "avg_duration" in locals():
            longest = avg_duration.sort_values("avg_duration_seconds", ascending=False)["comp"].iloc[0]
        render_observations(
            "Operations Observations",
            [
                f"Hottest average state is `{hottest.index[0]}` at {hottest.iloc[0]:.2f} oil temperature.",
                f"Longest average state duration is `{longest}`." if longest is not None else "State duration summary available in the right chart.",
                "Use these two charts together to identify high-temperature and long-duration operating modes.",
            ],
        )

with tab_explorer:
    st.markdown("### Interactive Metric Explorer")
    st.subheader("Metric Explorer")
    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available in current filters.")
    else:
        default_metric = "oil_temperature" if "oil_temperature" in numeric_cols else numeric_cols[0]
        metric_col = st.selectbox(
            "Select metric",
            options=numeric_cols,
            index=numeric_cols.index(default_metric),
            format_func=lambda c: f"{pretty_name(c)} ({c})",
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean", f"{df_filtered[metric_col].mean():.3f}")
        m2.metric("Std", f"{df_filtered[metric_col].std():.3f}")
        m3.metric("Min", f"{df_filtered[metric_col].min():.3f}")
        m4.metric("Max", f"{df_filtered[metric_col].max():.3f}")
        st.caption(column_description(metric_col))

        left, right = st.columns(2, gap="medium")
        with left:
            fig_hist = px.histogram(
                df_filtered,
                x=metric_col,
                nbins=40,
                title=f"Distribution - {pretty_name(metric_col)}",
                color_discrete_sequence=[BRAND["secondary"]],
                labels={metric_col: pretty_name(metric_col), "count": "Frequency"},
            )
            fig_hist.update_layout(xaxis_title=pretty_name(metric_col), yaxis_title="Frequency")
            style_figure(fig_hist, height=380)
            st.plotly_chart(fig_hist, width="stretch")
        with right:
            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered[metric_col],
                    mode="lines",
                    name=f"{metric_col} raw",
                    line={"color": BRAND["secondary"], "width": 1.3},
                )
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered[metric_col].rolling(window=rolling_window).mean(),
                    mode="lines",
                    name=f"{metric_col} SMA({rolling_window})",
                    line={"color": BRAND["accent"], "width": 2.2},
                )
            )
            fig_ts.update_layout(title=f"Timeline - {pretty_name(metric_col)}")
            fig_ts.update_yaxes(title=pretty_name(metric_col))
            style_figure(fig_ts, height=380)
            st.plotly_chart(fig_ts, width="stretch")
        st.caption("Use this tab to define, summarize, and investigate any selected metric quickly.")
        q10 = df_filtered[metric_col].quantile(0.10)
        q90 = df_filtered[metric_col].quantile(0.90)
        render_observations(
            "Metric Observations",
            [
                f"Middle 80% expected range for `{metric_col}` is {q10:.3f} to {q90:.3f}.",
                f"Current spread: min {df_filtered[metric_col].min():.3f}, max {df_filtered[metric_col].max():.3f}.",
                "Use this page for fast definition, distribution check, and timeline behavior of any metric.",
            ],
        )
