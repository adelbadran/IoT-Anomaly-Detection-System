# IoT Anomaly Detection for Industrial Compressors

Production-grade IoT analytics project for anomaly detection in compressor sensor streams.

This repository combines:
- a structured analysis notebook for anomaly-focused data investigation, and
- a Streamlit dashboard for operational monitoring, anomaly exploration, and deployment use.

## Dataset Link

- [UCI MetroPT-3 Dataset (Kaggle)](https://www.kaggle.com/datasets/pattinson9999/uci-metropt-3-dataset)

## Table of Contents
1. [Project Summary](#project-summary)
2. [Use Case](#use-case)
3. [Repository Structure](#repository-structure)
4. [Notebook Methodology](#notebook-methodology)
5. [Dashboard Capabilities](#dashboard-capabilities)
6. [Data Requirements](#data-requirements)
7. [Setup and Run](#setup-and-run)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

## Project Summary

### Objective
Detect and investigate abnormal behavior in IoT telemetry from compressor equipment by:
- validating data reliability,
- engineering monitoring features,
- analyzing temporal and sensor relationships,
- providing an interactive analytics layer for anomaly triage.

### Current Phase Scope
This phase focuses on anomaly-detection readiness:
- data integrity and continuity checks,
- time-gap analysis,
- sensor trend diagnostics,
- correlation analysis,
- operating-state behavior comparison.

## Use Case

Industrial IoT systems generate high-frequency sensor data where anomalies may indicate:
- equipment degradation,
- unstable operating states,
- data acquisition issues,
- impending failures.

This project helps operations and analytics teams identify those patterns early.

## Repository Structure

```text
Phase 1 - Analysis/
|-- Dashboard/
|   `-- app.py                   # Streamlit monitoring and anomaly exploration dashboard
|-- NoteBook/
|   `-- Phase 1 - Analysis.ipynb # IoT anomaly analysis notebook
|-- requirements.txt             # Dependencies
`-- README.md                    # Project documentation
```

## Notebook Methodology

Notebook: `NoteBook/Phase 1 - Analysis.ipynb`

### 1) Data Loading
- Load IoT sensor dataset from CSV.
- Inspect sample records and data schema.

### 2) Data Integrity Checks
- Detect duplicate records.
- Profile missing values.
- Review statistical baselines.

### 3) Data Cleaning
- Remove non-informative columns.
- Standardize feature names.
- Parse and index timestamps for time-series analysis.

### 4) Feature Engineering
- Temporal features: `hour`, `day_of_week`
- Pressure and dynamic features: `pressure_delta`, `pressure_change`
- Thermal and load features: `oil_temp_rolling`, `power_indicator`
- Normalized behavior feature: `current_per_pressure`

### 5) Anomaly-Focused Analysis
- Time-gap distribution and largest continuity gaps.
- Correlation heatmaps to detect unusual sensor coupling.
- Raw vs smoothed sensor trends to reveal drifts/spikes.
- Compressor state analytics (`comp`) for behavior segmentation.

## Dashboard Capabilities

Dashboard file: `Dashboard/app.py`

### Executive Monitoring
- KPI cards for volume, span, duplicates, and missing values.
- Auto-generated observations for quick incident context.

### Interactive Controls
- Time-range filtering.
- Compressor state filtering (`comp`).
- Trend row-range zoom across filtered rows.
- Smoothing window control.

### Analytical Views
1. `Overview`
   - Data quality profile
   - Time-gap histogram
   - Top continuity gaps
   - Data dictionary
2. `Correlations`
   - Full numeric correlation matrix
   - Core mechanical sensor matrix
3. `Sensor Trends`
   - Multi-sensor time-series (raw + rolling trend)
4. `Compressor Operations`
   - Average oil temperature by state
   - Average state duration by state
5. `Metric Explorer`
   - Per-feature definition, stats, distribution, and timeline

### Design and UX
- Professional card-based layout.
- Streamlit-themed visual styling.
- Right-side legends for easier interaction.
- Observation blocks for decision support.

## Data Requirements

### Required
- `timestamp`

### Recommended for full anomaly workflow
- `tp2`, `tp3`, `h1`, `oil_temperature`, `motor_current`, `reservoirs`, `comp`

The dashboard degrades gracefully when optional columns are unavailable.

## Setup and Run

### 1) Install dependencies
```powershell
cd "Phase 1 - Analysis"
pip install -r requirements.txt
```

### 2) Run Streamlit app
```powershell
streamlit run "Dashboard\app.py"
```

Or:
```powershell
python -m streamlit run "Dashboard\app.py"
```

### 3) Open app
- `http://localhost:8501`

## Deployment

### Recommended input options
1. Upload CSV through the dashboard UI.
2. Set environment variable `METRO_CSV_PATH` for default data loading.

### Environment variable example
```powershell
$env:METRO_CSV_PATH="D:\path\to\Metro.csv"
```

## Troubleshooting

### `ModuleNotFoundError`
Install dependencies with:
```powershell
pip install -r requirements.txt
```

### `missing ScriptRunContext`
Run via Streamlit, not direct Python script:
```powershell
streamlit run "Dashboard\app.py"
```

### Empty dashboard after filtering
- Expand time range.
- Recheck selected `comp` values.

### Timestamp parsing error
Ensure `timestamp` values are valid datetime strings.

## Next Steps

- Add explicit anomaly scoring (rule-based and model-based).
- Add threshold alerting and event flags per sensor.
- Add anomaly timeline export for incident reports.
- Add model evaluation section (precision/recall, false positives).
- Add live IoT stream integration for real-time detection.
