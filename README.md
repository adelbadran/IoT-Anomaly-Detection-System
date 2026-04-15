# IoT Anomaly Detection System

Brief project for detecting and exploring anomalies in industrial compressor IoT sensor data.

## Project Overview

This repository is organized into two phases:

- **Phase 0 - About Dataset**: dataset understanding, schema check, and data quality orientation.
- **Phase 1 - Analysis**: exploratory analysis notebook and Streamlit dashboard for anomaly investigation.

## Dataset Link

- [UCI MetroPT-3 Dataset (Kaggle)](https://www.kaggle.com/datasets/pattinson9999/uci-metropt-3-dataset)

## Repository Structure

```text
IoT-Anomaly-Detection-System/
|-- Phase 0 - About Dataset/
|   `-- Dataset/
|       `-- Metro.csv
|-- Phase 1 - Analysis/
|   |-- Dashboard/
|   |   `-- app.py
|   |-- NoteBook/
|   |   `-- Phase 1 - Analysis.ipynb
|   |-- requirements.txt
|   `-- README.md
`-- README.md
```

## Phase 0 (Dataset)

- Input file: `Phase 0 - About Dataset/Dataset/Metro.csv`
- Purpose:
  - understand available sensors and timestamps
  - check basic quality issues (missing values, duplicates, continuity)
  - prepare clean context for analysis

## Phase 1 (Analysis + Dashboard)

- Notebook: `Phase 1 - Analysis/NoteBook/Phase 1 - Analysis.ipynb`
- Dashboard: `Phase 1 - Analysis/Dashboard/app.py`
- Purpose:
  - analyze sensor behavior and temporal patterns
  - inspect correlations and operating-state behavior
  - support interactive anomaly exploration via Streamlit

## Quick Run (Phase 1 Dashboard)

```powershell
cd "Phase 1 - Analysis"
pip install -r requirements.txt
streamlit run "Dashboard\app.py"
```

Open `http://localhost:8501`.
