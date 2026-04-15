# IoT Anomaly Detection System

Brief project for detecting and exploring anomalies in industrial compressor IoT sensor data.

## Project Overview

This repository currently documents **Phase 1 - Analysis**: exploratory analysis notebook and Streamlit dashboard for anomaly investigation.

## Dataset Link

- [UCI MetroPT-3 Dataset (Kaggle)](https://www.kaggle.com/datasets/pattinson9999/uci-metropt-3-dataset)

## Repository Structure

```text
IoT-Anomaly-Detection-System/
|-- Phase 1 - Analysis/
|   |-- .streamlit/
|   |   `-- config.toml
|   |-- Dashboard/
|   |   `-- app.py
|   |-- NoteBook/
|   |   `-- Phase 1 - Analysis.ipynb
|   |-- requirements.txt
|   `-- README.md
`-- README.md
```

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

Upload note: `.streamlit/config.toml` sets `maxUploadSize = 300`, so files larger than 200 MB (for example `Metro.csv` at ~208 MB) can be uploaded.
