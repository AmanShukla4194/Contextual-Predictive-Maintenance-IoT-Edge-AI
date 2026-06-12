# Contextual Predictive Maintenance — IoT Edge AI

A machine learning system for predicting industrial equipment failures using multi-source IoT sensor fusion.

## Overview

This project develops a predictive maintenance pipeline on the Microsoft Azure PdM dataset, which contains telemetry readings, error logs, maintenance records, and machine metadata for 100 machines over one year. The system fuses four data streams into a unified feature space and trains a gradient boosting classifier to predict component failures up to 24 hours in advance.

## Dataset

Microsoft Azure Predictive Maintenance dataset (5 tables):

| File | Description |
|------|-------------|
| `PdM_telemetry.csv` | Hourly sensor readings — volt, rotate, pressure, vibration |
| `PdM_errors.csv` | Error event logs per machine |
| `PdM_failures.csv` | Component failure records (comp1–comp4) |
| `PdM_maint.csv` | Scheduled maintenance history |
| `PdM_machines.csv` | Machine model type and age |

## Project Structure

```
├── notebooks/          # EDA, feature exploration, and model experiments
├── src/
│   ├── data/           # Data loading and preprocessing modules
│   ├── features/       # Rolling window and contextual feature engineering
│   ├── models/         # Training, evaluation, and SHAP explainability
│   └── utils/          # Shared helper functions
├── app/                # Streamlit dashboard for live prediction demo
├── reports/            # Evaluation outputs, plots, and metric summaries
└── requirements.txt
```

## Technical Approach

- **Multi-source fusion** — telemetry + errors + maintenance + machine specs merged on machineID and datetime
- **Rolling features** — 3h, 24h, 72h window statistics (mean, std, variance) per sensor signal
- **Contextual features** — time since last maintenance, cumulative error counts, machine age
- **Class imbalance** — SMOTE applied within cross-validation folds to prevent leakage
- **Model** — LightGBM optimized for macro F1 score across four failure components
- **Explainability** — SHAP values for per-prediction feature attribution
- **Dashboard** — Streamlit interface for end-to-end demo and threshold simulation

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset files inside a local `archive/` folder (not tracked by git).

## Branch Structure

| Branch | Member | Focus Area |
|--------|--------|------------|
| `aman-shukla` | Aman Shukla *(Team Lead)* | Core ML pipeline architecture, model design, cross-validation strategy, system integration, and final optimization |
| `nakshatra` | Nakshatra | Feature engineering — rolling window statistics and contextual feature fusion |
| `gokul` | Gokul | Data ingestion, preprocessing, EDA, and multi-source dataset merging |
| `vrithik` | Vrithik | SHAP explainability, Streamlit dashboard, evaluation reports, and presentation |
