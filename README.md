# Contextual Predictive Maintenance — IoT Edge AI

A complete machine learning project for predicting industrial equipment failures before they happen using IoT sensor data, contextual features, and an interactive Streamlit dashboard.

## Overview

This project builds a predictive maintenance system on the Microsoft Azure Predictive Maintenance dataset. It combines telemetry, error logs, maintenance history, and machine metadata to predict whether a machine is likely to fail in the next 24 hours and which component is most likely to fail.

The goal is to support proactive maintenance rather than reactive repair by giving engineers early warnings and explainable predictions.

## Why This Project Matters

Unexpected machine failures can cause:

- production downtime
- costly emergency repairs
- safety risks
- lost revenue and delayed operations

Instead of waiting for a failure to happen, this system helps identify risky machine states early so maintenance can be scheduled in advance.

## Project Goals

- predict failure events before they occur
- identify the likely failing component
- create meaningful contextual features from sensor and maintenance history
- build an explainable ML workflow
- provide a user-friendly dashboard for monitoring and predictions

## Dataset

The project uses the Microsoft Azure Predictive Maintenance dataset with five data sources:

| File              | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| PdM_telemetry.csv | Hourly telemetry readings for voltage, rotation, pressure, and vibration |
| PdM_errors.csv    | Logged error events for each machine                                     |
| PdM_failures.csv  | Failure records with component labels such as comp1 to comp4             |
| PdM_maint.csv     | Maintenance and replacement history                                      |
| PdM_machines.csv  | Static machine metadata such as model type and age                       |

## Core Machine Learning Approach

### Model

The project uses LightGBM, a gradient boosting classifier well suited for tabular industrial data.

### Feature Engineering

The pipeline creates rich contextual features such as:

- rolling statistics over 3h, 24h, and 72h windows
- lag features from previous hours
- differences between current and past sensor values
- time-based features such as hour, day, and week patterns
- maintenance-related context such as hours since last service
- cumulative error count and other operational context

### Class Imbalance Handling

Because failures are rare, the system uses SMOTE inside cross-validation folds to avoid data leakage and improve learning on minority classes.

### Evaluation Metric

The model is evaluated with macro F1 score rather than simple accuracy because the dataset is highly imbalanced.

## Project Architecture

```text
Contextual-Predictive-Maintenance-IoT-Edge-AI/
├── archive/                # Raw dataset CSV files
├── app/                    # Streamlit dashboard application
│   ├── pages/              # Dashboard pages for charts and analysis
│   └── utils/              # Dashboard support utilities
├── notebooks/              # Exploratory data analysis and experiments
├── reports/                # Evaluation outputs and analysis reports
├── src/
│   ├── data/               # Data loading, preprocessing, and merging
│   ├── features/           # Feature engineering modules
│   ├── models/             # Training, evaluation, prediction, and explainability
│   └── utils/              # Shared helper utilities
├── config.py               # Project-level configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Dashboard Features

The Streamlit dashboard includes pages for:

- Overview
- Sensor Trends
- Predictions
- Maintenance Schedule
- Alerts
- Prediction History
- Model Metrics
- SHAP Analysis
- Explainability
- About

## Repository Structure Summary

- Data pipeline: loading, cleaning, merging, labeling, and preprocessing
- Feature engineering: rolling, lag, contextual, temporal, and interaction features
- Modeling: training, tuning, evaluation, model registry, and prediction
- Dashboard: interactive web app for demonstrating predictions and insights
- Notebooks: analysis and experiment notebooks for EDA and model understanding

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd Contextual-Predictive-Maintenance-IoT-Edge-AI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset in the archive folder

Download or copy the required CSV files into the archive directory so the project can load them locally.

## Running the Project

### Run the Streamlit dashboard

From the project root:

```powershell
python -m streamlit run app/dashboard.py
```

If you are using the virtual environment created above, the command should work directly. The app will open on:

```text
http://localhost:8501
```

### Run training pipeline

If you want to train the model locally, use the relevant training script from the src/models package.

## Current Project Status

The project has progressed through the main development stages:

- data loading and preprocessing completed
- feature engineering completed
- model training and evaluation implemented
- hyperparameter tuning included
- model registry and reporting added
- exploratory notebooks created
- Streamlit dashboard implemented and runnable

## Technology Stack

- Python
- pandas and numpy
- scikit-learn
- LightGBM
- imbalanced-learn
- SHAP
- Streamlit
- Plotly, Matplotlib, and Seaborn
- joblib

## Team and Branch Structure

| Branch      | Focus                                                   |
| ----------- | ------------------------------------------------------- |
| aman-shukla | Core ML pipeline, training, evaluation, and integration |
| gokul       | Data pipeline, preprocessing, and EDA                   |
| nakshatra   | Feature engineering                                     |
| vrithik     | Dashboard, explainability, and visualization            |

## Notes

This repository is intended for educational, academic, and demonstration purposes. It showcases a practical end-to-end predictive maintenance workflow using industrial IoT data and explainable machine learning.
