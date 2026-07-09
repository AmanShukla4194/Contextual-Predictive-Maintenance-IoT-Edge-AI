# Contextual Predictive Maintenance — IoT Edge AI
### Mid-Review Reference Document | Aman Shukla | Team Leader

---

## 1. What Is This Project?

This project is a machine learning system that predicts equipment failures in industrial machines **before they actually happen**, using real-time sensor data collected from IoT (Internet of Things) devices attached to the machines.

The full name is **Contextual Predictive Maintenance using IoT Edge AI**. The word "Contextual" is important — it means our model does not just look at raw sensor readings at a single point in time. It also considers the **context** around each reading: how the machine has been behaving over the past few hours and days, when it was last serviced, how many errors it has produced recently, and patterns in how sensor values are changing over time.

In simple terms: instead of waiting for a machine to break down and then fixing it (reactive maintenance), our system learns from historical patterns to **predict which machine is likely to fail, which component will fail, and how soon** — so engineers can act before the breakdown happens.

---

## 2. The Problem We Are Solving

### The Real-World Problem

In manufacturing plants, large machines run continuously. When a machine breaks down unexpectedly:
- Production halts, causing massive financial losses
- Emergency repairs are expensive and disruptive
- Spare parts may not be available immediately
- Safety risks increase

Traditional maintenance is either:
- **Reactive** — fix it after it breaks (too late, costly)
- **Scheduled/Preventive** — service every X days regardless of actual condition (wasteful, still misses unexpected failures)

### What We Want

A system that reads sensor data from machines in real time and predicts: **"Machine 42 is likely to suffer a Component 3 failure within the next 24 hours."**

This allows maintenance teams to schedule repairs proactively, reducing downtime and costs.

### Why This Is a Hard ML Problem

- The dataset is highly **imbalanced** — most time periods have no failure (the majority class), and failures are rare events. A naive model would just predict "no failure" every time and still achieve 95%+ accuracy, which is useless.
- The sensor readings have **temporal dependencies** — what happened 3 hours ago affects what happens now.
- There are **multiple failure types** (4 components), making this a multi-class classification problem.
- We need the model to be **explainable** so engineers can trust and act on its predictions.

---

## 3. The Dataset

### Microsoft Azure Predictive Maintenance Dataset

We are using the publicly available **Microsoft Azure Predictive Maintenance dataset**, one of the most well-known benchmark datasets in the industrial IoT and predictive maintenance space. It was chosen because:

1. It closely mirrors real industrial scenarios with realistic sensor noise and failure patterns
2. It contains multiple data sources (sensors, errors, maintenance logs, machine specs) — allowing us to build rich contextual features
3. It is large enough (over 876,000 hourly telemetry records) to train a robust model
4. It is a standard benchmark, meaning our results can be compared against published research

### Dataset Files (5 CSV Files)

| File | Description | Size |
|---|---|---|
| `PdM_telemetry.csv` | Hourly sensor readings (voltage, rotation, pressure, vibration) from 100 machines over 1 year | ~876,000 rows |
| `PdM_failures.csv` | Logged failure events with component labels (comp1–comp4) | ~761 rows |
| `PdM_errors.csv` | Non-failure error events (error1–error5) logged per machine | ~3,919 rows |
| `PdM_maint.csv` | Maintenance/replacement records per machine per component | ~3,286 rows |
| `PdM_machines.csv` | Static machine metadata: model type and age | 100 rows |

### Key Sensor Columns (Telemetry)

- **volt** — voltage reading (electrical health)
- **rotate** — rotation speed (mechanical health)
- **pressure** — pressure reading (pneumatic/hydraulic health)
- **vibration** — vibration level (structural health)

All four sensors are recorded every hour for each of the 100 machines.

### Target Variable

We label each hourly row: **will this machine experience a component failure within the next 24 hours?**

Labels:
- `none` — no failure expected
- `comp1` — Component 1 failure predicted
- `comp2` — Component 2 failure predicted
- `comp3` — Component 3 failure predicted
- `comp4` — Component 4 failure predicted

This is a 5-class classification problem with severe class imbalance (failures are ~3–5% of total records).

---

## 4. The AI/ML Approach

### Model: LightGBM Classifier

We use **LightGBM** (Light Gradient Boosting Machine), a state-of-the-art gradient boosting framework developed by Microsoft. It was chosen because:
- Handles tabular sensor data extremely well
- Trains fast even on large datasets
- Handles class imbalance natively via `class_weight="balanced"`
- Provides built-in feature importance
- Works well out of the box without heavy tuning

### Handling Class Imbalance: SMOTE

Since failures are rare, the model would be biased toward predicting "no failure" if we trained naively. We use **SMOTE** (Synthetic Minority Oversampling Technique) to synthetically generate new samples of minority failure classes during training. Critically, SMOTE is applied **inside each cross-validation fold**, not on the entire dataset — this prevents data leakage where synthetic samples from the test fold contaminate the training fold.

### Evaluation Metric: Macro F1 Score

Accuracy is a misleading metric for imbalanced data. We use **Macro F1 Score**, which computes F1 for each class independently and then averages them. This penalises the model equally for failing on rare failure classes as it does for the majority "none" class.

### Feature Engineering Strategy

Raw sensor readings alone are not very predictive. We engineer rich contextual features:

| Feature Type | Description | Example |
|---|---|---|
| Rolling Mean | Average sensor value over a time window | `volt_mean_3h`, `volt_mean_24h` |
| Rolling Std | Variability of sensor over a time window | `pressure_std_72h` |
| Lag Features | Sensor value from N hours ago | `rotate_lag_1h`, `volt_lag_6h` |
| Diff Features | Change in sensor value from N hours ago | `vibration_diff_3h` |
| Time Since Maintenance | Hours elapsed since last component service | `hours_since_maintenance` |
| Cumulative Error Count | Total errors logged up to this point | `cumulative_error_count` |
| Time Features | Hour of day, day of week, month | `hour_of_day`, `is_weekend` |
| Sensor Interactions | Ratios and deviations between sensors | `volt_div_rotate`, `pressure_dev` |

Rolling windows used: **3 hours, 24 hours, 72 hours**

### Model Explainability: SHAP

We integrate **SHAP** (SHapley Additive exPlanations) to explain individual predictions — showing which features drove a specific prediction. This is critical for adoption in real industrial settings where engineers need to trust the system.

---

## 5. Project Architecture

```
Contextual-Predictive-Maintenance-IoT-Edge-AI/
│
├── archive/                    # Raw dataset CSV files (not committed)
├── src/
│   ├── data/                   # Data loading, merging, cleaning, labelling
│   ├── features/               # All feature engineering modules
│   ├── models/                 # Training, prediction, evaluation, tuning
│   └── utils/                  # Shared utilities
├── notebooks/                  # EDA and analysis scripts
├── app/                        # Streamlit dashboard
│   └── pages/                  # Modular dashboard pages
├── reports/                    # Evaluation outputs
├── models/                     # Saved model files (not committed)
├── config.py                   # Global configuration constants
├── requirements.txt            # Python dependencies
└── README.md                   # Repository overview
```

---

## 6. Team Structure and Branch Strategy

| Branch | Member | Role |
|---|---|---|
| `aman-shukla` | Aman Shukla (Team Leader) | Core ML pipeline — training, inference, evaluation |
| `gokul` | Gokul | Data pipeline, EDA notebooks |
| `nakshatra` | Nakshatra | Feature engineering modules |
| `vrithik` | Vrithik | Streamlit dashboard, visualisation utilities |

The `main` branch is protected. No one pushes directly to main. Each member works on their own branch and all code is reviewed before merging.

---

## 7. Aman Shukla's Work — Full Code Breakdown

As team leader, I designed the overall system architecture and personally built the core ML pipeline. Below is a detailed explanation of every file I have written.

---

### `src/data/loader.py`

**What it does:** The entry point for all data. Reads all five CSV files from the `archive/` folder and returns them as a dictionary of pandas DataFrames. Also provides individual loader functions (`load_telemetry()`, `load_failures()`, etc.) for granular access.

**Why it matters:** Every other module depends on this. A single `load_all()` call gives any team member access to all data with consistent column naming and datetime parsing.

**Key function:** `load_all()` — returns `{"telemetry": df, "failures": df, "errors": df, "maintenance": df, "machines": df}`

---

### `src/data/preprocessor.py`

**What it does:** Two core preprocessing steps:
1. `label_failures()` — for each row in the telemetry data, looks 24 hours into the future and checks if a failure occurs. If yes, labels the row with the failure component (e.g., `comp1`). If no failure is coming, labels it `none`.
2. `merge_machine_info()` — joins the static machine metadata (age, model type) onto the telemetry data and one-hot encodes the categorical `model` column (model1, model2, model3, model4).

**Why it matters:** This is where we create the target variable. The 24-hour prediction horizon is a deliberate design choice — it gives engineers enough lead time to act.

---

### `src/utils/helpers.py`

**What it does:** Shared utility functions used across the pipeline:
- `encode_labels()` — converts string labels like `comp1`, `none` into integer class codes (0, 1, 2, 3, 4). Also returns the reverse mapping so predictions can be decoded back to readable names.
- `split_by_time()` — splits the dataset chronologically. The first 80% of time is training data, the last 20% is the test set. This is the correct way to split time series data — random splitting would leak future information into training.
- `get_feature_cols()` — automatically identifies all numeric feature columns, excluding metadata columns like `machineID`, `datetime`, and label columns.

---

### `src/features/feature_pipeline.py`

**What it does:** Orchestrates all feature engineering in one place. Calls rolling features, context features, lag features, diff features, and time features in the correct order and returns a fully enriched DataFrame ready for model training.

**Why it matters:** Before this module, individual feature functions existed separately. This orchestrator ensures consistent feature construction whether we are training or predicting.

**Key function:** `build_features(telemetry, maintenance, errors)` — single call produces all features.

---

### `src/models/trainer.py`

**What it does:** The core model training logic:
- `train_lgbm()` — trains a LightGBM classifier on provided training data and returns the fitted model.
- `cross_validate()` — runs stratified K-fold cross-validation (5 folds). Inside each fold, SMOTE is applied only to the training portion of that fold. Computes mean and standard deviation of macro F1 across folds. This gives an honest estimate of generalisation performance.

**Why SMOTE inside folds matters:** If you apply SMOTE before splitting into folds, synthetic samples derived from the test fold's real samples appear in the training fold — this is data leakage and gives falsely optimistic CV scores. Doing SMOTE inside each fold is the correct approach.

---

### `src/models/train_pipeline.py`

**What it does:** The end-to-end training script. Chains together: load data → preprocess → build features → encode labels → temporal split → cross-validate → train final model on full training set → save model to disk as a `.pkl` file (using joblib).

**Key function:** `run_training()` — one call trains and saves a production-ready model.

**Saved artefact:** `models/lgbm_pdm.pkl` — contains the model object, the list of feature columns used, and the label map.

---

### `src/models/predictor.py`

**What it does:** Handles inference (making predictions with a saved model):
- `load_model()` — deserialises the saved `.pkl` file
- `predict()` — batch prediction returning integer class codes
- `predict_proba()` — returns probability scores for each class (used for confidence display in the dashboard)
- `predict_single()` — takes a single dictionary of sensor values and returns a prediction, designed for the dashboard's input form

---

### `src/models/evaluate_pipeline.py`

**What it does:** Loads the saved model, runs it on the held-out test set, computes the macro F1 score and full classification report (precision, recall, F1 per class), and writes the results to `reports/evaluation_results.txt`.

**Why it's separate from training:** Evaluation should be reproducible and runnable independently of training. If we retrain with different features, we can re-evaluate without changing this script.

---

### `src/models/hyperparameter_tuner.py`

**What it does:** Defines a grid of LightGBM hyperparameter combinations (varying `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`) and evaluates each using stratified K-fold cross-validation with macro F1 scoring. Returns the best configuration.

**Design choice:** We use a manual grid rather than automated search (like Optuna) to keep it transparent and reproducible for the internship context, while still demonstrating proper tuning methodology.

---

### `src/models/model_registry.py`

**What it does:** Manages multiple saved model versions:
- `save_model_version()` — saves a model with a version number and records its macro F1 score in a JSON metadata file
- `load_model_version()` — loads a specific version by number
- `list_versions()` — lists all saved versions with their scores, useful for comparing experiments

**Why it matters:** During development, we run multiple experiments with different feature sets and hyperparameters. The registry tracks which version performed best.

---

### `src/data/sampler.py`

**What it does:**
- `stratified_sample()` — draws a representative fraction of data maintaining class proportions, useful for fast development and debugging
- `temporal_train_test_split()` — an alternative clean split function that uses a configurable test fraction and sorts by datetime before splitting
- `class_counts()` — prints class distribution with percentages, useful for sanity-checking label balance before training

---

### `src/models/threshold_tuner.py`

**What it does:** After a model produces probability scores, we normally classify by taking `argmax` (the class with the highest probability). But for imbalanced problems, adjusting the **decision threshold** can improve macro F1. This module:
- `tune_threshold()` — sweeps thresholds from 0.1 to 0.9 and finds the one that maximises macro F1 on a validation set
- `apply_threshold()` — applies a chosen threshold to convert probabilities to class predictions

**Why it matters:** This is a post-processing optimisation step that can squeeze additional performance without retraining the model.

---

## 8. Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Cross-validation, metrics, preprocessing |
| LightGBM | Gradient boosting classifier |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| Plotly, Matplotlib, Seaborn | Visualisation |
| joblib | Model serialisation |
| Git + GitHub | Version control, branching |

---

## 9. Current Project Status (Day 12 of 23)

| Area | Status |
|---|---|
| Data loading and preprocessing | Complete |
| Feature engineering (rolling, lag, context, time, interaction) | Complete |
| LightGBM training with SMOTE-in-fold CV | Complete |
| Hyperparameter tuning | Complete |
| Model evaluation and reporting | Complete |
| Model versioning registry | Complete |
| EDA notebooks (8 notebooks) | Complete |
| Streamlit dashboard pages (Overview, Sensor Trends, Predictions, Explainability) | In progress |
| SHAP integration in dashboard | Pending |
| Final project report | Pending |

---

## 10. Key Talking Points for Mentor Review

1. **Why LightGBM?** — Fast, handles imbalance, industry-standard for tabular sensor data, interpretable via feature importance.

2. **Why SMOTE inside CV folds?** — To prevent data leakage. Applying SMOTE before folding contaminates the test fold with synthetic data derived from real test samples, giving falsely high CV scores.

3. **Why Macro F1 and not Accuracy?** — With 95% "no failure" samples, a model predicting "no failure" always would get 95% accuracy but be completely useless. Macro F1 equally weighs performance across all 5 classes.

4. **Why temporal split and not random split?** — Sensor data has time dependencies. A random split would allow training on data from the future relative to some test samples, which is unrealistic and causes leakage.

5. **Why 24-hour prediction horizon?** — It gives engineers enough lead time to schedule a maintenance visit. Too short (1h) and it's not actionable. Too long (7 days) and the signal is too weak.

6. **Why contextual features?** — A single sensor reading tells us the current state. Rolling averages and standard deviations tell us if the machine is trending toward failure. Lag features capture momentum. Time since maintenance captures wear. Together, they give the model a rich picture of machine health.
