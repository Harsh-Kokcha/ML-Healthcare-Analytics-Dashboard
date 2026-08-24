# ML-Healthcare-Analytics-Dashboard
An end-to-end machine learning pipeline built with Streamlit that predicts the presence of medical conditions from Electronic Health Records (EHR) data. The dashboard covers the full ML lifecycle — from exploratory data analysis and feature engineering to model training, evaluation, temporal shift analysis, continual learning, and feature importance interpretation.

## Project Overview

This dashboard predicts whether a patient has a specific medical condition using structured EHR data. The target is a real clinical diagnosis (e.g., hypertension, diabetes) selected by the user from the top-15 most prevalent conditions in the dataset. The pipeline is designed to be **leakage-free** — financial outcome variables like `HEALTHCARE_EXPENSES` are explicitly excluded from the feature matrix.

A key design choice is a **temporal split at the year 2020**, representing the COVID-19 pandemic onset as a clinically meaningful boundary. This allows the project to study how models generalise across a documented real-world distribution shift.

## Dataset

The project uses a **Synthea-style synthetic EHR dataset** organized into the following CSV files, expected under a `Dataset/` folder in the project root:

|          File          |                           Description                                         |
|------------------------|-------------------------------------------------------------------------------|
| `patients.csv`         | Demographics — age, gender, race, ethnicity, income, insurance coverage       |
| `encounters.csv`       | Clinical visits with cost, type, and timestamps                               |
| `conditions.csv`       | Diagnosed medical conditions per patient                                      |
| `medications.csv`      | Prescribed medications per patient                                            |
| `procedures.csv`       | Medical procedures performed per patient                                      |
| `immunizations.csv`    | Vaccination history                                                           |
|  `observations.csv`    | Numeric clinical vitals and lab results (e.g., BMI, blood pressure, glucose)  |

---

## Feature Engineering

Features are aggregated per patient from multiple tables:

| Feature Group           | Source              | Aggregation            |
|-------------------------|---------------------|------------------------|
| Demographics            | `patients.csv`      | Raw / label-encoded    |
| Encounter stats         | `encounters.csv`    | Count, mean, std, sum  |
| Comorbidity burden      | `conditions.csv`    | Count of diagnoses     |
| Medication load         | `medications.csv`   | Count                  |
| Procedure load          | `procedures.csv`    | Count                  |
| Clinical vitals & labs  | `observations.csv`  | Mean & std per patient |

**Clinical observations included:** Body Height, Body Weight, BMI, Diastolic/Systolic Blood Pressure, Heart Rate, Body Temperature, Respiratory Rate, Glucose, Hemoglobin.

> `HEALTHCARE_EXPENSES` is intentionally **excluded** to prevent data leakage.

---

## Models

Three classifiers are trained and compared:

| Model | Configuration |
|---|---|
| **Decision Tree** | `max_depth=10`, `min_samples_split=10` |
| **SVM (RBF kernel)** | `C=1.0`, `gamma='scale'`, probability output enabled |
| **MLP Neural Network** | Hidden layers: `(100, 50)`, early stopping, `max_iter=500` |

SVM and MLP use `StandardScaler`-normalized features. Decision Tree uses raw features.

---

## ⏰ Temporal Split

The dataset is split at **January 1, 2020** based on each patient's first recorded encounter:

- **Dataset 1 (Historical):** Patients with first encounter before 2020 — used for primary training and evaluation.
- **Dataset 2 (Current):** Patients with first encounter from 2020 onwards — used for temporal shift and continual learning analysis.

**Justification:** The COVID-19 pandemic caused a well-documented global disruption to healthcare utilisation patterns, making 2020 a clinically meaningful and realistic distribution shift boundary.

---

## 🧪 Dashboard Tabs

The Streamlit app is organized into 8 interactive tabs:

| Tab | Description |
|---|---|
| 📊 **EDA Dashboard** | Class distribution, top conditions, age/gender breakdowns, clinical observation stats, correlation heatmaps — separately for Dataset 1 and Dataset 2 |
| 🔧 **Preprocessing** | Target variable info, temporal split justification, feature engineering table, data drift stats (D1 vs D2 descriptive statistics) |
| 🤖 **Model Training** | Trains Decision Tree, SVM, and MLP on Dataset 1 with a live progress bar |
| 📈 **Evaluation Metrics** | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC curves, Classification Report |
| 📉 **Complexity Analysis** | Decision Tree depth vs accuracy, SVM C-parameter tuning, MLP training loss curve |
| ⏰ **Temporal Shift** | Evaluates D1-trained models on D2 test set; compares performance degradation across the 2020 boundary |
| 🧠 **Continual Learning** | Fine-tunes models on Dataset 2 training data and measures accuracy/F1 improvement; MLP uses `warm_start` to preserve learned weights |
| 🎯 **Feature Importance** | Gini importance (Decision Tree) and Permutation Importance (all three models) on the held-out test set |

---

## 🔄 Continual Learning Strategy

| Model | Strategy |
|---|---|
| **MLP** | `warm_start=True` — resumes training from existing weights (no catastrophic forgetting) |
| **Decision Tree** | Retrained on combined D1 + D2 training data |
| **SVM** | Retrained on combined D1 + D2 training data |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run Team09_Assignment2_dashboard.py
```

Make sure the `Dataset/` folder containing all required CSV files is in the same directory as the script.

---

## Dependencies

```
streamlit
pandas
numpy
scikit-learn
plotly
```

Install all at once:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

## Project Structure

```
.
├── Team09_Assignment2_dashboard.py   # Main Streamlit app
├── Dataset/
│   ├── patients.csv
│   ├── encounters.csv
│   ├── conditions.csv
│   ├── medications.csv
│   ├── procedures.csv
│   ├── immunizations.csv
│   └── observations.csv
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **No data leakage:** `HEALTHCARE_EXPENSES` excluded from all feature matrices.
- **Real medical targets:** Condition presence derived from `conditions.csv`, not synthetic labels.
- **Proper test splits:** Temporal shift and continual learning tabs use true held-out test sets, not full datasets.
- **Variance-aware features:** STD aggregations added to encounter cost features to capture patient variability.
- **Model-agnostic importance:** Permutation importance computed for all three models on the same test set for a fair comparison.
