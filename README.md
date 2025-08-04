# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## 📍 Project Overview
This project addresses the critical challenge of fraud detection using two datasets:
- **E-commerce fraud data**
- **Credit card transaction data**

Our goal is to build reliable machine learning models to detect fraud, apply explainability, and interpret the key drivers behind fraudulent activity.

---

## ✅ Task 1: Data Cleaning, Preprocessing & Feature Engineering

### ✅ Activities Completed:
- Loaded all datasets and confirmed absence of missing values.
- Converted time columns to `datetime64` format.
- Removed 1,081 duplicates from credit card data.
- Converted `ip_address` to integer and merged with geolocation data.
- Performed univariate and bivariate analysis:
  - Class imbalance observed in both datasets.
- Created key engineered features:
  - `time_since_signup`, `hour_of_day`, `day_of_week`, transaction count per user and device.
- Used **SMOTE** to oversample minority class.

### 🔎 Visuals:
- Class distribution plots for e-commerce and credit card datasets.

---

## ✅ Task 2: Model Training & Evaluation

### ✅ Modeling Steps:
- Encoded categorical variables using OneHotEncoding.
- Used SMOTE for handling imbalance.
- Split data into train/test using stratified sampling.

### ✅ Models Trained:
1. **Logistic Regression** – Baseline model
2. **XGBoost Classifier** – Chosen ensemble model

### 📊 Evaluation Metrics:
- **Precision-Recall Curve (AUC-PR)**
- **F1-Score**
- **Confusion Matrix**

### 🎯 Results:
| Dataset       | Model                | F1 Score | AUC-PR |
|---------------|----------------------|----------|--------|
| E-commerce    | Logistic Regression  | 0.61     | 0.67   |
| E-commerce    | XGBoost              | 0.69     | 0.71   |
| Credit Card   | Logistic Regression  | 0.09     | 0.72   |
| Credit Card   | XGBoost              | 0.77     | 0.78   |

📌 **Best Model**: XGBoost (both datasets)

---

## ✅ Task 3: Model Explainability (SHAP)

### 🧠 SHAP Visualizations:
- **Summary Plot**: Global feature importance.
- **Force Plot**: Local explanation for a single prediction.

### 🔍 Key Drivers Identified:
- `transaction_count`, `hour_of_day`, and `device_transaction_count` were among top influential features in predicting fraud.

---

## 🗂 Project Structure

├── data/
│ ├── Fraud_Data.csv
│ ├── creditcard.csv
│ └── IpAddress_to_Country.csv
├── src/
│ ├── preprocessing.py
│ ├── features.py
│ ├── models.py
│ └── shap_explainer.py
├── notebooks/
│ ├── EDA_and_Preprocessing.ipynb
│ ├── Model_Training.ipynb
│ └── SHAP_Explainability.ipynb
├── tests/
│ ├── test_preprocessing.py
│ ├── test_features.py
│ └── test_shap_explainer.py
├── scripts/
│ └── run_training.py
└── .github/
└── workflows/
└── main.yml

---

## ✅ Tech Stack

- Python, Pandas, Scikit-learn, XGBoost, SHAP
- Imbalanced-learn (SMOTE)
- Matplotlib for visualizations
- Modular OOP Codebase for maintainability

---

## ✅ Authors
Hawi T.  
Submission Date: August 4, 2025