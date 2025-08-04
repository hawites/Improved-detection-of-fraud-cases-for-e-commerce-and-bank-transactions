# 🛡️ Fraud Detection for E-commerce and Bank Transactions

## 📁 Project Overview

This project aims to improve the detection of fraudulent transactions in both e-commerce and banking systems. Using advanced machine learning, enriched features, and explainable AI, we help financial institutions reduce losses while minimizing false positives that harm user experience.

---

## ✅ Task 1: Data Analysis and Preprocessing

### 📂 Objective:
Prepare high-quality, clean datasets that are ready for modeling. This task focused on cleaning, exploratory analysis, feature engineering, and addressing class imbalance — all critical for detecting fraud in both e-commerce and banking transactions.

---

### 🔧 Key Steps Completed:

- **Data Loading & Initial Checks**
  - Loaded `Fraud_Data.csv`, `creditcard.csv`, and `IpAddress_to_Country.csv`
  - Confirmed no missing values in any of the datasets

- **Data Cleaning**
  - Removed 1,081 duplicates from the credit card dataset
  - Converted `signup_time` and `purchase_time` into `datetime` format
  - Transformed float-style `ip_address` into integers for later merging

- **Univariate & Bivariate EDA**
  - Analyzed class distributions to understand fraud imbalance across both datasets
  - [✓ See image: E-commerce fraud class distribution]
  - [✓ See image: Credit card fraud class distribution]

- **Feature Engineering**
  - Extracted time-based features: `hour_of_day`, `day_of_week`, and `time_since_signup`
  - Calculated frequency-based metrics: `transaction_count` and `device_transaction_count`
  - Merged IP data with geolocation dataset to extract user `country`

- **Class Imbalance Analysis**
  - E-commerce: ~9.4% fraud cases
  - Credit card: ~0.17% fraud cases
  - Planned use of **SMOTE** during Task 2 to address imbalance

---

### 🧱 Structure and Code Hygiene

- **Notebook:** `notebooks/01_data_exploration.ipynb`
- **Modules:** `src/preprocessing.py`, `src/features.py`
- **Test Scaffold:** `tests/test_preprocessing.py`
- **CI & Scripts:** `.github/workflows/`, `scripts/`

---

## 🚀 Task 2 - Model Building and Training

In this task, we trained and evaluated machine learning models on both the e-commerce fraud and credit card fraud datasets.

### 📌 Data Preparation
- The target columns were `class` (for e-commerce) and `Class` (for credit card).
- Unnecessary columns such as timestamps, user/device IDs, and IP fields were dropped.
- One-hot encoding was applied to categorical features.
- SMOTE was used to address class imbalance, oversampling only the training data.

### 🔍 Models Trained
- **Logistic Regression**: Chosen for its simplicity and interpretability.
- **XGBoost Classifier**: Selected as a powerful gradient boosting ensemble model.

### 📊 Evaluation Metrics
The used metrics suitable for imbalanced datasets:
- **F1 Score**
- **Confusion Matrix**
- **Precision-Recall AUC (AUC-PR)**

#### 📈 E-Commerce Dataset
| Model               | F1 Score | AUC-PR | Notes |
|--------------------|----------|--------|-------|
| Logistic Regression| 0.61     | 0.67   | Strong precision, weaker recall. |
| XGBoost            | 0.69     | 0.71   | Better balance of precision and recall. |

#### 📈 Credit Card Dataset
| Model               | F1 Score | AUC-PR | Notes |
|--------------------|----------|--------|-------|
| Logistic Regression| 0.09     | 0.72   | High recall but very low precision. |
| XGBoost            | 0.77     | 0.78   | Excellent performance despite class imbalance. |

### ⚖️ Class Imbalance Strategy
Both datasets showed heavy class imbalance (fraud under 1–9%). We applied **SMOTE** (Synthetic Minority Over-sampling Technique) only to the training set to preserve test integrity. This significantly improved recall and F1-score for both models.

### 🧪 Model Summary
- XGBoost consistently outperformed Logistic Regression.
- Visual precision-recall curves confirmed superior AUC-PR for XGBoost in both datasets.

---

## ✅ Reproducibility & Code Hygiene

- All tasks organized in Jupyter notebooks and OOP `.py` modules
- Git branches used per task (`task-1`, `task-2`, etc.)
- Project hygiene established:
  - `.github/workflows/ci.yml` for CI pipelines
  - `scripts/` for automation
  - `tests/` folder scaffolded for unit testing

---


