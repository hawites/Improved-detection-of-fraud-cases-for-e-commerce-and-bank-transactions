# 🛡️ Fraud Detection for E-commerce and Bank Transactions

## 📁 Project Overview

This project aims to improve the detection of fraudulent transactions in both e-commerce and banking systems. Using advanced machine learning, enriched features, and explainable AI, we help financial institutions reduce losses while minimizing false positives that harm user experience.

---

## ✅ Task 1: Data Preprocessing and Analysis

### Objectives:
- Load, inspect, and clean data from three sources:
  - `Fraud_Data.csv` (e-commerce)
  - `creditcard.csv` (bank transactions)
  - `IpAddress_to_Country.csv` (geo enrichment)
- Convert raw timestamps and IPs for analysis
- Prepare datasets for feature engineering and modeling

### Key Steps:
- Converted `signup_time` and `purchase_time` to datetime
- Transformed `ip_address` float values into integers
- Checked and removed duplicates (credit data had 1081)
- Modularized preprocessing into `Preprocessor` class

---

## ✅ Task 2: EDA and Feature Engineering

### Visual Insights (E-commerce):
- 📊 **Purchase Value Distribution:** Skewed toward lower values  
- 🌍 **Fraud by Source:** Highest fraud ratio observed in `SEO` and `Ads`
- 🧭 **Browser Distribution:** Chrome was dominant, but Firefox and Safari had higher fraud ratios
- 📈 **Boxplot (Fraud vs Purchase):** Subtle difference in purchase values across classes


### Features Engineered:
- ⏰ `hour_of_day`, `day_of_week`, `time_since_signup`
- 📊 `transaction_count` per user, `device_transaction_count`
- 🌍 IP-to-country enrichment

### Reusability:
- Created OOP classes:
  - `EDA` → Plot and interpret patterns
  - `FeatureEngineer` → Add derived features
- Stored in `src/` for easy import and testing

---

## ✅ Reproducibility & Code Hygiene

- All tasks organized in Jupyter notebooks and OOP `.py` modules
- Git branches used per task (`task-1`, `task-2`, etc.)
- Project hygiene established:
  - `.github/workflows/ci.yml` for CI pipelines
  - `scripts/` for automation
  - `tests/` folder scaffolded for unit testing

---

## 📌 Next Steps

- Handle class imbalance using SMOTE
- Train Logistic Regression and XGBoost models
- Evaluate using F1-score and AUC-PR
- Begin Task 3: Explainability with SHAP
