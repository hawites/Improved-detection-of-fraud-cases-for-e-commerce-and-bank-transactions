## ✅ Task 1: Data Analysis and Preprocessing

### Datasets Used:
- Fraud_Data.csv (E-commerce transactions)
- creditcard.csv (Bank transactions)
- IpAddress_to_Country.csv (Geolocation enrichment)

### Key Steps:
- Organized data loading and cleaning code into `src/preprocessing.py`
- Converted time columns and IPs, dropped duplicates
- Verified class imbalance: ~9% fraud (ecom), ~0.17% fraud (bank)
- Ready for EDA and geolocation merging in Task 2

### Next:
- Merge IP ranges for country-based fraud insights.
- Engineer new time and user behavior features.

## ✅ Task 2: Exploratory Data Analysis and Feature Engineering

### Summary

In Task 2, we performed exploratory data analysis (EDA) and created new features to enrich the e-commerce fraud dataset. All logic was modularized into Python classes using an OOP approach and called from a clean notebook.

### Key Activities

- **Modularized Analysis:**
  - Created `EDA` and `FeatureEngineer` classes in `src/eda.py` and `src/features.py`.

- **Exploratory Data Analysis (EDA):**
  - Plotted purchase value distribution.
  - Examined fraud distribution by source, browser, and purchase value.
  - Observed a higher proportion of fraud from `SEO` and `Ads` sources.
  - Noted that the distribution of purchase value is right-skewed.

- **Feature Engineering:**
  - Extracted time-based features:
    - `hour_of_day`, `day_of_week` from `purchase_time`
    - `time_since_signup` in hours
  - Added frequency-based features:
    - `transaction_count` per `user_id`
    - `device_transaction_count` per `device_id`
  - Enriched data with a `country` column by merging IP ranges with the IP-to-country dataset.

### Notebooks and Files

- `notebooks/02_eda_feature_engineering.ipynb`
- `src/eda.py` — EDA class
- `src/features.py` — Feature engineering class

### Visuals Generated

- Distribution of `purchase_value`
- Count of `browser`
- Fraud by `source`
- Boxplot of `purchase_value` vs `class`

### Next Steps

- Explore and preprocess the credit card fraud dataset.
- Address class imbalance in both datasets.
- Train and evaluate fraud detection models (Logistic Regression and XGBoost).
