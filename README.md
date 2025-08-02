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
