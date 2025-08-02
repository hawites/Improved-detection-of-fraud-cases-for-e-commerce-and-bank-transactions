import pandas as pd

class Preprocessor:
    def __init__(self, fraud_path, ip_path, credit_path):
        self.fraud_path = fraud_path
        self.ip_path = ip_path
        self.credit_path = credit_path

        self.fraud_df = None
        self.ip_df = None
        self.credit_df = None

    def load_data(self):
        self.fraud_df = pd.read_csv(self.fraud_path)
        self.ip_df = pd.read_csv(self.ip_path)
        self.credit_df = pd.read_csv(self.credit_path)
        return self.fraud_df, self.ip_df, self.credit_df

    def convert_datetime_columns(self):
        self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'])
        self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'])

    def convert_ip_to_int(self):
        self.fraud_df['ip_int'] = self.fraud_df['ip_address'].astype(float).astype(int)

    def drop_creditcard_duplicates(self):
        self.credit_df.drop_duplicates(inplace=True)

    def check_data_integrity(self):
        datasets = {
            'Fraud Data': self.fraud_df,
            'IP Data': self.ip_df,
            'Credit Card Data': self.credit_df
        }

        for name, df in datasets.items():
            print(f"\n== {name.upper()} ==")
            print("Shape:", df.shape)
            print("Missing values:\n", df.isnull().sum())
            print("Duplicates:", df.duplicated().sum())
