import pytest
from src.preprocessing import Preprocessor
from src.models import ModelTrainer

def test_model_training():
    pp = Preprocessor(
        fraud_path='../data/Fraud_Data.csv',
        ip_path='../data/IpAddress_to_Country.csv',
        credit_path='../data/creditcard.csv'
    )
    pp.load_data()
    pp.convert_datetime_columns()
    pp.convert_ip_to_int()
    fraud_df = pp.fraud_df
    drop_cols = ['signup_time', 'purchase_time', 'ip_address', 'ip_int', 'user_id', 'device_id', 'country']
    mt = ModelTrainer(fraud_df, target_col='class', drop_cols=drop_cols)
    mt.preprocess()
    assert mt.X_train is not None and mt.y_train is not None
