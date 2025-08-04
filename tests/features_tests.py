import pytest
from src.preprocessing import Preprocessor
from src.features import FeatureEngineer

def test_add_time_features():
    pp = Preprocessor(
        fraud_path='../data/Fraud_Data.csv',
        ip_path='../data/IpAddress_to_Country.csv',
        credit_path='../data/creditcard.csv'
    )
    pp.load_data()
    pp.convert_datetime_columns()
    fe = FeatureEngineer(pp.fraud_df)
    result_df = fe.add_time_features()
    assert 'hour_of_day' in result_df.columns
    assert 'day_of_week' in result_df.columns
    assert 'time_since_signup' in result_df.columns
