import pytest
import pandas as pd
from src.preprocessing import Preprocessor

def test_load_data():
    pp = Preprocessor(
        fraud_path='../data/Fraud_Data.csv',
        ip_path='../data/IpAddress_to_Country.csv',
        credit_path='../data/creditcard.csv'
    )
    fraud_df, ip_df, credit_df = pp.load_data()
    assert not fraud_df.empty
    assert not ip_df.empty
    assert not credit_df.empty

def test_datetime_conversion():
    pp = Preprocessor(
        fraud_path='../data/Fraud_Data.csv',
        ip_path='../data/IpAddress_to_Country.csv',
        credit_path='../data/creditcard.csv'
    )
    pp.load_data()
    pp.convert_datetime_columns()
    assert pd.api.types.is_datetime64_any_dtype(pp.fraud_df['signup_time'])
    assert pd.api.types.is_datetime64_any_dtype(pp.fraud_df['purchase_time'])
