class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def add_time_features(self):
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds() / 3600
        return self.df

    def add_transaction_frequency(self):
        user_freq = self.df.groupby('user_id').size().reset_index(name='transaction_count')
        self.df = self.df.merge(user_freq, on='user_id', how='left')

        device_freq = self.df.groupby('device_id').size().reset_index(name='device_transaction_count')
        self.df = self.df.merge(device_freq, on='device_id', how='left')

        return self.df

    def add_country_from_ip(self, ip_df):
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float).astype(int)
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float).astype(int)

        def get_country(ip):
            match = ip_df[
                (ip_df['lower_bound_ip_address'] <= ip) &
                (ip_df['upper_bound_ip_address'] >= ip)
            ]
            return match['country'].values[0] if not match.empty else 'Unknown'

        self.df['country'] = self.df['ip_int'].apply(get_country)
        return self.df
