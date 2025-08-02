import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df):
        self.df = df

    def plot_distribution(self, col, bins=50, kde=True):
        sns.histplot(self.df[col], bins=bins, kde=kde)
        plt.title(f"Distribution of {col}")
        plt.show()

    def plot_count(self, col):
        sns.countplot(x=col, data=self.df)
        plt.title(f"Count of {col}")
        plt.show()

    def plot_fraud_by_category(self, col, target='class'):
        sns.countplot(x=col, hue=target, data=self.df)
        plt.title(f"Fraud by {col}")
        plt.show()

    def plot_box(self, col, target='class'):
        sns.boxplot(x=target, y=col, data=self.df)
        plt.title(f"{col} vs Fraud")
        plt.show()
