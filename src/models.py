import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, df, target_col, drop_cols=[]):
        self.df = df.copy()
        self.target = target_col
        self.drop_cols = drop_cols
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = {}

    def preprocess(self):
        # Drop unwanted columns
        X = self.df.drop(self.drop_cols + [self.target], axis=1)
        y = self.df[self.target]

        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        #  Scale features to ensure convergence
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        print("Before SMOTE:", self.y_train.value_counts())

        # Apply SMOTE to training set
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        print("After SMOTE:", pd.Series(self.y_train).value_counts())

    def train_logistic_regression(self):
        model = LogisticRegression(
            max_iter=1000,             
            solver='lbfgs',            
            class_weight='balanced'    
        )
        model.fit(self.X_train, self.y_train)
        self.evaluate(model, "Logistic Regression")

    def train_xgboost(self):
        model = XGBClassifier(
            eval_metric='logloss',  
            verbosity=0,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.evaluate(model, "XGBoost")


    def evaluate(self, model, name):
        preds = model.predict(self.X_test)
        probs = model.predict_proba(self.X_test)[:, 1]
        f1 = f1_score(self.y_test, preds)
        cm = confusion_matrix(self.y_test, preds)
        print(f"\n== {name} Evaluation ==")
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
        print(classification_report(self.y_test, preds))

        # AUC-PR
        precision, recall, _ = precision_recall_curve(self.y_test, probs)
        auc_pr = auc(recall, precision)
        print("AUC-PR:", auc_pr)

        plt.plot(recall, precision, label=f'{name} (AUC-PR = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

        self.results[name] = {'model': model, 'f1': f1, 'auc_pr': auc_pr}
