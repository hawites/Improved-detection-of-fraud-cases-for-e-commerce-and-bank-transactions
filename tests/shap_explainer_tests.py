import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.shap_explainer import SHAPExplainer

def test_shap_explainer_basic_usage():
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(df, y)

    # Use SHAPExplainer
    explainer = SHAPExplainer(model, df)
    explainer.compute_shap_values()

    # Assert that SHAP values are computed
    assert explainer.shap_values is not None
    assert len(explainer.shap_values[1]) == len(df)

