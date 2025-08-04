import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test
        self.explainer = shap.Explainer(self.model, self.X_test)
        self.shap_values = None

    def compute_shap_values(self):
        print("Computing SHAP values...")
        self.shap_values = self.explainer(self.X_test)
        return self.shap_values

    def plot_summary(self, save_path=None):
        if self.shap_values is None:
            self.compute_shap_values()
        print("Generating summary plot...")
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_force(self, index=0, save_path=None):
        if self.shap_values is None:
            self.compute_shap_values()
        print(f"Generating force plot for index {index}...")
        force_plot = shap.plots.force(self.shap_values[index], matplotlib=True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
