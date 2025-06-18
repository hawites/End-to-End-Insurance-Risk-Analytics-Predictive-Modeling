import shap
import matplotlib.pyplot as plt

class ModelInterpreter:
    def __init__(self, model, data_sample):

        self.model = model
        self.data_sample = data_sample.select_dtypes(include="number")
        self.explainer = shap.Explainer(self.model, self.data_sample)
        self.shap_values = None

    def compute_shap_values(self):
       
        self.shap_values = self.explainer(self.data_sample)
        return self.shap_values

    def summary_plot(self, plot_type="bar"):
      
        if self.shap_values is None:
            self.compute_shap_values()
        shap.summary_plot(self.shap_values, self.data_sample, plot_type=plot_type)
