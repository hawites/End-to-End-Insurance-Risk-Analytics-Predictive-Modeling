from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import shap

class ModelEvaluator:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def evaluate(self, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"RMSE": rmse, "R2": r2}

    def explain_with_shap(self, sample_size=100):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_train[:sample_size])
        shap.summary_plot(shap_values, self.X_train[:sample_size])