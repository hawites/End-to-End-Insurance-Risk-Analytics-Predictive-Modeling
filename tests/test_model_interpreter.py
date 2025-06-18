from src.modeling.model_interpreter import ModelInterpreter
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

def test_shap_summary_plot():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1)
    model = LinearRegression().fit(X, y)
    interpreter = ModelInterpreter(model, X)
    interpreter.explain_with_shap(50)  # Test SHAP runs
