from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from src.modeling.model_evaluator import ModelEvaluator

def test_evaluation_metrics():
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    evaluator = ModelEvaluator(model, X)
    scores = evaluator.evaluate(y, y_pred)
    assert "RMSE" in scores and "R2" in scores
