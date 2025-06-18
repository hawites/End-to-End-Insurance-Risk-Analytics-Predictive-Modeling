from src.modeling.model_builder import ModelBuilder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def test_linear_model():
    builder = ModelBuilder("linear")
    model = builder.build()
    assert isinstance(model, LinearRegression)

def test_random_forest_model():
    builder = ModelBuilder("random_forest")
    model = builder.build()
    assert isinstance(model, RandomForestRegressor)

def test_xgboost_model():
    builder = ModelBuilder("xgboost")
    model = builder.build()
    assert isinstance(model, XGBRegressor)
