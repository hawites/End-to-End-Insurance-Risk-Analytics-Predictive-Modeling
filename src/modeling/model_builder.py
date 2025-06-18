from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class ModelBuilder:
    def __init__(self, model_type="linear"):
        self.model_type = model_type

    def build(self):
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        else:
            raise ValueError("Invalid model type.")
