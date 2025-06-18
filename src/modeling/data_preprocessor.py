import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


class ModelingDataPreprocessor:
    def __init__(self, csv_path="../data/cleaned/machine_learning_rating_cleaned.csv"):
        self.data = pd.read_csv(csv_path, low_memory=False)
        self.df = self.data.copy()
    
    def filter_claims(self, target="totalclaims", condition=">0"):
        if condition == ">0":
            self.df = self.df[self.df[target] > 0]
        return self.df

    def prepare_features(self, target_column):
        df = self.df.copy()

        # Select only numeric columns and drop the target
        X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
        y = df[target_column]

        # Replace inf/-inf with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally, clip extreme values (optional safeguard)
        X = X.clip(-1e10, 1e10)

        # Impute NaNs with column mean
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        # Return as DataFrame with original columns
        X = pd.DataFrame(X_imputed, columns=X.columns)

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
