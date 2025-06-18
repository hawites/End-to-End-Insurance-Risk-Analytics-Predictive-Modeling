import pytest
from src.modeling.data_preprocessor import ModelingDataPreprocessor

def test_filter_claims():
    prep = ModelingDataPreprocessor()
    df_filtered = prep.filter_claims()
    assert (df_filtered["totalclaims"] > 0).all()

def test_prepare_features():
    prep = ModelingDataPreprocessor()
    prep.filter_claims()
    X, y = prep.prepare_features("totalclaims")
    assert X.shape[0] == y.shape[0]
