from src.analysis.risk_hypothesis_testing import HypothesisTester
import pandas as pd

def test_province_risk_hypothesis():
    df = pd.read_csv("data/cleaned/machine_learning_rating_cleaned.csv")
    tester = HypothesisTester(df)
    tester.test_province_risk_difference()
    assert "province_risk" in tester.results
