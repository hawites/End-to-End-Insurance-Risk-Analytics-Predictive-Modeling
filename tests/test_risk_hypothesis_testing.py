import pandas as pd
from src.risk_hypothesis_testing import compute_claim_metrics

def test_compute_claim_metrics():
    df = pd.DataFrame({
        'totalclaims': [100, 0, 50],
        'totalpremium': [150, 100, 50]
    })
    df = compute_claim_metrics(df)
    assert 'claim_occurred' in df.columns
    assert 'margin' in df.columns
    assert df['claim_occurred'].sum() == 2
    assert (df['margin'] == [50, 100, 0]).all()