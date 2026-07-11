"""
Regression tests against known-leaky fixture datasets. If these ever
start passing when they shouldn't, a leakage check has regressed.
"""
from pathlib import Path

import pandas as pd

from agentic_ml.harness.leakage import check_suspicious_feature_correlation

FIXTURES_DIR = Path(__file__).parent / "leaky_fixtures"


def test_obvious_feature_leak_is_caught():
    df = pd.read_csv(FIXTURES_DIR / "obvious_feature_leak.csv")
    X = df.drop(columns=["churned"])
    y = df["churned"]
    check = check_suspicious_feature_correlation(X, y)
    assert not check.passed, "known leaky fixture was not caught — regression!"
    assert "churn_flag_copy" in check.detail
