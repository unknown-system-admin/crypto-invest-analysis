import pandas as pd
from bot.funding import funding_bias, funding_series_from_rows


def test_funding_bias_positive_threshold():
    assert funding_bias(0.0005, 0.0001) == "short"    # crowded longs -> short
    assert funding_bias(-0.0005, 0.0001) == "long"    # crowded shorts -> long
    assert funding_bias(0.00005, 0.0001) == "neutral"  # mild -> no bias


def test_funding_series_from_rows_sorts_and_indexes():
    rows = [
        {"timestamp": 2000, "fundingRate": 0.0002},
        {"timestamp": 1000, "fundingRate": 0.0001},
        {"timestamp": 3000, "fundingRate": -0.0001},
    ]
    s = funding_series_from_rows(rows)
    assert isinstance(s, pd.Series)
    assert s.index.is_monotonic_increasing
    assert float(s.iloc[0]) == 0.0001
    assert float(s.iloc[-1]) == -0.0001


def test_funding_bias_works_with_series_value():
    s = pd.Series([0.0005])
    assert funding_bias(float(s.iloc[-1]), 0.0001) == "short"
