import pandas as pd
from analysis.momentum import states_from_df


def _make_df(n=250, base=100, up=True):
    price = [base + (i if up else -i) for i in range(n)]
    return pd.DataFrame({
        "open": price, "high": [p + 1 for p in price],
        "low": [p - 1 for p in price], "close": price,
        "volume": [1000] * n,
    })


def test_states_from_df_offsets():
    df = _make_df(250, base=100, up=True)
    states = states_from_df(df, step=1)
    assert len(states) == 3
    assert states[-1]["direction"] in ("偏多", "震盪")
