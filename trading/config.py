DEFAULT_STRATEGIES = {
    "ma_cross": {
        "enabled": True,
        "params": {"fast": 12, "slow": 26, "type": "ema"},
        "weight": 1,
    },
    "rsi": {
        "enabled": True,
        "params": {"period": 14, "overbought": 70, "oversold": 30},
        "weight": 1,
    },
    "macd": {
        "enabled": False,
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "weight": 1,
    },
    "composite": {
        "enabled": True,
        "params": {},
        "weight": 2,
    },
}

DEFAULT_RISK = {
    "max_position_pct": 25,
    "min_bars_hold": 1,
    "max_daily_trades": 10,
    "max_drawdown_stop": 30,
    "signal_threshold": 0.6,
}
