from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import itertools
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from trading.strategy import CustomComposite
from trading.portfolio import Portfolio, calculate_position_size
from trading.executor import PaperExecutor
from trading.config import DEFAULT_STRATEGIES, DEFAULT_RISK
from trading.backtest import _calc_sharpe, _calc_max_drawdown


@dataclass
class ParamSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    risk_config: dict = field(default_factory=lambda: DEFAULT_RISK.copy())
    strategy_order: list = field(default_factory=lambda: ["ma_cross", "rsi", "macd", "composite"])
    weight_min: int = 0
    weight_max: int = 3
    threshold_start: float = 0.3
    threshold_stop: float = 0.9
    threshold_step: float = 0.1


@dataclass
class SweepResult:
    params: dict
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_equity: float
    cagr: float

    def to_dict(self):
        return asdict(self)


def _run_single_simulation(
    overlay: pd.DataFrame,
    subplots: dict,
    strategy_configs: dict,
    signal_threshold: float,
    config: ParamSweepConfig,
) -> SweepResult:
    strategy = CustomComposite(strategy_configs, threshold=signal_threshold)
    portfolio = Portfolio(cash=config.initial_capital, positions=[], orders=[])
    executor = PaperExecutor(portfolio, slippage=config.slippage, fee_rate=config.fee_rate)

    equity_curve = []
    trades_log = []
    daily_trade_count = 0
    last_trade_date = ""

    risk = config.risk_config
    max_pos_pct = risk.get("max_position_pct", 25)
    min_bars_hold = risk.get("min_bars_hold", 1)
    max_daily_trades = risk.get("max_daily_trades", 10)
    max_dd_stop = risk.get("max_drawdown_stop", 30)

    for i in range(min_bars_hold, len(overlay)):
        window = overlay.iloc[:i + 1]
        sub_window = {k: v.iloc[:i + 1] for k, v in subplots.items()}
        sig = strategy.evaluate(window, sub_window)
        current_price = float(overlay["close"].iloc[i])
        idx = overlay.index[i]

        current_date = str(idx.date()) if hasattr(idx, "date") else str(i)
        if current_date != last_trade_date:
            daily_trade_count = 0
            last_trade_date = current_date

        has_position = any(
            p.symbol == config.symbol and p.side == "long"
            for p in portfolio.positions
        )

        current_dd = _calc_max_drawdown(equity_curve + [portfolio.total_equity])
        if current_dd > max_dd_stop:
            pos = [p for p in portfolio.positions if p.symbol == config.symbol and p.side == "long"]
            if pos:
                executor.execute_sell(config.symbol, current_price, pos[0].quantity)
            equity_curve.append(portfolio.total_equity)
            continue

        if sig and sig.direction == "偏多" and not has_position and executor.can_trade(
            config.symbol, daily_trade_count, max_daily_trades
        ):
            qty = calculate_position_size(portfolio.cash, current_price, max_pos_pct)
            if qty > 0:
                order = executor.execute_buy(config.symbol, current_price, qty)
                trades_log.append({
                    "date": current_date, "action": "buy",
                    "price": round(order.price, 2), "quantity": order.quantity,
                })
                daily_trade_count += 1

        elif sig and sig.direction == "偏空" and has_position and executor.can_trade(
            config.symbol, daily_trade_count, max_daily_trades
        ):
            pos_list = [p for p in portfolio.positions if p.symbol == config.symbol and p.side == "long"]
            if pos_list:
                order = executor.execute_sell(config.symbol, current_price, pos_list[0].quantity)
                if order:
                    trades_log.append({
                        "date": current_date, "action": "sell",
                        "price": round(order.price, 2), "quantity": order.quantity,
                        "pnl": round(order.pnl, 2), "pnl_pct": round(order.pnl_pct, 2),
                    })
                    daily_trade_count += 1

        for p in portfolio.positions:
            p.update_market(current_price)
        equity_curve.append(portfolio.total_equity)

    final_equity = portfolio.total_equity
    total_return = ((final_equity - config.initial_capital) / config.initial_capital) * 100 if config.initial_capital > 0 else 0.0
    total_days = max((overlay.index[-1] - overlay.index[0]).days, 1)
    cagr = ((final_equity / config.initial_capital) ** (365 / max(total_days, 1)) - 1) * 100 if config.initial_capital > 0 and total_days > 0 else 0.0

    equity_series = pd.Series(equity_curve)
    returns_series = equity_series.pct_change().dropna()
    sharpe = _calc_sharpe(returns_series)
    max_dd = _calc_max_drawdown(equity_curve)

    filled_sells = [o for o in portfolio.orders if o.side == "sell" and o.status == "filled"]
    wins = sum(1 for o in filled_sells if o.pnl > 0)
    total_closed = len(filled_sells)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    return SweepResult(
        params={},
        total_return_pct=round(total_return, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        total_trades=len(trades_log),
        final_equity=round(final_equity, 2),
        cagr=round(cagr, 2),
    )


def run_parameter_sweep(
    config: ParamSweepConfig,
    progress_callback: Optional[Callable[[int, int, Optional[SweepResult]], None]] = None,
) -> list[SweepResult]:
    df = fetch_ohlcv(config.symbol, timeframe=config.timeframe, limit=config.limit)
    if df.empty:
        raise ValueError(f"No data for {config.symbol} {config.timeframe}")

    result = compute_all(df)
    overlay = result["overlay"]
    subplots = result["subplots"]

    weight_values = list(range(config.weight_min, config.weight_max + 1))
    thresholds = []
    t = config.threshold_start
    while t <= config.threshold_stop + 1e-9:
        thresholds.append(round(t, 1))
        t = round(t + config.threshold_step, 1)

    n_strategies = len(config.strategy_order)
    total = (len(weight_values) ** n_strategies) * len(thresholds)
    count = 0
    results = []

    for weights in itertools.product(weight_values, repeat=n_strategies):
        for thresh in thresholds:
            strategy_configs = {}
            for i, sname in enumerate(config.strategy_order):
                params_copy = DEFAULT_STRATEGIES.get(sname, {}).get("params", {}).copy()
                strategy_configs[sname] = {
                    "enabled": weights[i] > 0,
                    "params": params_copy,
                    "weight": int(weights[i]),
                }

            sr = _run_single_simulation(overlay, subplots, strategy_configs, thresh, config)
            sr.params = {
                **{f"{sname}_weight": int(weights[j]) for j, sname in enumerate(config.strategy_order)},
                "threshold": thresh,
            }

            results.append(sr)
            count += 1
            if progress_callback:
                progress_callback(count, total, sr)

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results


RESULTS_DIR = Path(__file__).resolve().parent / "optimization_results"


def save_sweep_results(results: list[SweepResult], config: ParamSweepConfig) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_sym = config.symbol.replace("/", "_")
    filename = f"{timestamp}_{safe_sym}_{config.timeframe}.json"
    path = RESULTS_DIR / filename
    data = {
        "config": asdict(config),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(path)


def load_sweep_results(path: str) -> tuple[ParamSweepConfig, list[SweepResult]]:
    with open(path) as f:
        data = json.load(f)
    config = ParamSweepConfig(**data["config"])
    results = [SweepResult(**r) for r in data["results"]]
    return config, results


def list_sweep_result_files() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted([str(p) for p in RESULTS_DIR.iterdir() if p.suffix == ".json"], reverse=True)
