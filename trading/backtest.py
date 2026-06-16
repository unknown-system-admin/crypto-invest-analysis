import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from trading.strategy import CustomComposite, Signal
from trading.portfolio import Portfolio, Position, Order, calculate_position_size
from trading.executor import PaperExecutor


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    equity_curve: list
    trades: list
    monthly_returns: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _calc_sharpe(returns_series: pd.Series, risk_free: float = 0.02) -> float:
    if len(returns_series) < 2 or returns_series.std() == 0:
        return 0.0
    daily_rf = risk_free / 365
    excess = returns_series - daily_rf
    return float(np.sqrt(365) * excess.mean() / returns_series.std())


def _calc_max_drawdown(equity_series: list) -> float:
    if not equity_series:
        return 0.0
    peak = equity_series[0]
    max_dd = 0.0
    for val in equity_series:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    strategy_configs: Optional[dict] = None,
    signal_threshold: float = 0.6,
    risk_config: Optional[dict] = None,
) -> BacktestResult:
    if strategy_configs is None:
        from trading.config import DEFAULT_STRATEGIES
        strategy_configs = DEFAULT_STRATEGIES
    if risk_config is None:
        from trading.config import DEFAULT_RISK
        risk_config = DEFAULT_RISK

    df_raw = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if df_raw.empty:
        raise ValueError(f"No data for {symbol} {timeframe}")

    result = compute_all(df_raw)
    overlay = result["overlay"]
    subplots = result["subplots"]

    strategy = CustomComposite(strategy_configs, threshold=signal_threshold)
    portfolio = Portfolio(cash=initial_capital, positions=[], orders=[])
    executor = PaperExecutor(portfolio, slippage=slippage)

    equity_curve = []
    trades_log = []
    daily_trade_count = 0
    last_trade_date = ""

    max_position_pct = risk_config.get("max_position_pct", 25)
    min_bars_hold = risk_config.get("min_bars_hold", 1)
    max_daily_trades = risk_config.get("max_daily_trades", 10)
    max_drawdown_stop = risk_config.get("max_drawdown_stop", 30)

    for i in range(min_bars_hold, len(overlay)):
        window = overlay.iloc[:i+1]
        sub_window = {k: v.iloc[:i+1] for k, v in subplots.items()}

        sig = strategy.evaluate(window, sub_window)
        if sig is None or sig.direction == "中立":
            current_dd = _calc_max_drawdown(equity_curve + [portfolio.total_equity])
            if current_dd > max_drawdown_stop:
                pos = [p for p in portfolio.positions if p.side == "long"]
                if pos:
                    executor.execute_sell(symbol, float(overlay["close"].iloc[i]), pos[0].quantity)
            equity_curve.append(portfolio.total_equity)
            continue

        current_price = float(overlay["close"].iloc[i])
        idx = overlay.index[i]
        current_date = str(idx.date()) if hasattr(idx, "date") else str(i)

        if current_date != last_trade_date:
            daily_trade_count = 0
            last_trade_date = current_date

        has_position = any(p.symbol == symbol and p.side == "long" for p in portfolio.positions)

        current_dd = _calc_max_drawdown(equity_curve + [portfolio.total_equity])
        if current_dd > max_drawdown_stop:
            if has_position:
                pos = [p for p in portfolio.positions if p.symbol == symbol and p.side == "long"][0]
                executor.execute_sell(symbol, current_price, pos.quantity)
            equity_curve.append(portfolio.total_equity)
            continue

        if sig.direction == "偏多" and not has_position and executor.can_trade(symbol, daily_trade_count, max_daily_trades):
            qty = calculate_position_size(portfolio.cash, current_price, max_position_pct)
            if qty > 0:
                order = executor.execute_buy(symbol, current_price, qty)
                trades_log.append({
                    "date": current_date, "action": "buy",
                    "price": round(order.price, 2), "quantity": order.quantity,
                })
                daily_trade_count += 1

        elif sig.direction == "偏空" and has_position and executor.can_trade(symbol, daily_trade_count, max_daily_trades):
            pos_list = [p for p in portfolio.positions if p.symbol == symbol and p.side == "long"]
            if pos_list:
                order = executor.execute_sell(symbol, current_price, pos_list[0].quantity)
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
    total_return = ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0

    total_days = len(equity_curve)
    cagr = ((final_equity / initial_capital) ** (365 / max(total_days, 1)) - 1) * 100 if initial_capital > 0 and total_days > 0 else 0.0

    equity_series = pd.Series(equity_curve)
    returns_series = equity_series.pct_change().dropna()
    sharpe = _calc_sharpe(returns_series)

    max_dd = _calc_max_drawdown(equity_curve)

    filled_sells = [o for o in portfolio.orders if o.side == "sell" and o.status == "filled"]
    wins = sum(1 for o in filled_sells if o.pnl > 0)
    total_closed = len(filled_sells)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    monthly_returns = {}
    for i in range(1, len(equity_curve)):
        prev_eq = equity_curve[i-1]
        if prev_eq > 0:
            ret = ((equity_curve[i] - prev_eq) / prev_eq) * 100
        else:
            ret = 0.0
        month_key = str(i // 30)
        if month_key not in monthly_returns:
            monthly_returns[month_key] = []
        monthly_returns[month_key].append(ret)
    monthly_returns_agg = {k: round(sum(v), 2) for k, v in monthly_returns.items()}

    start_date = str(overlay.index[0]) if len(overlay) > 0 else ""
    end_date = str(overlay.index[-1]) if len(overlay) > 0 else ""

    return BacktestResult(
        symbol=symbol, timeframe=timeframe,
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, final_equity=round(final_equity, 2),
        total_return_pct=round(total_return, 2),
        cagr=round(cagr, 2), max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2), win_rate=round(win_rate, 1),
        total_trades=len(trades_log),
        equity_curve=[round(e, 2) for e in equity_curve],
        trades=trades_log, monthly_returns=monthly_returns_agg,
    )
