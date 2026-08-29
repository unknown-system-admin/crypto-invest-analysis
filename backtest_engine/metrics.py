import numpy as np
from typing import List, Dict, Optional


def calculate_metrics(
    equity_curve: List[float],
    initial_capital: float,
    trades: Optional[List[dict]] = None,
) -> Dict[str, float]:
    if not equity_curve:
        return _empty_metrics()
    
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # Basic metrics
    total_return = ((equity[-1] - initial_capital) / initial_capital) * 100
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_drawdown = np.max(drawdown)
    
    # Sharpe ratio (annualized, assuming hourly data)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252 * 24)
    else:
        sortino = 0.0
    
    # Calmar ratio
    if max_drawdown > 0:
        calmar = total_return / max_drawdown
    else:
        calmar = 0.0
    
    # Trade metrics
    total_trades = 0
    wins = 0
    total_profit = 0.0
    total_loss = 0.0
    
    if trades:
        sell_trades = [t for t in trades if t.get("action") == "sell" and "pnl" in t]
        total_trades = len(sell_trades)
        for t in sell_trades:
            if t["pnl"] > 0:
                wins += 1
                total_profit += t["pnl"]
            else:
                total_loss += abs(t["pnl"])
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0.0
    
    return {
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
    }


def format_report(metrics: Dict[str, float]) -> str:
    lines = [
        "📊 績效報告",
        "─" * 30,
        f"總報酬率:     {metrics['total_return']:.1f}%",
        f"最大回撤:     {metrics['max_drawdown']:.1f}%",
        f"夏普比率:     {metrics['sharpe_ratio']:.2f}",
        f"索提諾比率:   {metrics['sortino_ratio']:.2f}",
        f"卡瑪比率:     {metrics['calmar_ratio']:.2f}",
        f"勝率:         {metrics['win_rate']:.1f}%",
        f"盈虧比:       {metrics['profit_factor']:.2f}",
        f"總交易次數:   {metrics['total_trades']}",
    ]
    return "\n".join(lines)
