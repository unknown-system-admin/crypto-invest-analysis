from dataclasses import dataclass, field


@dataclass
class BotConfig:
    symbols: list = field(default_factory=lambda: ["BTC/USDT:USDT", "SOL/USDT:USDT"])
    htf_timeframe: str = "1h"
    ltf_timeframe: str = "15m"
    htf_candles: int = 800
    ltf_candles: int = 1440
    leverage: int = 5
    margin_pct: float = 0.20
    # v2 grid (2026-09-06, Binance 15m/1h, ATR stop + vol filter): best thr=0.30 but no intraday edge
    # (BTC -0.4% / SOL +6.5%, trades 17-26 <100, vol filter OFF) — keep v1 default; see calibration_results_v2.json
    htf_threshold: float = 0.25  # calibrated (preliminary): BTC=0.25 SOL=0.30; see calibration_results.json
    stop_loss_pct: float = 0.025
    atr_stop_mult: float = 2.5
    min_atr_pct: float = 0.0015
    signal_mode: str = "momentum"  # "momentum" | "meanreversion"
    rsi_oversold: int = 25
    rsi_overbought: int = 75
    max_daily_loss_pct: float = 0.10
    poll_seconds: int = 20
    dry_run: bool = True
    fee_rate: float = 0.001  # per side (fee + slippage) for calibration
    initial_capital: float = 10000.0

    def position_notional(self, equity: float) -> float:
        return equity * self.margin_pct * self.leverage