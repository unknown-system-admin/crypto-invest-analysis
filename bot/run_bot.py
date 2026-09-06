# bot/run_bot.py
"""HFT bot CLI.

Usage:
  python -m bot.run_bot --dry-run            # signals only (default)
  python -m bot.run_bot --live               # place orders on OKX demo (needs API keys)
  python -m bot.run_bot --dry-run --threshold 0.15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import BotConfig
from bot.executor import OKXExecutor
from bot.loop import run_bot
from data_cache import load_or_fetch


def main():
    parser = argparse.ArgumentParser(description="HFT bot (OKX demo)")
    parser.add_argument("--live", action="store_true",
                        help="place real demo orders (requires OKX_API_KEY/SECRET/PASSPHRASE env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="signals only, no orders (default if no --live)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override HTF direction threshold")
    parser.add_argument("--poll", type=int, default=None,
                        help="override poll interval seconds")
    args = parser.parse_args()

    cfg = BotConfig(dry_run=not args.live)
    if args.threshold:
        cfg.htf_threshold = args.threshold
    if args.poll:
        cfg.poll_seconds = args.poll

    executor = None
    if args.live:
        executor = OKXExecutor()
        print("LIVE mode: placing orders on OKX demo (sandbox).")
    else:
        print("DRY-RUN mode: signals only, no orders.")

    print(f"Symbols: {cfg.symbols}")
    print(f"HTF {cfg.htf_timeframe} (threshold {cfg.htf_threshold}) -> LTF {cfg.ltf_timeframe}")
    print(f"Leverage {cfg.leverage}x, margin {cfg.margin_pct:.0%}, "
          f"stop {cfg.stop_loss_pct:.1%}, daily stop {cfg.max_daily_loss_pct:.0%}")
    print("Press Ctrl+C to stop (positions preserved).")

    run_bot(cfg, executor, load_or_fetch)


if __name__ == "__main__":
    main()