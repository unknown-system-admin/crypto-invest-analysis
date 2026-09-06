import json
from pathlib import Path

STATE_PATH = Path(__file__).parent / "state.json"


def init_state(symbols: list, start_equity: float) -> dict:
    return {
        "session_start_equity": start_equity,
        "started_at": None,
        "daily_loss_stopped": False,
        "positions": {s: None for s in symbols},
        "realized_pnl": 0.0,
    }


def load_state(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)