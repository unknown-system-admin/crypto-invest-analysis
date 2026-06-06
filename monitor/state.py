import json
from pathlib import Path

RECORD_PATH = Path(__file__).parent / "record.json"


def load_state(path: Path = RECORD_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_state(data: dict, path: Path = RECORD_PATH) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
