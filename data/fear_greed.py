import json
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError


def fetch_fear_greed(limit: int = 1) -> list:
    url = f"https://api.alternative.me/fng/?limit={limit}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        raise ConnectionError(f"Failed to fetch Fear & Greed Index: {e}")

    if "data" not in data:
        raise ValueError("Unexpected API response")

    for item in data["data"]:
        item["timestamp"] = datetime.fromtimestamp(int(item["timestamp"]))
        item["value"] = int(item["value"])
    return data["data"]
