# 動能趨勢分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓手動觸發的日報能比較「本次 vs 前兩次」的動能狀態，呈現動能演進（如「偏多但開始減弱」），純即時計算、不儲存歷史。

**Architecture:** 新增 `analysis/momentum.py` 提供純函式計算任意 K 棒 offset 的動能狀態與演進判斷。`monitor/main.py` 的 `/report` 接受 `tf`/`step` 查詢參數（config 提供預設），對同一份 OHLCV 用多個 offset 各算一次 momentum state，並把結果併入 `build_report_embed`。

**Tech Stack:** Python 3.12, pandas, FastAPI, pytest, ccxt

---

### Task 1: momentum state 計算（純函式）

**Files:**
- Create: `analysis/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: 寫失敗測試**

`tests/test_momentum.py`:
```python
from analysis.momentum import classify_state


def test_classify_strong_bullish():
    signals = {"direction": "偏多", "bullish_count": 4, "bearish_count": 0, "total": 4}
    s = classify_state(signals)
    assert s["direction"] == "偏多"
    assert s["strength"] == "強"


def test_classify_medium_bullish():
    signals = {"direction": "偏多", "bullish_count": 3, "bearish_count": 1, "total": 4}
    assert classify_state(signals)["strength"] == "中"


def test_classify_weak_bullish():
    signals = {"direction": "偏多", "bullish_count": 1, "bearish_count": 2, "total": 3}
    assert classify_state(signals)["strength"] == "弱"
```

- [ ] **Step 2: 確認測試失敗**

Run: `.venv/bin/pytest tests/test_momentum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.momentum'`

- [ ] **Step 3: 寫最小實作**

`analysis/momentum.py`:
```python
STRENGTH_STRONG = 0.75
STRENGTH_MEDIUM = 0.5


def classify_state(signals: dict) -> dict:
    direction = signals["direction"]
    ratio = signals["bullish_count"] / signals["total"]
    if ratio >= STRENGTH_STRONG:
        strength = "強"
    elif ratio >= STRENGTH_MEDIUM:
        strength = "中"
    else:
        strength = "弱"
    return {"direction": direction, "strength": strength}
```

- [ ] **Step 4: 確認測試通過**

Run: `.venv/bin/pytest tests/test_momentum.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add analysis/momentum.py tests/test_momentum.py
git commit -m "feat: add momentum state classification"
```

### Task 2: momentum 演進判斷（純函式）

**Files:**
- Modify: `analysis/momentum.py`
- Test: `tests/test_momentum.py`

- [ ] **Step 1: 寫失敗測試（追加到 test_momentum.py）**

```python
from analysis.momentum import classify_state, momentum_trend


def test_trend_weakening():
    states = [
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "中"},
        {"direction": "偏多", "strength": "弱"},
    ]
    trend = momentum_trend(states)
    assert trend["label"] == "減弱中"
    assert trend["trend"] == "weakening"


def test_trend_strengthening():
    states = [
        {"direction": "偏空", "strength": "弱"},
        {"direction": "偏空", "strength": "中"},
        {"direction": "偏空", "strength": "強"},
    ]
    assert momentum_trend(states)["label"] == "增強中"
    assert momentum_trend(states)["trend"] == "strengthening"


def test_trend_stable():
    states = [
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "強"},
    ]
    trend = momentum_trend(states)
    assert trend["label"] == "維持"
    assert trend["trend"] == "stable"


def test_trend_reversal():
    states = [
        {"direction": "偏空", "strength": "強"},
        {"direction": "偏多", "strength": "中"},
        {"direction": "偏多", "strength": "強"},
    ]
    assert momentum_trend(states)["label"] == "方向反轉"
    assert momentum_trend(states)["trend"] == "reversal"
```

注意：`trend` 一律用英文字串（`weakening` / `strengthening` / `stable` / `reversal`），避免中文識別元。

- [ ] **Step 2: 確認測試失敗**

Run: `.venv/bin/pytest tests/test_momentum.py -v`
Expected: FAIL — `ImportError: cannot import name 'momentum_trend'`

- [ ] **Step 3: 寫實作**

`analysis/momentum.py` 追加：
```python
def _strength_key(strength: str) -> int:
    return {"強": 2, "中": 1, "弱": 0}.get(strength, 1)


def momentum_trend(states: list) -> dict:
    """states: oldest → newest，回傳 {label, trend, states}"""
    dirs = [s["direction"] for s in states]
    if len(set(dirs)) > 1:
        return {"label": "方向反轉", "trend": "reversal", "states": states}
    scores = [_strength_key(s["strength"]) for s in states]
    if scores == sorted(scores) and len(set(scores)) > 1:
        return {"label": "增強中", "trend": "strengthening", "states": states}
    if scores == sorted(scores, reverse=True) and len(set(scores)) > 1:
        return {"label": "減弱中", "trend": "weakening", "states": states}
    return {"label": "維持", "trend": "stable", "states": states}
```

- [ ] **Step 4: 確認測試通過**

Run: `.venv/bin/pytest tests/test_momentum.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add analysis/momentum.py tests/test_momentum.py
git commit -m "feat: add momentum trend classification"
```

### Task 3: config 新增 reports 預設

**Files:**
- Modify: `monitor/config.yaml`

- [ ] **Step 1: 加設定**

於 `monitor/config.yaml` 底部新增：
```yaml
reports:
  default_tf: "1d"
  default_step: 1
```

- [ ] **Step 2: 驗證 YAML 可載入**

Run: `python3 -c "import yaml; print(yaml.safe_load(open('monitor/config.yaml'))['reports'])"`
Expected: `{'default_tf': '1d', 'default_step': 1}`

- [ ] **Step 3: Commit**

```bash
git add monitor/config.yaml
git commit -m "config: add report momentum defaults"
```

### Task 4: build_report_embed 加入動能演進

**Files:**
- Modify: `monitor/notifier.py`
- Test: `tests/test_monitor_notifier.py`

- [ ] **Step 1: 寫失敗測試（追加）**

```python
from monitor.notifier import build_report_embed


def test_report_embed_includes_momentum():
    momentum = {
        "label": "減弱中",
        "states": [
            {"direction": "偏多", "strength": "強"},
            {"direction": "偏多", "strength": "中"},
            {"direction": "偏多", "strength": "弱"},
        ],
    }
    embed = build_report_embed("BTC/USDT", "整體偏多", [], momentum=momentum)
    desc = embed["embeds"][0]["description"]
    assert "動能演進" in desc
    assert "減弱中" in desc
```

- [ ] **Step 2: 確認測試失敗**

Run: `.venv/bin/pytest tests/test_monitor_notifier.py::test_report_embed_includes_momentum -v`
Expected: FAIL — `build_report_embed()` 收到意外的 `momentum` keyword（TypeError）

- [ ] **Step 3: 修改 `notifier.py`**

`build_report_embed` 改簽名並加入 render logic。先確認 file-top 有 `from typing import Optional`（無則加）：
```python
from typing import Optional
```

函式：
```python
def build_report_embed(symbol: str, summary, tf_results: list,
                       momentum: Optional[dict] = None) -> dict:
    if isinstance(summary, dict):
        summary = "\n".join(f"{k}: {v}" for k, v in summary.items())
    tf_lines = [f"{r['label']}: {r['direction']}" for r in tf_results]
    description = summary
    if tf_lines:
        description += "\n\n📈 **多時間框架**\n" + " | ".join(tf_lines)
    if momentum:
        arrow = " → ".join(
            f"{s['direction']}({s['strength']})" for s in momentum["states"])
        description += f"\n\n⚡ **動能演進**: {arrow} — {momentum['label']}"
    return {
        "embeds": [{
            "title": f"📊 市場日報 — {symbol}",
            "description": description,
            "color": DISCORD_COLORS["report"],
        }]
    }
```

- [ ] **Step 4: 確認測試通過**

Run: `.venv/bin/pytest tests/test_monitor_notifier.py -q`
Expected: PASS（含新測試、既有 notifier 測試仍過）

- [ ] **Step 5: Commit**

```bash
git add monitor/notifier.py tests/test_monitor_notifier.py
git commit -m "feat: add momentum line to report embed"
```

### Task 5: `/report` 支援 tf/step 並接入 momentum

**Files:**
- Modify: `analysis/momentum.py`（追加 `states_from_df`）
- Modify: `monitor/main.py`（`report()` 函式）
- Test: `tests/test_monitor_report.py`（新增）

**背景**：現有 `report()`（約 line 183-223）用 `_process_symbol(symbol)`（固定 1h, limit=200）對每個 symbol 產生 summary + multi_tf + embed。`compute_all`/`analyze_signals` 都吃完整 df 並用 `.iloc[-1]`；為取得過往 offset 的訊號，需將 df 切片為截至 `-1`、`-1-step`、`-1-2*step` 的三份局部 df，各自 `compute_all` + `analyze_signals`。

- [ ] **Step 1: 寫失敗測試**

`tests/test_monitor_report.py`:
```python
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
```

- [ ] **Step 2: 確認測試失敗**

Run: `.venv/bin/pytest tests/test_monitor_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'states_from_df'`

- [ ] **Step 3: 實作 `states_from_df`**

於 `analysis/momentum.py` 頂部確認有 `import pandas as pd`，並追加函式（`compute_all`/`analyze_signals` 在函式內局部 import 避免循環依賴）：
```python
def states_from_df(df: pd.DataFrame, step: int) -> list:
    """回傳 oldest → newest 的三個 momentum state"""
    from indicators.calculator import compute_all
    from analysis.summary import analyze_signals
    offsets = [0, step, 2 * step]
    states = []
    for off in offsets:
        df_view = df.iloc[: len(df) - off] if off > 0 else df
        if len(df_view) < 30:
            states.append({"direction": "震盪", "strength": "弱"})
            continue
        res = compute_all(df_view)
        sig = analyze_signals(res["overlay"], res["subplots"])
        states.append(classify_state(sig))
    return list(reversed(states))
```

- [ ] **Step 4: 確認測試通過**

Run: `.venv/bin/pytest tests/test_monitor_report.py -v`
Expected: PASS

- [ ] **Step 5: 修改 `/report` 路由接上參數與 momentum**

在 `monitor/main.py` 中：
1. 新增 import：`from typing import Optional`、`from analysis.momentum import momentum_trend, states_from_df`、`from indicators.calculator import compute_all`。
2. 改寫 `report()`：
```python
@app.get("/report")
def report(tf: Optional[str] = None, step: Optional[int] = None):
    rcfg = CONFIG.get("reports", {})
    tf = tf or rcfg.get("default_tf", "1d")
    step = step or rcfg.get("default_step", 1)
    webhook = CONFIG["discord"]["webhook_url"]
    reports_sent = []
    errors = []

    for symbol in CONFIG["symbols"]:
        try:
            df = fetch_ohlcv(symbol, timeframe=tf, limit=220)
        except Exception as e:
            errors.append(f"{symbol}: fetch/compute failed: {e}")
            continue
        try:
            comp = compute_all(df)
            summary = generate_market_summary(comp["overlay"], comp["subplots"])
        except Exception as e:
            errors.append(f"{symbol}: generate_market_summary failed: {e}")
            continue
        try:
            tf_results = analyze_multi_timeframe(symbol)
        except Exception:
            tf_results = []
        states = states_from_df(df, step)
        mtrend = momentum_trend(states)
        try:
            embed = build_report_embed(symbol, summary, tf_results, momentum=mtrend)
            send_webhook(webhook, embed)
            reports_sent.append(symbol)
        except Exception as e:
            errors.append(f"{symbol}: send_webhook failed: {e}")
    return {"status": "ok",
            "reports": reports_sent, "errors": errors,
            "time": datetime.now(timezone.utc).isoformat()}
```
3. 注意：`compute_all(df)` 只算一次存到 `comp`；`build_report_embed` 的 `momentum` 參數已由 Task 4 補上（可選），既有測試相容。

- [ ] **Step 6: 執行測試確認既有都過**

Run: `.venv/bin/pytest tests/ -q`
Expected: 全部 PASS（新增測試 + 既有測試）

- [ ] **Step 7: Commit**

```bash
git add analysis/momentum.py tests/test_monitor_report.py monitor/main.py
git commit -m "feat: report supports tf/step momentum trend"
```

### Task 6: 全量驗證 + push

- [ ] **Step 1: 跑全量測試**

Run: `.venv/bin/pytest tests/ -q`
Expected: PASS，並確認無 `report_schedule` 殘留引用：`grep -rn "report_schedule" --include=*.py`（僅 docs 檔案可留）

- [ ] **Step 2: 確認 git 乾淨**

```bash
git status
```
Expected: working tree clean。

- [ ] **Step 3: push**

```bash
git push
```
若遇權限錯誤，請使用者 `gh auth login -h github.com -w` 後重試。

---

## Self-Review 檢查清單

- **spec 覆蓋**：direction/strength mapping（Task1）、演進判斷（Task2）、URL 參數 tf/step + config 預設（Task3+5）、`⚡ 動能演進` render（Task4）、測試（各 task）。全齊。
- **無佔位**：所有測試與實作碼完整，無 "TODO"。
- **命名一致性**：`classify_state` / `momentum_trend` / `states_from_df` 全表同名；`trend` 值為英文字串。
- **執行順序**：Task4（notifier）在 Task5（route）之前，符合依賴。
- **已知風險**：`states_from_df` 依賴 `analyze_signals(overlay, subplots)`；多 offset 切片下 `ta` 各指標需足量歷史資料，`step` 較大或 `tf=1w` 時可能因 SMA200 需 200 bar 而資料不足。Task5 已加 `len(df_view) < 30 → 降級` 防護；若遇到列不足，把下限調大即可。
