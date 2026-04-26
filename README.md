# PSX Trading API

A **time-machine stock data API** built on FastAPI that replays PSX historical data
(2000–2025) day-by-day for DQN agent training, backtesting, and inference.

---

## Project Structure

```
trading_api/
├── main.py                  # FastAPI app — all endpoints
├── data_engine.py           # Data loading, indicators, session management
├── dqn_env.py               # Gym-style env wrapper your DQN talks to
├── requirements.txt
├── Procfile                 # For Railway/Render deployment
└── data/
    ├── psx_final.csv        # Your master dataset (2000-2025, 14 tickers)
    └── psx_new_records.csv  # Auto-created when you POST /data/ingest
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API
uvicorn main:app --reload --port 8000

# 3. Open interactive docs
# http://localhost:8000/docs
```

---

## Endpoint Reference

### Stock Data
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/companies` | List all 14 tickers |
| GET | `/stocks/history?ticker=OGDC&split=train` | OHLCV + indicators for a ticker |
| GET | `/stocks/date?ticker=HBL&date=2023-06-15` | Single day snapshot |
| GET | `/stocks/window?ticker=LUCK&end_date=2022-03-01&window=60` | Last N days (state builder) |
| GET | `/data/splits` | Date ranges per split per ticker |

### DQN Training Loop
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/session/reset` | Create a session, get session_id + first state |
| POST | `/session/step` | Execute action, get reward + next state |
| GET | `/session/info` | Portfolio value, cursor, ROI |
| GET | `/session/state` | Current state without stepping |

### Inference / Projection
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/project?ticker=ENGRO&from_date=2024-06-01` | Get state vector for any date |
| POST | `/data/ingest` | Submit new OHLCV rows; auto-persisted + re-indexed |

---

## Data Splits

| Split | Date Range | Purpose |
|-------|-----------|---------|
| `train` | 2000-01-01 → 2020-12-31 | DQN training episodes |
| `test` | 2021-01-01 → 2025-11-28 | Backtesting / evaluation |
| `all` | Full history | Projection with full context |

---

## DQN Integration

### Training Loop

```python
from dqn_env import PSXTradingEnv

env   = PSXTradingEnv(ticker="OGDC", split="train", initial_cash=100_000)
state = env.reset()   # shape: (22,)

for episode in range(500):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)          # 0=Hold, 1=Buy, 2=Sell
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        total_reward += reward
        if done:
            break

    print(f"Episode {episode}: total_reward={total_reward:.2f}")
```

### State Vector (22 features)

```
Index  Feature
────────────────────────
0      Close price
1      Open price
2      High price
3      Low price
4      Volume
5      EMA 10
6      EMA 20
7      EMA 50
8      RSI (14)
9      MACD
10     MACD Signal
11     MACD Histogram
12     Bollinger Upper
13     Bollinger Mid (SMA 20)
14     Bollinger Lower
15     ATR (14)
16     Volume MA (10)
17     Price change (pct)
18     Volatility (10-day std)
19     Cash ratio (cash / initial_cash)
20     Holdings ratio (shares*price / initial_cash)
21     Portfolio ratio (total / initial_cash)
```

### Inference (Any Date → Strategy)

```python
from dqn_env import get_projection_state

state_vec, meta = get_projection_state(
    ticker    = "ENGRO",
    from_date = "2025-06-01",   # can be any date with data before it
    lookback  = 60
)

# state_vec is shape (19,) — portfolio features not included at inference
action = trained_dqn.forward(state_vec).argmax()
print(["HOLD","BUY","SELL"][action])
print("Last RSI :", meta["latest_indicators"]["rsi"])
```

### Ingesting New / Unseen Data

```python
import requests

resp = requests.post("http://localhost:8000/data/ingest", json={
    "rows": [
        {
            "Time":    "2025-12-01",
            "Open":    1250.50,
            "High":    1270.00,
            "Low":     1240.00,
            "Close":   1260.75,
            "Volume":  3500000.0,
            "Company": "ENGRO"
        }
    ]
})
print(resp.json())
# {"status": "ok", "ingested": 1, "rejected": 0, ...}
```

The row is persisted to `data/psx_new_records.csv` and the DataStore
is immediately reloaded with recalculated indicators. Future calls to
`/project` for dates on or after your new rows will include them.

---

## Deployment (Railway — Free Tier)

```bash
# 1. Create a GitHub repo and push this folder
git init && git add . && git commit -m "PSX Trading API"
gh repo create psx-trading-api --public --push

# 2. Go to railway.app → New Project → Deploy from GitHub repo
# 3. Add environment variable (optional): PORT=8000
# 4. Railway auto-detects Procfile and deploys

# Your live URL: https://psx-trading-api-production.up.railway.app
```

Update `BASE_URL` in `dqn_env.py` to your live URL for remote training.

---

## Reward Design Notes

The current reward is `Δ portfolio value` (simple P&L). For your DQN
you may want to experiment with:

| Reward | Formula | Effect |
|--------|---------|--------|
| Sharpe-adjusted | `Δpnl / rolling_std(Δpnl)` | Penalises volatility |
| Risk-penalised | `Δpnl - λ * drawdown` | Discourages large losses |
| Trade-penalised | `Δpnl - cost_per_trade` | Reduces overtrading |

Modify `session_step()` in `data_engine.py` to implement these.
