"""
data_engine.py
--------------
Core data layer for the PSX Trading API.

Responsibilities:
- Load and index the master CSV once at startup
- Compute all technical indicators (RSI, MACD, BB, ATR, EMA)
- Manage per-session time cursors so multiple training runs can
  step through data independently
- Accept and persist new/unseen data rows submitted at runtime
- Enforce train (2000-2020) / test (2021-2025) split cleanly
"""

import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Literal

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
DATA_PATH   = "data/psx_final.csv"
NEW_DATA_PATH = "data/psx_new_records.csv"   # unseen data appended here

COMPANIES = [
    "ENGRO","FFC","NML","NCL","OGDC","KAPCO",
    "UBL","HBL","DGKC","PPL","HUBC","LUCK","MCB","PSO"
]


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series):
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    macd   = ema12 - ema26
    signal = _ema(macd, 9)
    return macd, signal, macd - signal

def _bollinger(close: pd.Series, window: int = 20):
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma + 2*std, ma, ma - 2*std          # upper, mid, lower

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to a per-company dataframe."""
    df = df.copy().sort_values("Time")
    c = df["Close"]
    df["EMA_10"]       = _ema(c, 10)
    df["EMA_20"]       = _ema(c, 20)
    df["EMA_50"]       = _ema(c, 50)
    df["RSI"]          = _rsi(c)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = _macd(c)
    df["BB_upper"], df["BB_mid"], df["BB_lower"]   = _bollinger(c)
    df["ATR"]          = _atr(df["High"], df["Low"], c)
    df["Volume_MA_10"] = df["Volume"].rolling(10).mean()
    df["Price_change"] = c.pct_change()
    df["Volatility_10"]= c.rolling(10).std()
    return df


# ── Master DataStore ──────────────────────────────────────────────────────────

class DataStore:
    """
    Singleton-style store loaded once at API startup.
    Holds a dict { ticker -> DataFrame (full history + indicators) }.
    """

    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}
        self._sessions: dict[str, dict] = {}    # session_id -> session state
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self):
        base = pd.read_csv(DATA_PATH, parse_dates=["Time"])

        # Merge any persisted new data
        try:
            extra = pd.read_csv(NEW_DATA_PATH, parse_dates=["Time"])
            base  = pd.concat([base, extra], ignore_index=True)
        except FileNotFoundError:
            pass

        base.sort_values(["Company", "Time"], inplace=True)
        base.drop_duplicates(subset=["Company", "Time"], keep="last", inplace=True)

        for ticker, grp in base.groupby("Company"):
            self._store[ticker] = _compute_indicators(grp.reset_index(drop=True))

        print(f"[DataStore] Loaded {len(base)} rows across {len(self._store)} tickers.")

    def reload(self):
        """Re-read CSVs and recompute indicators (called after new data is ingested)."""
        self._store.clear()
        self._load()

    # ── Getters ───────────────────────────────────────────────────────────────

    def companies(self) -> list[str]:
        return list(self._store.keys())

    def get_full(self, ticker: str) -> pd.DataFrame:
        self._require(ticker)
        return self._store[ticker]

    def get_split(self, ticker: str, split: Literal["train","test","all"]) -> pd.DataFrame:
        df = self.get_full(ticker)
        if split == "train":
            return df[df["Time"] <= TRAIN_END].reset_index(drop=True)
        elif split == "test":
            return df[df["Time"] >= TEST_START].reset_index(drop=True)
        return df.reset_index(drop=True)

    def get_row(self, ticker: str, date: str) -> Optional[pd.Series]:
        """Return the single row for a ticker on a given date (or None)."""
        df = self.get_full(ticker)
        mask = df["Time"] == pd.Timestamp(date)
        if mask.any():
            return df[mask].iloc[0]
        return None

    def get_window(self, ticker: str, end_date: str, window: int = 60) -> pd.DataFrame:
        """
        Return the last `window` trading days up to and including end_date.
        Used by the DQN environment to build the state vector.
        """
        df = self.get_full(ticker)
        idx = df[df["Time"] <= pd.Timestamp(end_date)].index
        if len(idx) == 0:
            return pd.DataFrame()
        last_idx = idx[-1]
        start    = max(0, last_idx - window + 1)
        return df.loc[start:last_idx].reset_index(drop=True)

    # ── New / unseen data ingestion ───────────────────────────────────────────

    def ingest_new_rows(self, rows: list[dict]) -> dict:
        """
        Accept a list of OHLCV dicts for any ticker.
        Validates, deduplicates against existing data, persists to
        psx_new_records.csv, then reloads the store.

        Expected dict keys: Time, Open, High, Low, Close, Volume, Company
        """
        required = {"Time","Open","High","Low","Close","Volume","Company"}
        errors   = []
        valid    = []

        for i, row in enumerate(rows):
            missing = required - set(row.keys())
            if missing:
                errors.append(f"Row {i}: missing fields {missing}")
                continue
            if row["Company"] not in COMPANIES:
                errors.append(f"Row {i}: unknown company '{row['Company']}'")
                continue
            try:
                pd.Timestamp(row["Time"])
            except Exception:
                errors.append(f"Row {i}: invalid date '{row['Time']}'")
                continue
            valid.append(row)

        if valid:
            new_df = pd.DataFrame(valid)
            new_df["Time"] = pd.to_datetime(new_df["Time"])
            # Append to persisted file
            try:
                existing = pd.read_csv(NEW_DATA_PATH, parse_dates=["Time"])
                combined = pd.concat([existing, new_df], ignore_index=True)
            except FileNotFoundError:
                combined = new_df
            combined.drop_duplicates(subset=["Company","Time"], keep="last", inplace=True)
            combined.to_csv(NEW_DATA_PATH, index=False)
            self.reload()   # recompute indicators across full history

        return {
            "ingested": len(valid),
            "rejected": len(errors),
            "errors":   errors
        }

    # ── Session management (sequential stepping for DQN) ─────────────────────

    def session_reset(
        self,
        ticker: str,
        split: Literal["train","test","all"] = "train",
        initial_cash: float = 100_000.0,
        window: int = 30
    ) -> dict:
        """
        Create (or reset) a session.  Returns the first state vector.
        The window parameter controls how many past days form the state.
        """
        self._require(ticker)
        df = self.get_split(ticker, split)
        # Drop rows where indicators are NaN (first ~50 rows)
        df = df.dropna().reset_index(drop=True)

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "ticker":       ticker,
            "split":        split,
            "df":           df,
            "cursor":       window,          # start after enough history for window
            "window":       window,
            "cash":         initial_cash,
            "shares":       0,
            "initial_cash": initial_cash,
            "trade_log":    [],
            "done":         False,
        }
        return {
            "session_id":  session_id,
            "ticker":      ticker,
            "split":       split,
            "total_steps": len(df) - window,
            "start_date":  str(df.iloc[window]["Time"].date()),
            "end_date":    str(df.iloc[-1]["Time"].date()),
            "state":       self._build_state(session_id),
        }

    def session_step(self, session_id: str, action: int) -> dict:
        """
        Execute one action (0=Hold, 1=Buy, 2=Sell) and advance one day.
        Returns (state, reward, done, info).
        Action always trades 1 share for simplicity; scale in your DQN wrapper.
        """
        s = self._require_session(session_id)
        if s["done"]:
            raise ValueError("Session is done. Call /session/reset to start over.")

        df      = s["df"]
        cursor  = s["cursor"]
        row     = df.iloc[cursor]
        price   = float(row["Close"])
        date    = str(row["Time"].date())

        prev_value = s["cash"] + s["shares"] * price
        trade_info = {"date": date, "price": price, "action": action, "shares_before": s["shares"]}

        # ── Execute action ────────────────────────────────────────────────────
        if action == 1:                         # BUY
            max_shares = int(s["cash"] // price)
            if max_shares > 0:
                buy_qty      = max(1, max_shares // 10)   # buy 10% of affordable
                s["cash"]   -= buy_qty * price
                s["shares"] += buy_qty
                trade_info["executed"] = f"BUY {buy_qty}"
            else:
                trade_info["executed"] = "BUY skipped (insufficient cash)"

        elif action == 2:                       # SELL
            if s["shares"] > 0:
                sell_qty     = max(1, s["shares"] // 2)   # sell 50% of holdings
                s["cash"]   += sell_qty * price
                s["shares"] -= sell_qty
                trade_info["executed"] = f"SELL {sell_qty}"
            else:
                trade_info["executed"] = "SELL skipped (no shares)"
        else:
            trade_info["executed"] = "HOLD"

        # ── Advance cursor ────────────────────────────────────────────────────
        s["cursor"] += 1
        done = s["cursor"] >= len(df)
        s["done"] = done

        # ── Reward = change in total portfolio value ──────────────────────────
        next_price = float(df.iloc[min(s["cursor"], len(df)-1)]["Close"])
        curr_value = s["cash"] + s["shares"] * next_price
        reward     = curr_value - prev_value

        s["trade_log"].append(trade_info)

        return {
            "state":          self._build_state(session_id) if not done else None,
            "reward":         reward,
            "portfolio_value":curr_value,
            "cash":           s["cash"],
            "shares":         s["shares"],
            "done":           done,
            "info":           trade_info,
        }

    def session_info(self, session_id: str) -> dict:
        s = self._require_session(session_id)
        df    = s["df"]
        price = float(df.iloc[min(s["cursor"], len(df)-1)]["Close"])
        return {
            "session_id":     session_id,
            "ticker":         s["ticker"],
            "split":          s["split"],
            "cursor":         s["cursor"],
            "current_date":   str(df.iloc[min(s["cursor"], len(df)-1)]["Time"].date()),
            "cash":           s["cash"],
            "shares":         s["shares"],
            "portfolio_value":s["cash"] + s["shares"] * price,
            "roi_pct":        round((s["cash"] + s["shares"] * price - s["initial_cash"]) / s["initial_cash"] * 100, 2),
            "done":           s["done"],
            "trade_count":    len(s["trade_log"]),
        }

    def get_projection_window(
        self,
        ticker: str,
        from_date: str,
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        Return last `lookback` days before from_date.
        Used by the /project endpoint to feed the DQN its observation.
        """
        return self.get_window(ticker, from_date, window=lookback)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _require(self, ticker: str):
        if ticker not in self._store:
            raise ValueError(f"Unknown ticker '{ticker}'. Valid: {list(self._store.keys())}")

    def _require_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        return self._sessions[session_id]

    def _build_state(self, session_id: str) -> dict:
        """
        Build the state vector for the current cursor position.
        Returns both raw values (for inspection) and a flat numeric list
        (ready to feed directly into the DQN).
        """
        s      = self._sessions[session_id]
        df     = s["df"]
        cursor = s["cursor"]
        window = s["window"]

        start  = max(0, cursor - window)
        hist   = df.iloc[start:cursor]
        latest = df.iloc[cursor - 1]

        price  = float(latest["Close"])
        port_v = s["cash"] + s["shares"] * price

        feature_cols = [
            "Close","Open","High","Low","Volume",
            "EMA_10","EMA_20","EMA_50",
            "RSI","MACD","MACD_signal","MACD_hist",
            "BB_upper","BB_mid","BB_lower",
            "ATR","Volume_MA_10","Price_change","Volatility_10"
        ]

        # Last row of window as flat feature vector + portfolio context
        row_features = [float(latest.get(c, 0) or 0) for c in feature_cols]
        portfolio_features = [
            s["cash"] / s["initial_cash"],          # normalised cash ratio
            s["shares"] * price / s["initial_cash"], # normalised holdings ratio
            port_v / s["initial_cash"],              # normalised portfolio ratio
        ]

        return {
            "date":               str(latest["Time"].date()),
            "close":              price,
            "rsi":                float(latest.get("RSI") or 0),
            "macd":               float(latest.get("MACD") or 0),
            "ema_10":             float(latest.get("EMA_10") or 0),
            "bb_upper":           float(latest.get("BB_upper") or 0),
            "bb_lower":           float(latest.get("BB_lower") or 0),
            "atr":                float(latest.get("ATR") or 0),
            "cash":               s["cash"],
            "shares":             s["shares"],
            "portfolio_value":    port_v,
            "vector":             row_features + portfolio_features,  # flat list for DQN
        }
