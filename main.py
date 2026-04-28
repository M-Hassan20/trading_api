"""
main.py
-------
FastAPI application for the PSX Historical Trading Data API.

Endpoints
─────────
GET  /                          Health check
GET  /companies                 List all tickers
GET  /stocks/history            OHLCV + indicators for a ticker over a date range
GET  /stocks/date               Single day snapshot for a ticker
GET  /stocks/window             Last N days before a date (state builder for DQN)

POST /session/reset             Create a new stepping session (DQN training loop)
POST /session/step              Execute buy/sell/hold and advance one day
GET  /session/info              Current session portfolio & cursor info
GET  /session/state             Current state vector without stepping

POST /data/ingest               Submit new/unseen OHLCV rows; persisted + re-indexed
GET  /data/splits               Show available date ranges per split

GET  /project                   Return the observation window for a from_date
                                (plug this into your trained DQN to get a strategy)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import pandas as pd
import numpy as np

from data_engine import DataStore

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PSX Historical Trading API",
    description=(
        "Time-machine API that replays PSX stock data (2000-2025) day-by-day "
        "for DQN training, backtesting, and forward projection."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global DataStore — loaded once at startup
store = DataStore()


# ── Pydantic models ───────────────────────────────────────────────────────────

class OHLCVRow(BaseModel):
    Time:    str   = Field(..., example="2025-12-01")
    Open:    float = Field(..., example=1250.50)
    High:    float = Field(..., example=1270.00)
    Low:     float = Field(..., example=1240.00)
    Close:   float = Field(..., example=1260.75)
    Volume:  float = Field(..., example=3500000.0)
    Company: str   = Field(..., example="ENGRO")

class IngestRequest(BaseModel):
    rows: List[OHLCVRow]

class SessionResetRequest(BaseModel):
    ticker:       str                              = Field(..., example="OGDC")
    split:        Literal["train","test","all"]    = Field("train")
    initial_cash: float                            = Field(100_000.0, ge=1000)
    window:       int                              = Field(30, ge=5, le=120)

class StepRequest(BaseModel):
    session_id: str = Field(..., example="<uuid from /session/reset>")
    action:     int = Field(..., ge=0, le=2,
                            description="0 = Hold  |  1 = Buy  |  2 = Sell")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "ok",
        "companies": store.companies(),
        "message":   "PSX Trading API is running."
    }


# ── Company / catalogue ───────────────────────────────────────────────────────

@app.get("/companies", tags=["Catalogue"])
def get_companies():
    """Return all available tickers."""
    return {"companies": store.companies()}


# ── Stock data ────────────────────────────────────────────────────────────────

@app.get("/stocks/history", tags=["Stock Data"])
def get_history(
    ticker:     str = Query(..., example="OGDC"),
    split:      Literal["train", "test", "all"] = Query("all", description="train | test | all"),
    start_date: Optional[str] = Query(None, example="2015-01-01"),
    end_date:   Optional[str] = Query(None, example="2015-12-31"),
    include_indicators: bool  = Query(True)
):
    """
    Return OHLCV + technical indicators for a ticker.
    Only `ticker` is required — all other parameters have defaults.
    Filter by split (train/test/all) and optional date range.
    """
    try:
        df = store.get_split(ticker, split)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to load data for '{ticker}': {e}")

    try:
        if start_date:
            df = df[df["Time"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["Time"] <= pd.Timestamp(end_date)]
    except Exception as e:
        raise HTTPException(400, f"Invalid date filter: {e}")

    if not include_indicators:
        df = df[["Time", "Open", "High", "Low", "Close", "Volume", "Company"]]

    # Safe serialisation
    # 1. Format timestamps
    df = df.copy()
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d")

    # 2. Convert to records then sanitise every float value.
    #    pandas .where(notnull) doesn't work reliably on float64 columns —
    #    None gets silently upcast back to NaN. Walk the records manually.
    raw_records = df.to_dict(orient="records")

    def _clean(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val

    records = [{k: _clean(v) for k, v in row.items()} for row in raw_records]

    return {
        "ticker": ticker,
        "split":  split,
        "rows":   len(records),
        "data":   records,
    }


@app.get("/stocks/date", tags=["Stock Data"])
def get_date_snapshot(
    ticker: str = Query(..., example="HBL"),
    date:   str = Query(..., example="2023-06-15")
):
    """
    Return a single day's OHLCV + indicators for a ticker.
    If the date is a weekend/holiday, returns the nearest prior trading day.
    """
    try:
        row = store.get_row(ticker, date)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if row is None:
        # Find nearest prior trading day
        df   = store.get_full(ticker)
        mask = df["Time"] <= pd.Timestamp(date)
        if not mask.any():
            raise HTTPException(404, f"No data for {ticker} on or before {date}")
        row = df[mask].iloc[-1]

    return {"ticker": ticker, "data": {k: str(v) for k, v in row.items()}}


@app.get("/stocks/window", tags=["Stock Data"])
def get_window(
    ticker:   str = Query(..., example="LUCK"),
    end_date: str = Query(..., example="2022-03-01"),
    window:   int = Query(60,  ge=5, le=252,
                          description="Number of trading days to return")
):
    """
    Return the last N trading days up to (and including) end_date.
    This is the primary state-builder endpoint for the DQN environment.
    """
    try:
        df = store.get_window(ticker, end_date, window)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if df.empty:
        raise HTTPException(404, f"No data found for {ticker} before {end_date}")

    df["Time"] = df["Time"].astype(str)
    return {
        "ticker":   ticker,
        "end_date": end_date,
        "window":   len(df),
        "data":     df.to_dict(orient="records")
    }


@app.get("/data/splits", tags=["Stock Data"])
def get_splits():
    """Show the date ranges for each split across all tickers."""
    result = {}
    for ticker in store.companies():
        train = store.get_split(ticker, "train")
        test  = store.get_split(ticker, "test")
        result[ticker] = {
            "train": {
                "start": str(train["Time"].min().date()) if len(train) else "N/A",
                "end":   str(train["Time"].max().date()) if len(train) else "N/A",
                "rows":  len(train)
            },
            "test": {
                "start": str(test["Time"].min().date()) if len(test) else "N/A",
                "end":   str(test["Time"].max().date()) if len(test) else "N/A",
                "rows":  len(test)
            }
        }
    return result


# ── New data ingestion ────────────────────────────────────────────────────────

@app.post("/data/ingest", tags=["Data Ingestion"])
def ingest_data(request: IngestRequest):
    """
    Submit new or unseen OHLCV rows.

    - Rows are validated (required fields, known ticker, parseable date)
    - Deduplicated against existing data (Company + Date is the key)
    - Persisted to data/psx_new_records.csv
    - The entire DataStore is reloaded so indicators recompute over the
      full history including the new rows

    Use this when you have data for dates AFTER 2025-11-28 or want to
    correct/supplement existing entries.
    """
    rows_dicts = [r.model_dump() for r in request.rows]
    result = store.ingest_new_rows(rows_dicts)
    return {
        "status":   "ok" if result["rejected"] == 0 else "partial",
        "ingested": result["ingested"],
        "rejected": result["rejected"],
        "errors":   result["errors"],
        "message":  (
            f"Successfully ingested {result['ingested']} rows. "
            f"DataStore reloaded with updated indicators."
        ) if result["ingested"] > 0 else "No valid rows to ingest."
    }


# ── DQN Session (stepping loop) ───────────────────────────────────────────────

@app.post("/session/reset", tags=["DQN Session"])
def session_reset(req: SessionResetRequest):
    """
    Start a new stepping session for DQN training or backtesting.

    Returns a session_id.  Use it in /session/step calls.

    split="train"  → data from 2000-01-01 to 2020-12-31
    split="test"   → data from 2021-01-01 to 2025-11-28
    split="all"    → full history
    """
    try:
        result = store.session_reset(
            ticker=req.ticker,
            split=req.split,
            initial_cash=req.initial_cash,
            window=req.window
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/session/step", tags=["DQN Session"])
def session_step(req: StepRequest):
    """
    Execute one action and step forward one trading day.

    action: 0 = Hold | 1 = Buy | 2 = Sell

    Returns:
      state          → next observation (None when done=True)
      reward         → change in portfolio value (your DQN loss signal)
      portfolio_value→ total current value (cash + shares × price)
      done           → True when the split's last date is reached
      info           → trade execution details
    """
    try:
        result = store.session_step(req.session_id, req.action)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/session/info", tags=["DQN Session"])
def session_info(session_id: str = Query(...)):
    """Current portfolio value, cursor position, and session metadata."""
    try:
        return store.session_info(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/session/state", tags=["DQN Session"])
def session_state(session_id: str = Query(...)):
    """
    Return the current state vector without stepping forward.
    Useful for inspecting what the DQN is observing.
    """
    try:
        s = store._require_session(session_id)
        return store._build_state(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ── Projection (inference endpoint) ──────────────────────────────────────────

@app.get("/project", tags=["Projection"])
def project(
    ticker:   str = Query(..., example="ENGRO"),
    from_date:str = Query(..., example="2024-06-01",
                          description=(
                              "The date FROM which you want a strategy. "
                              "The API returns the lookback window ending on this date — "
                              "feed the 'vector' field into your trained DQN to get "
                              "the Buy/Hold/Sell decision."
                          )),
    lookback: int = Query(60, ge=10, le=252,
                          description="How many past trading days to include in the observation"),
    horizon_days: int = Query(21, ge=1, le=90,
                              description="How many future trading days the projection covers (metadata only)")
):
    """
    Projection endpoint — used at inference time.

    Returns:
    - The observation window (last `lookback` days of OHLCV + indicators)
    - A flat `state_vector` ready to pass to your DQN's forward() method
    - Metadata about what future window the strategy will cover

    The API does NOT run the DQN for you — it prepares the input.
    Your Python client calls this, feeds state_vector into the model,
    and interprets the output action (0=Hold, 1=Buy, 2=Sell).

    If from_date is AFTER the last known date, the API will use whatever
    data it has up to that date (including any ingested new rows).
    """
    try:
        df = store.get_projection_window(ticker, from_date, lookback)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if df.empty:
        raise HTTPException(
            404,
            f"No data for {ticker} before {from_date}. "
            "If this is a future date, ingest recent rows via POST /data/ingest first."
        )

    latest = df.iloc[-1]

    feature_cols = [
        "Close","Open","High","Low","Volume",
        "EMA_10","EMA_20","EMA_50",
        "RSI","MACD","MACD_signal","MACD_hist",
        "BB_upper","BB_mid","BB_lower",
        "ATR","Volume_MA_10","Price_change","Volatility_10"
    ]
    state_vector = [float(latest.get(c) or 0) for c in feature_cols]

    df["Time"] = df["Time"].astype(str)

    # Approximate horizon end date (skip weekends naively for display)
    from datetime import datetime, timedelta
    try:
        start_dt  = datetime.strptime(from_date, "%Y-%m-%d")
        end_dt    = start_dt + timedelta(days=int(horizon_days * 1.4))  # pad for weekends
        horizon_end = end_dt.strftime("%Y-%m-%d")
    except Exception:
        horizon_end = "unknown"

    return {
        "ticker":        ticker,
        "from_date":     from_date,
        "horizon_days":  horizon_days,
        "horizon_end":   horizon_end,
        "actual_last_date": str(latest["Time"])[:10] if hasattr(latest["Time"], "strftime") else str(latest["Time"])[:10],
        "lookback_rows": len(df),
        "state_vector":  state_vector,
        "feature_names": feature_cols,
        "latest_indicators": {
            "close":    float(latest.get("Close") or 0),
            "rsi":      round(float(latest.get("RSI") or 0), 2),
            "macd":     round(float(latest.get("MACD") or 0), 4),
            "ema_10":   round(float(latest.get("EMA_10") or 0), 2),
            "ema_50":   round(float(latest.get("EMA_50") or 0), 2),
            "bb_upper": round(float(latest.get("BB_upper") or 0), 2),
            "bb_lower": round(float(latest.get("BB_lower") or 0), 2),
            "atr":      round(float(latest.get("ATR") or 0), 4),
            "volatility_10": round(float(latest.get("Volatility_10") or 0), 4),
        },
        "observation_window": df.to_dict(orient="records"),
    }