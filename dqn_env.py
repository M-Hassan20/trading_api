"""
dqn_env.py
----------
OpenAI Gym-compatible environment that wraps the PSX Trading API.

Your DQN training loop uses this class — it never touches CSV files directly.
The environment calls the API over HTTP, so:
  - Running locally: set BASE_URL = "http://localhost:8000"
  - Deployed:        set BASE_URL = "https://your-app.railway.app"

Usage
─────
    env = PSXTradingEnv(ticker="OGDC", split="train")
    state = env.reset()

    for _ in range(1000):
        action = agent.select_action(state)   # 0, 1, or 2
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        if done:
            break
"""

import numpy as np
import requests
from typing import Optional

BASE_URL = "http://localhost:8000"   # ← change to deployed URL when live


class PSXTradingEnv:
    """
    Gym-style environment backed by the PSX Trading API.

    Observation space  : flat numpy array (22 features)
    Action space       : Discrete(3)  — 0=Hold, 1=Buy, 2=Sell
    """

    # Feature count must match data_engine._build_state vector length
    # 19 price/indicator features + 3 portfolio features = 22
    OBS_DIM = 22

    def __init__(
        self,
        ticker:       str   = "OGDC",
        split:        str   = "train",   # "train" | "test" | "all"
        initial_cash: float = 100_000.0,
        window:       int   = 30,
        base_url:     str   = BASE_URL,
    ):
        self.ticker       = ticker
        self.split        = split
        self.initial_cash = initial_cash
        self.window       = window
        self.base_url     = base_url
        self.session_id: Optional[str] = None
        self._last_info   = {}

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start (or restart) a session and return the initial state vector."""
        resp = requests.post(
            f"{self.base_url}/session/reset",
            json={
                "ticker":       self.ticker,
                "split":        self.split,
                "initial_cash": self.initial_cash,
                "window":       self.window,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id  = data["session_id"]
        self._last_info  = data
        return self._vec(data["state"]["vector"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Send action to API, get (next_state, reward, done, info).
        action: 0=Hold, 1=Buy, 2=Sell
        """
        assert self.session_id, "Call reset() before step()."
        resp = requests.post(
            f"{self.base_url}/session/step",
            json={"session_id": self.session_id, "action": int(action)},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        next_state = self._vec(data["state"]["vector"]) if data["state"] else np.zeros(self.OBS_DIM)
        reward     = float(data["reward"])
        done       = bool(data["done"])
        info       = data.get("info", {})
        info["portfolio_value"] = data.get("portfolio_value")
        info["cash"]            = data.get("cash")
        info["shares"]          = data.get("shares")

        return next_state, reward, done, info

    def portfolio_info(self) -> dict:
        """Return current portfolio state without stepping."""
        assert self.session_id
        resp = requests.get(
            f"{self.base_url}/session/info",
            params={"session_id": self.session_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def observation_space_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_space_n(self) -> int:
        return 3

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vec(raw: list) -> np.ndarray:
        arr = np.array(raw, dtype=np.float32)
        # Replace NaN/Inf that can sneak in from indicator edges
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


# ── Projection helper (inference time) ───────────────────────────────────────

def get_projection_state(
    ticker:    str,
    from_date: str,
    lookback:  int = 60,
    base_url:  str = BASE_URL,
) -> tuple[np.ndarray, dict]:
    """
    Fetch the state vector for a given ticker and date without a session.
    Use this at INFERENCE TIME to feed your trained DQN and get a strategy.

    Returns
    -------
    state_vector : np.ndarray  — shape (19,) — pass to model.forward()
    metadata     : dict        — latest indicators, horizon info, etc.
    """
    resp = requests.get(
        f"{base_url}/project",
        params={"ticker": ticker, "from_date": from_date, "lookback": lookback},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    vec = np.array(data["state_vector"], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, data


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing PSXTradingEnv against local API...")

    env   = PSXTradingEnv(ticker="OGDC", split="train", window=30)
    state = env.reset()
    print(f"  Initial state shape : {state.shape}")
    print(f"  First 5 values      : {state[:5]}")

    total_reward = 0
    for step in range(5):
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={action}  reward={reward:.2f}  "
              f"portfolio={info['portfolio_value']:.2f}  done={done}")
        if done:
            break

    print(f"  Total reward over 5 steps: {total_reward:.2f}")
    print("Done — API is working correctly.")
