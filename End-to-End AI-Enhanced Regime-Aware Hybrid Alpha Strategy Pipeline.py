#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Script: Dr. Ray Islam

Script 1/6 — Steps 1–5 (Scope + Ingest + Clean)

Steps covered:

define scope, 2) ingest OHLCV, 3) ingest news, 4) clean market, 5) clean news"""

# script_01_define_and_ingest.py
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests
from dotenv import load_dotenv


@dataclass
class Config:
    ticker: str = "AAPL"
    start: str = "2025-01-01"
    end: str = "2025-12-31"
    polygon_base_url: str = "https://api.polygon.io"
    polygon_limit: int = 50000
    news_sleep_s: float = 13.0
    base_dir: Path = Path.cwd()


def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    features_dir = data_dir / "features"
    sentiment_dir = data_dir / "sentiment"
    portfolio_dir = data_dir / "portfolio"
    backtests_dir = data_dir / "backtests"
    reports_dir = data_dir / "reports"
    logs_dir = cfg.base_dir / "logs"

    for d in [raw_dir, processed_dir, features_dir, sentiment_dir, portfolio_dir, backtests_dir, reports_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "features_dir": features_dir,
        "sentiment_dir": sentiment_dir,
        "portfolio_dir": portfolio_dir,
        "backtests_dir": backtests_dir,
        "reports_dir": reports_dir,
        "logs_dir": logs_dir,
    }


def polygon_get(api_key: str, url: str, params: Optional[dict] = None, max_retries: int = 8) -> dict:
    params = dict(params or {})
    params["apiKey"] = api_key
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                time.sleep(min(60, 5 * attempt))
                continue

            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(60, 2 * attempt))
                continue

            raise RuntimeError(f"Polygon error {r.status_code}: {r.text[:400]}")
        except Exception as e:
            last_err = e
            time.sleep(min(60, 2 * attempt))

    raise RuntimeError(f"Polygon request failed: {last_err}")


def step01_define_scope(cfg: Config, out_dirs: Dict[str, Path]) -> None:
    spec = {
        "ticker": cfg.ticker,
        "start": cfg.start,
        "end": cfg.end,
        "objective": "Predict next-week (5 trading days) return; weekly rebalance; long-only; hybrid trend+reversion+sentiment with regime gating.",
    }
    spec_path = out_dirs["processed_dir"] / "strategy_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"[STEP 1] Wrote strategy spec -> {spec_path}")


def step02_ingest_ohlcv(cfg: Config, api_key: str, out_dirs: Dict[str, Path]) -> Path:
    url = f"{cfg.polygon_base_url}/v2/aggs/ticker/{cfg.ticker}/range/1/day/{cfg.start}/{cfg.end}"
    data = polygon_get(api_key, url, params={"adjusted": "true", "sort": "asc", "limit": cfg.polygon_limit})
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No OHLCV returned. Response head: {json.dumps(data)[:500]}")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["ticker"] = cfg.ticker
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    out_path = out_dirs["raw_dir"] / f"{cfg.ticker}_ohlcv_daily_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"[STEP 2] Saved raw OHLCV -> {out_path}")
    return out_path


def step03_ingest_news(cfg: Config, api_key: str, out_dirs: Dict[str, Path]) -> Path:
    url = f"{cfg.polygon_base_url}/v2/reference/news"
    params = {
        "ticker": cfg.ticker,
        "published_utc.gte": cfg.start,
        "published_utc.lte": cfg.end,
        "order": "asc",
        "limit": 100,
    }

    all_items = []
    next_url = None
    while True:
        data = polygon_get(api_key, next_url or url, params={} if next_url else params)
        all_items.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        time.sleep(cfg.news_sleep_s)

    df = pd.DataFrame(all_items)
    out_path = out_dirs["raw_dir"] / f"{cfg.ticker}_news_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"[STEP 3] Saved raw news -> {out_path} (rows={len(df)})")
    return out_path


def step04_clean_market(raw_ohlcv_path: Path, out_dirs: Dict[str, Path]) -> Path:
    df = pd.read_csv(raw_ohlcv_path, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date", "ticker"]).sort_values("date").reset_index(drop=True)

    # basic sanity checks
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] >= 0]

    ticker = str(df["ticker"].iloc[0])
    out_path = out_dirs["processed_dir"] / f"{ticker}_ohlcv_daily_clean.csv"

    df.to_csv(out_path, index=False)
    print(f"[STEP 4] Saved cleaned OHLCV -> {out_path}")
    return out_path


def step05_clean_news(raw_news_path: Path, ticker: str, out_dirs: Dict[str, Path]) -> Path:
    df = pd.read_csv(raw_news_path)
    if df.empty:
        out_path = out_dirs["processed_dir"] / f"{ticker}_news_clean.csv"
        df.to_csv(out_path, index=False)
        print(f"[STEP 5] No news rows; wrote empty -> {out_path}")
        return out_path

    if "published_utc" in df.columns:
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
        df["date"] = df["published_utc"].dt.tz_convert(None).dt.normalize()
    else:
        df["date"] = pd.NaT

    # remove empty titles/urls if present
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
    if "article_url" in df.columns:
        df["article_url"] = df["article_url"].fillna("").astype(str)

    # dedup
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    elif "article_url" in df.columns:
        df = df.drop_duplicates(subset=["article_url"], keep="first")

    # keep useful columns
    keep = [c for c in ["date", "published_utc", "title", "description", "article_url", "id"] if c in df.columns]
    df = df[keep].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out_path = out_dirs["processed_dir"] / f"{ticker}_news_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[STEP 5] Saved cleaned news -> {out_path} (rows={len(df)})")
    return out_path


def main():
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Missing POLYGON_API_KEY in environment or .env")

    cfg = Config()
    out_dirs = make_dirs(cfg)

    step01_define_scope(cfg, out_dirs)
    raw_ohlcv = step02_ingest_ohlcv(cfg, api_key, out_dirs)
    raw_news = step03_ingest_news(cfg, api_key, out_dirs)

    clean_ohlcv = step04_clean_market(raw_ohlcv, out_dirs)
    _ = step05_clean_news(raw_news, cfg.ticker, out_dirs)

    print("\nDONE script_01. Next: script_02_market_features.py")


if __name__ == "__main__":
    main()


# In[4]:


"""
Script: Dr. Ray Islam

Script 2/6 — Step 6 (Market Feature Engineering)

Steps covered:
6) price/volume panel (returns, vol, liquidity)"""

# script_02_market_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Config:
    ticker: str = "AAPL"
    vol_lookback: int = 21
    base_dir: Path = Path.cwd()


def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    processed_dir = data_dir / "processed"
    features_dir = data_dir / "features"
    for d in [processed_dir, features_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"processed_dir": processed_dir, "features_dir": features_dir}


def step06_build_returns_panel(clean_ohlcv_path: Path, cfg: Config, out_dirs: Dict[str, Path]) -> Path:
    df = pd.read_csv(clean_ohlcv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df["ret_1d"] = df["close"].pct_change()
    df["logret_1d"] = np.log(df["close"]).diff()
    df["fwd_ret_5d"] = df["close"].shift(-5) / df["close"] - 1.0

    df["dollar_volume"] = df["close"] * df["volume"]
    df["adv_shares_20d"] = df["volume"].rolling(20).mean()
    df["adv_dollars_20d"] = df["dollar_volume"].rolling(20).mean()

    df["vol_21d"] = df["ret_1d"].rolling(cfg.vol_lookback).std()
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)

    df = df.dropna(subset=["ret_1d"]).reset_index(drop=True)

    out_csv = out_dirs["features_dir"] / f"{cfg.ticker}_returns_panel.csv"
    out_parq = out_dirs["features_dir"] / f"{cfg.ticker}_returns_panel.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False)
    print(f"[STEP 6] Saved returns panel -> {out_csv} and parquet")
    return out_parq


def main():
    cfg = Config()
    out_dirs = make_dirs(cfg)

    clean_ohlcv_path = out_dirs["processed_dir"] / f"{cfg.ticker}_ohlcv_daily_clean.csv"
    if not clean_ohlcv_path.exists():
        raise FileNotFoundError(f"Missing {clean_ohlcv_path}. Run script_01 first.")

    _ = step06_build_returns_panel(clean_ohlcv_path, cfg, out_dirs)
    print("\nDONE script_02. Next: script_03_sentiment_features.py")


if __name__ == "__main__":
    main()


# In[1]:


pip install -U xformers


# In[3]:


"""
Script: Dr. Ray Islam
 
 Script 3/6 — Steps 7–9 (FinBERT scoring + Daily Sentiment + Feature Store Merge)

Steps covered:
7) sentiment scoring, 8) daily sentiment, 9) feature store merge"""

# script_03_sentiment_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from pathlib import Path

def resolve_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

@dataclass
class Config:
    ticker: str = "AAPL"
    base_dir: Path = resolve_base_dir()


def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    processed_dir = data_dir / "processed"
    features_dir = data_dir / "features"
    sentiment_dir = data_dir / "sentiment"
    for d in [processed_dir, features_dir, sentiment_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"processed_dir": processed_dir, "features_dir": features_dir, "sentiment_dir": sentiment_dir}


def try_load_finbert():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)
    except Exception:
        return None


def step07_score_news_finbert(news_clean_path: Path, cfg: Config, out_dirs: Dict[str, Path]) -> Path:
    news = pd.read_csv(news_clean_path, parse_dates=["date"])
    if news.empty:
        out_path = out_dirs["sentiment_dir"] / f"{cfg.ticker}_news_scored.csv"
        news.to_csv(out_path, index=False)
        print(f"[STEP 7] Empty news; wrote empty scored file -> {out_path}")
        return out_path

    nlp = try_load_finbert()
    if nlp is None:
        print("[STEP 7] [WARN] FinBERT not available. Using neutral sentiment for all.")
        news["sentiment_label"] = "neutral"
        news["sentiment_score"] = 0.0
    else:
        texts = []
        if "title" in news.columns and "description" in news.columns:
            texts = (news["title"].fillna("") + ". " + news["description"].fillna("")).tolist()
        elif "title" in news.columns:
            texts = news["title"].fillna("").tolist()
        else:
            texts = [""] * len(news)

        labels = []
        scores = []
        batch = 16
        for i in range(0, len(texts), batch):
            preds = nlp(texts[i:i+batch])
            for p in preds:
                lbl = str(p.get("label", "neutral")).lower()
                labels.append(lbl)
                if "positive" in lbl:
                    scores.append(1.0)
                elif "negative" in lbl:
                    scores.append(-1.0)
                else:
                    scores.append(0.0)

        news["sentiment_label"] = labels
        news["sentiment_score"] = scores

    out_path = out_dirs["sentiment_dir"] / f"{cfg.ticker}_news_scored.csv"
    news.to_csv(out_path, index=False)
    print(f"[STEP 7] Saved scored news -> {out_path}")
    return out_path


def step08_aggregate_daily_sentiment(news_scored_path: Path, cfg: Config, out_dirs: Dict[str, Path]) -> Path:
    df = pd.read_csv(news_scored_path, parse_dates=["date"])
    if df.empty:
        daily = pd.DataFrame(columns=["date", "sent_mean", "sent_sum", "sent_count"])
    else:
        daily = (
            df.groupby("date", as_index=False)
            .agg(
                sent_mean=("sentiment_score", "mean"),
                sent_sum=("sentiment_score", "sum"),
                sent_count=("sentiment_score", "count"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

    out_path = out_dirs["sentiment_dir"] / f"{cfg.ticker}_sentiment_daily.csv"
    daily.to_csv(out_path, index=False)
    print(f"[STEP 8] Saved daily sentiment -> {out_path}")
    return out_path


def step09_build_feature_store(returns_panel_path: Path, sentiment_daily_path: Path, cfg: Config, out_dirs: Dict[str, Path]) -> Path:
    panel = pd.read_parquet(returns_panel_path)
    sent = pd.read_csv(sentiment_daily_path, parse_dates=["date"])

    merged = panel.merge(sent, on="date", how="left")
    merged["sent_mean"] = merged["sent_mean"].fillna(0.0)
    merged["sent_sum"] = merged["sent_sum"].fillna(0.0)
    merged["sent_count"] = merged["sent_count"].fillna(0).astype(int)

    merged = merged.sort_values("date").reset_index(drop=True)

    out_parq = out_dirs["features_dir"] / f"{cfg.ticker}_feature_store.parquet"
    out_csv = out_dirs["features_dir"] / f"{cfg.ticker}_feature_store.csv"
    merged.to_parquet(out_parq, index=False)
    merged.to_csv(out_csv, index=False)
    print(f"[STEP 9] Saved feature store -> {out_parq} (and csv)")
    return out_parq


def main():
    cfg = Config()
    out_dirs = make_dirs(cfg)

    news_clean_path = out_dirs["processed_dir"] / f"{cfg.ticker}_news_clean.csv"
    returns_panel_path = out_dirs["features_dir"] / f"{cfg.ticker}_returns_panel.parquet"

    if not news_clean_path.exists():
        raise FileNotFoundError(f"Missing {news_clean_path}. Run script_01 first.")
    if not returns_panel_path.exists():
        raise FileNotFoundError(f"Missing {returns_panel_path}. Run script_02 first.")

    scored = step07_score_news_finbert(news_clean_path, cfg, out_dirs)
    daily = step08_aggregate_daily_sentiment(scored, cfg, out_dirs)
    _ = step09_build_feature_store(returns_panel_path, daily, cfg, out_dirs)

    print("\nDONE script_03. Next: script_04_alphas_regime_gating.py")


if __name__ == "__main__":
    main()


# In[5]:


"""
Script: Dr. Ray Islam

Script 4/6 — Steps 10–12 (Alpha sleeves + Regime + Gating)

Steps covered:
10) alpha sleeves, 11) regime detection, 12) gating ensemble"""

# script_04_alphas_regime_gating.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Config:
    ticker: str = "AAPL"
    mom_lookback: int = 20
    rev_lookback: int = 5
    z_window: int = 60
    vol_regime_threshold: float = 0.02
    base_dir: Path = resolve_base_dir()


def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    features_dir = data_dir / "features"
    for d in [features_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"features_dir": features_dir}


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-12)


def step10_alpha_sleeves(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mom = df["close"] / df["close"].shift(cfg.mom_lookback) - 1.0
    rev = df["close"] / df["close"].shift(cfg.rev_lookback) - 1.0

    df["alpha_trend"] = zscore(mom, cfg.z_window)
    df["alpha_reversion"] = -zscore(rev, cfg.z_window)
    df["alpha_sentiment"] = zscore(df["sent_mean"].fillna(0.0), cfg.z_window)

    # winsorize
    for c in ["alpha_trend", "alpha_reversion", "alpha_sentiment"]:
        df[c] = df[c].clip(-3, 3)

    print("[STEP 10] Built alpha sleeves: trend, reversion, sentiment")
    return df


def step11_regime(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    vol = df["vol_21d"].fillna(method="ffill").fillna(0.0)
    df["regime"] = np.where(vol > cfg.vol_regime_threshold, "risk_off", "risk_on")
    print("[STEP 11] Built regime labels: risk_on / risk_off")
    return df


def step12_gating(df: pd.DataFrame) -> pd.DataFrame:
    # risk_on => trend-heavy; risk_off => reversion-heavy; sentiment always blended
    alpha = np.zeros(len(df), dtype=float)

    risk_on = (df["regime"] == "risk_on").to_numpy()
    risk_off = ~risk_on

    alpha[risk_on] = (
        0.6 * df.loc[risk_on, "alpha_trend"].to_numpy()
        + 0.2 * df.loc[risk_on, "alpha_reversion"].to_numpy()
        + 0.2 * df.loc[risk_on, "alpha_sentiment"].to_numpy()
    )
    alpha[risk_off] = (
        0.2 * df.loc[risk_off, "alpha_trend"].to_numpy()
        + 0.6 * df.loc[risk_off, "alpha_reversion"].to_numpy()
        + 0.2 * df.loc[risk_off, "alpha_sentiment"].to_numpy()
    )

    df["alpha_combined"] = pd.Series(alpha, index=df.index).clip(-3, 3)
    print("[STEP 12] Built gated ensemble alpha: alpha_combined")
    return df


def main():
    cfg = Config()
    dirs = make_dirs(cfg)

    fs_path = dirs["features_dir"] / f"{cfg.ticker}_feature_store.parquet"
    if not fs_path.exists():
        raise FileNotFoundError(f"Missing {fs_path}. Run script_03 first.")

    df = pd.read_parquet(fs_path).sort_values("date").reset_index(drop=True)

    df = step10_alpha_sleeves(df, cfg)
    df = step11_regime(df, cfg)
    df = step12_gating(df)

    out_path = dirs["features_dir"] / f"{cfg.ticker}_alphas_regime_gated.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OUTPUT] Saved -> {out_path}")

    print("\nDONE script_04. Next: script_05_portfolio_execution.py")


if __name__ == "__main__":
    main()


# In[11]:


# script_07_export_alpha_to_excel.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def resolve_base_dir() -> Path:
    """Notebook-safe + script-safe base directory resolver."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def main():
    BASE_DIR = resolve_base_dir()
    TICKER = "AAPL"

    in_path = BASE_DIR / "data" / "features" / f"{TICKER}_alphas_regime_gated.parquet"
    out_path = BASE_DIR / "data" / "features" / f"{TICKER}_alpha_results.xlsx"

    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing alpha parquet: {in_path}\n"
            "Run script_04_alphas_regime_gating.py first."
        )

    df = pd.read_parquet(in_path)

    # Select the alpha fields you care about
    cols = [
        "date",
        "alpha_trend",
        "alpha_reversion",
        "alpha_sentiment",
        "regime",
        "alpha_combined",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].copy().sort_values("date")

    # Export to Excel
    out.to_excel(out_path, index=False)

    print(f"✅ Alpha results exported to Excel: {out_path}")


if __name__ == "__main__":
    main()


# In[8]:


"""
Script: Dr. Ray Islam

Script 5/6 — Steps 13–15 (Constraints + Sizing + Execution Model)

Steps covered:
13) constraints/risk controls, 14) sizing, 15) execution model"""

# script_05_portfolio_execution.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Config:
    ticker: str = "AAPL"
    max_position: float = 1.0          # long-only cap
    max_turnover: float = 0.25         # max weekly change in weight
    min_adv_dollars: float = 5_000_000 # liquidity cap (ADV$ 20d)
    vol_target: float = 0.10           # annual vol target (single-asset sizing proxy)
    base_cost_bps: float = 10.0
    slip_bps: float = 5.0
    base_dir: Path = resolve_base_dir()


def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    features_dir = data_dir / "features"
    portfolio_dir = data_dir / "portfolio"
    for d in [features_dir, portfolio_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"features_dir": features_dir, "portfolio_dir": portfolio_dir}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def step13_constraints(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # liquidity filter: if ADV$ too low -> force weight 0
    adv_ok = df["adv_dollars_20d"].fillna(0.0) >= cfg.min_adv_dollars
    df["liquidity_ok"] = adv_ok.astype(int)
    print("[STEP 13] Liquidity constraint created: liquidity_ok")
    return df


def step14_position_sizing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Position sizing: alpha -> base weight, then vol-target adjustment (simple).
    For single asset, vol targeting means reduce weight when vol is high.
    """
    alpha = df["alpha_combined"].fillna(0.0).to_numpy()
    base_w = sigmoid(alpha) * cfg.max_position  # 0..max_position

    # vol targeting (annualize daily vol_21d)
    daily_vol = df["vol_21d"].fillna(method="ffill").fillna(0.0).to_numpy()
    ann_vol = daily_vol * np.sqrt(252)
    # avoid division by zero
    adj = np.where(ann_vol > 1e-6, cfg.vol_target / ann_vol, 0.0)
    adj = np.clip(adj, 0.0, 1.0)  # don't lever in this prototype

    w = base_w * adj

    # liquidity constraint
    w = np.where(df["liquidity_ok"].to_numpy() == 1, w, 0.0)

    df["target_weight_raw"] = w
    print("[STEP 14] Built target_weight_raw using sigmoid + vol targeting + liquidity")
    return df


def step15_execution_model(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Execution cost model fields (applied in backtest):
    - cost_bps: base + volatility-sensitive slippage proxy
    """
    daily_vol = df["vol_21d"].fillna(method="ffill").fillna(0.0).to_numpy()

    # simple volatility-sensitive slippage: add up to +10 bps in extreme vol
    vol_bump = np.clip(daily_vol / 0.03, 0.0, 1.0) * 10.0
    cost_bps = cfg.base_cost_bps + cfg.slip_bps + vol_bump

    df["exec_cost_bps"] = cost_bps
    print("[STEP 15] Built exec_cost_bps = base + slippage + vol bump")
    return df


def main():
    # Optional kill switch (prod hygiene partial)
    if os.getenv("KILL_SWITCH") == "1":
        raise SystemExit("KILL_SWITCH=1 set. Exiting safely.")

    cfg = Config()
    dirs = make_dirs(cfg)

    in_path = dirs["features_dir"] / f"{cfg.ticker}_alphas_regime_gated.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run script_04 first.")

    df = pd.read_parquet(in_path).sort_values("date").reset_index(drop=True)

    df = step13_constraints(df, cfg)
    df = step14_position_sizing(df, cfg)
    df = step15_execution_model(df, cfg)

    out_path = dirs["portfolio_dir"] / f"{cfg.ticker}_targets_with_costs.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OUTPUT] Saved -> {out_path}")

    print("\nDONE script_05. Next: script_06_backtest_report_prod.py")


if __name__ == "__main__":
    main()


# In[9]:


"""
Script: Dr. Ray Islam

Script 6/6 — Steps 16–17 (Backtest + Walk-forward Scaffold + Tear Sheet + Prod Hygiene)

Steps covered:
16) backtesting (includes walk-forward scaffold)
17) reporting + production hygiene (logging, sanity checks, kill switch)"""


# script_06_backtest_report_prod.py
from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class Config:
    ticker: str = "AAPL"
    rebalance_freq: str = "W-FRI"
    initial_capital: float = 100_000.0
    max_turnover: float = 0.25
    base_dir: Path = resolve_base_dir()

def make_dirs(cfg: Config) -> Dict[str, Path]:
    data_dir = cfg.base_dir / "data"
    portfolio_dir = data_dir / "portfolio"
    backtests_dir = data_dir / "backtests"
    reports_dir = data_dir / "reports"
    logs_dir = cfg.base_dir / "logs"
    for d in [portfolio_dir, backtests_dir, reports_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"portfolio_dir": portfolio_dir, "backtests_dir": backtests_dir, "reports_dir": reports_dir, "logs_dir": logs_dir}


def setup_logging(logs_dir: Path) -> None:
    log_path = logs_dir / "strategy_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Logging initialized -> %s", log_path)


# --------------------------
# Production hygiene checks
# --------------------------
def sanity_checks(df: pd.DataFrame) -> None:
    required = ["date", "close", "ret_1d", "target_weight_raw", "exec_cost_bps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["date"].isna().any():
        raise ValueError("date contains NaN")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("date not sorted ascending")

    if (df["close"] <= 0).any():
        raise ValueError("close has non-positive values")

    if (df["exec_cost_bps"] < 0).any():
        raise ValueError("exec_cost_bps negative")

    logging.info("Sanity checks passed.")


def apply_turnover_cap(prev_w: float, target_w: float, max_turnover: float) -> float:
    delta = target_w - prev_w
    capped_delta = float(np.clip(delta, -max_turnover, max_turnover))
    return prev_w + capped_delta


def step16_backtest_weekly(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly rebalance long-only with cash. Applies:
    - turnover cap per rebalance
    - execution cost bps based on turnover * capital
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    rb_dates = set(pd.date_range(out["date"].min(), out["date"].max(), freq=cfg.rebalance_freq).to_pydatetime().tolist())

    capital = cfg.initial_capital
    w = 0.0
    nav = []
    w_series = []
    turnover_series = []
    cost_series = []

    prev_close = None

    for i, row in out.iterrows():
        dt = row["date"].to_pydatetime()
        ret = float(row["ret_1d"])
        target_raw = float(row["target_weight_raw"])
        cost_bps = float(row["exec_cost_bps"])

        # mark-to-market
        if prev_close is not None:
            capital *= (1.0 + w * ret)

        # rebalance on schedule
        if dt in rb_dates:
            new_w = apply_turnover_cap(w, target_raw, cfg.max_turnover)
            turnover = abs(new_w - w)
            cost = (cost_bps / 10_000.0) * turnover * capital
            capital -= cost
            w = new_w
        else:
            turnover = 0.0
            cost = 0.0

        nav.append(capital)
        w_series.append(w)
        turnover_series.append(turnover)
        cost_series.append(cost)
        prev_close = float(row["close"])

    out["nav"] = nav
    out["position_weight"] = w_series
    out["turnover"] = turnover_series
    out["trade_cost"] = cost_series
    out["strategy_ret_1d"] = out["nav"].pct_change().fillna(0.0)

    logging.info("Backtest complete. Final NAV: %.2f", out["nav"].iloc[-1])
    return out


# Walk-forward scaffold: shown as structure (for ML later)
def walk_forward_splits(dates: pd.Series, train_years: float = 2.0, test_months: float = 3.0) -> list[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Produces rolling train/test splits for ML expansion.
    For now, returns splits but does not train an ML model.
    """
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    if d.empty:
        return []

    start = d.iloc[0]
    end = d.iloc[-1]

    splits = []
    cur = start + pd.DateOffset(years=int(train_years))
    while True:
        train_start = start
        train_end = cur
        test_start = cur + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=int(test_months))
        if test_end > end:
            break
        splits.append((train_start, train_end, test_start, test_end))
        cur = cur + pd.DateOffset(months=1)
    return splits


def sharpe(daily_returns: pd.Series) -> float:
    mu = float(daily_returns.mean())
    sd = float(daily_returns.std())
    if sd < 1e-12:
        return 0.0
    return math.sqrt(252) * mu / sd


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def compute_metrics(bt: pd.DataFrame) -> dict:
    rets = bt["strategy_ret_1d"]
    nav = bt["nav"]

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_return = float((1.0 + total_return) ** (252 / max(1, len(nav))) - 1.0)
    ann_vol = float(rets.std() * math.sqrt(252))

    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": float(sharpe(rets)),
        "max_drawdown": float(max_drawdown(nav)),
        "avg_turnover": float(bt["turnover"].mean()),
        "total_cost": float(bt["trade_cost"].sum()),
        "final_nav": float(nav.iloc[-1]),
        # a few extra useful ones:
        "win_rate": float((rets > 0).mean()),
        "avg_daily_return": float(rets.mean()),
    }


def step17_tearsheet_pdf(bt: pd.DataFrame, metrics: dict, out_pdf: Path) -> None:
    bt = bt.copy()
    bt["date"] = pd.to_datetime(bt["date"])

    with PdfPages(out_pdf) as pdf:
        # Equity curve
        fig = plt.figure()
        plt.plot(bt["date"], bt["nav"])
        plt.title("Equity Curve (NAV)")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        pdf.savefig(fig)
        plt.close(fig)

        # Drawdown
        fig = plt.figure()
        peak = bt["nav"].cummax()
        dd = bt["nav"] / peak - 1.0
        plt.plot(bt["date"], dd)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        pdf.savefig(fig)
        plt.close(fig)

        # Position weight
        fig = plt.figure()
        plt.plot(bt["date"], bt["position_weight"])
        plt.title("Position Weight")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        pdf.savefig(fig)
        plt.close(fig)

        # Rolling Sharpe (63d)
        fig = plt.figure()
        roll = bt["strategy_ret_1d"].rolling(63)
        roll_sh = (roll.mean() / (roll.std() + 1e-12)) * math.sqrt(252)
        plt.plot(bt["date"], roll_sh)
        plt.title("Rolling Sharpe (63d)")
        plt.xlabel("Date")
        plt.ylabel("Sharpe")
        pdf.savefig(fig)
        plt.close(fig)

        # Metrics page
        fig = plt.figure()
        plt.axis("off")
        lines = ["Performance Summary"] + [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        plt.text(0.02, 0.98, "\n".join(lines), va="top")
        pdf.savefig(fig)
        plt.close(fig)

    logging.info("Tear sheet saved -> %s", out_pdf)


def main():
    # Kill switch
    if os.getenv("KILL_SWITCH") == "1":
        raise SystemExit("KILL_SWITCH=1 set. Exiting safely.")

    cfg = Config()
    dirs = make_dirs(cfg)
    setup_logging(dirs["logs_dir"])

    in_path = dirs["portfolio_dir"] / f"{cfg.ticker}_targets_with_costs.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run script_05 first.")

    df = pd.read_parquet(in_path)
    sanity_checks(df)

    # Step 16: backtest
    bt = step16_backtest_weekly(cfg, df)
    out_bt = dirs["backtests_dir"] / f"{cfg.ticker}_backtest.csv"
    bt.to_csv(out_bt, index=False)
    logging.info("Backtest saved -> %s", out_bt)

    # Walk-forward scaffold (not training ML here yet)
    splits = walk_forward_splits(bt["date"])
    logging.info("Walk-forward splits available (scaffold): %d splits", len(splits))

    # Step 17: reporting + hygiene artifacts
    metrics = compute_metrics(bt)
    metrics_path = dirs["backtests_dir"] / f"{cfg.ticker}_metrics.json"
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
    logging.info("Metrics saved -> %s", metrics_path)

    report_pdf = dirs["reports_dir"] / f"{cfg.ticker}_tear_sheet.pdf"
    step17_tearsheet_pdf(bt, metrics, report_pdf)

    logging.info("ALL DONE (Steps 16–17).")


if __name__ == "__main__":
    main()

