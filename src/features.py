"""Build the daily (pair, trading date) feature table.

Timing (see reports/timing_decision.md):
- A price return is "stamped" with move_date D: roughly noon New York on the previous trading day
  to noon New York on D. Features at D may use returns stamped D or earlier.
- Headline timestamps are treated as UTC. A headline published at or after CUTOFF_HOUR (UTC) falls
  inside the next return's window, so it rolls to the next calendar day. Weekend headlines roll to
  the next trading date.
- Target: direction of the return stamped (1 + gap) trading dates after D. gap=0 is the main
  setting; gap>=1 is the leakage robustness check.

Run: python -m src.features [--cutoff 16] [--gap 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "processed" / "llm_labels.csv"
PRICES = ROOT / "data" / "raw" / "prices_daily.csv"
SCORE = {"positive": 1, "neutral": 0, "negative": -1}
CUTOFF_HOUR = 16  # = noon New York during daylight saving; conservative before March 12


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log return stamped by move_date (the Yahoo `date` column is one trading day late)."""
    p = prices.sort_values(["ticker", "move_date"]).copy()
    p["ret"] = p.groupby("ticker")["Close"].transform(lambda s: np.log(s).diff())
    return p[["ticker", "move_date", "ret"]].rename(columns={"move_date": "td"})


def assign_trading_date(ts: pd.Series, trading_dates: pd.DatetimeIndex,
                        cutoff_hour: int = CUTOFF_HOUR) -> pd.Series:
    """First trading date on or after the headline's effective calendar date (NaT if none)."""
    roll = (ts.dt.hour >= cutoff_hour).astype(int)
    eff = ts.dt.normalize() + pd.to_timedelta(roll, unit="D")
    idx = np.searchsorted(trading_dates.values, eff.values, side="left")
    ok = idx < len(trading_dates)
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    out.loc[ok] = trading_dates.values[idx[ok]]
    return out


def build_daily(headlines: pd.DataFrame, prices: pd.DataFrame,
                cutoff_hour: int = CUTOFF_HOUR, gap: int = 0) -> pd.DataFrame:
    h = headlines.copy()
    h["fin"] = h["finbert_sentiment"].str.lower().map(SCORE)
    h["llm"] = h["llm_sentiment"].str.lower().map(SCORE)
    h["differ"] = (h["fin"] != h["llm"]).astype(float)
    parts = []
    for ticker, r in price_returns(prices).groupby("ticker"):
        r = r.sort_values("td").reset_index(drop=True)
        hh = h[h["ticker"] == ticker].copy()
        hh["td"] = assign_trading_date(hh["published_at"], pd.DatetimeIndex(r["td"]), cutoff_hour)
        hh = hh.dropna(subset=["td"])
        agg = hh.groupby("td").agg(mean_finbert=("fin", "mean"), mean_llm=("llm", "mean"),
                                   disagree=("differ", "mean"), n_headlines=("fin", "size"),
                                   last_headline_ts=("published_at", "max")).reset_index()
        r = r.merge(agg, on="td", how="left")
        r["n_headlines"] = r["n_headlines"].fillna(0).astype(int)
        r["ret_1"] = r["ret"]
        r["ret_5"] = r["ret"].rolling(5).sum()
        r["vol_10"] = r["ret"].rolling(10).std()
        r["target_date"] = r["td"].shift(-(1 + gap))
        r["target_ret"] = r["ret"].shift(-(1 + gap))
        r["target_up"] = (r["target_ret"] > 0).astype(float).where(r["target_ret"].notna() & (r["target_ret"] != 0))
        parts.append(r.drop(columns="ret"))
    d = pd.concat(parts, ignore_index=True)
    d["eligible"] = ((d["n_headlines"] > 0) & d["target_up"].notna()
                     & d["ret_1"].notna() & d["ret_5"].notna() & d["vol_10"].notna())
    return d.sort_values(["td", "ticker"]).reset_index(drop=True)


def main(cutoff: int, gap: int) -> None:
    headlines = pd.read_csv(LABELS, parse_dates=["published_at"])
    prices = pd.read_csv(PRICES, parse_dates=["date", "move_date"])
    d = build_daily(headlines, prices, cutoff, gap)
    name = "daily.csv" if (cutoff, gap) == (CUTOFF_HOUR, 0) else f"daily_c{cutoff}_g{gap}.csv"
    d.to_csv(ROOT / "data" / "processed" / name, index=False)
    e = d[d["eligible"]]
    print(f"cutoff={cutoff}h UTC  gap={gap}  -> data/processed/{name}")
    print(f"headlines assigned to a trading date: {d['n_headlines'].sum()} of {len(headlines)}")
    print(f"pair-days with price rows: {len(d)}   with >=1 headline: {(d['n_headlines'] > 0).sum()}   eligible: {len(e)}")
    print("eligible rows by pair:", e["ticker"].value_counts().to_dict())
    print(f"eligible date range: {e['td'].min().date()} to {e['td'].max().date()}   target up-rate: {e['target_up'].mean():.3f}")
    print("eligible by month:", e.groupby(e["td"].dt.to_period("M")).size().to_dict())
    print(f"headlines per eligible row: median {e['n_headlines'].median():.0f}, min {e['n_headlines'].min()}, max {e['n_headlines'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=CUTOFF_HOUR)
    ap.add_argument("--gap", type=int, default=0)
    a = ap.parse_args()
    main(a.cutoff, a.gap)
