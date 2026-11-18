"""Download daily FX prices for the five pairs in the Fatouros dataset (OPTIONAL refresh).

You do not need this to reproduce the results. `python -m src.data` restores the frozen snapshot
(data/snapshots/prices_daily.csv, pulled 2026-10-29) that every number in the paper was computed
from. Re-downloading may return slightly different values and change results, so it only runs with
--refresh and needs `pip install -r requirements-refresh.txt`.

Run: python -m src.prices --refresh
"""
import argparse
from pathlib import Path

import pandas as pd

PAIRS = ["AUDUSD", "EURCHF", "EURUSD", "GBPUSD", "USDJPY"]
# Start early enough for 5-day / 20-day lookback features on the first headline date.
START, END = "2022-11-01", "2023-05-20"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_prices(out: Path = RAW / "prices_daily.csv") -> pd.DataFrame:
    import yfinance as yf  # optional dependency; imported here so the rest of the project does not need it

    frames = []
    for pair in PAIRS:
        df = yf.download(f"{pair}=X", start=START, end=END, interval="1d",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna().reset_index()
        df.insert(0, "ticker", pair)
        df = df.rename(columns={"Date": "date"})
        # Yahoo's row dated t holds the move FRED stamps t-1 (see reports/timing_decision.md), so
        # features must use move_date, never date. First row has no prior date and is dropped.
        df["move_date"] = df["date"].shift(1)
        df = df.dropna(subset=["move_date"])
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(out, index=False)
    return prices


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download from Yahoo (may change results)")
    if not ap.parse_args().refresh:
        print("Nothing to do: prices come from the frozen snapshot restored by `python -m src.data`.\n"
              "Only re-download if you want to test a refresh (results may differ): "
              "python -m src.prices --refresh")
        raise SystemExit(0)
    p = download_prices()
    print(p.groupby("ticker").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max")))
    print("\nWrote data/raw/prices_daily.csv. Results may now differ from the paper; "
          "run `python -m src.data` to restore the frozen snapshot.")
