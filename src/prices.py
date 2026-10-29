"""Download and cache daily FX prices for the five pairs in the Fatouros dataset."""
from pathlib import Path

import pandas as pd
import yfinance as yf

PAIRS = ["AUDUSD", "EURCHF", "EURUSD", "GBPUSD", "USDJPY"]
# Start early enough for 5-day / 20-day lookback features on the first headline date.
START, END = "2022-11-01", "2023-05-20"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_prices(out: Path = RAW / "prices_daily.csv") -> pd.DataFrame:
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
    p = download_prices()
    print(p.groupby("ticker").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max")))
