"""Timing checks that decide the M3 alignment rule.

1. Are Yahoo daily bars date-aligned? Compare Yahoo daily returns with the Federal Reserve's noon
   New York rates from FRED (an independent source) at lags -1, 0, +1. Highest correlation at lag 0
   means the bar dates line up; a peak at +1 or -1 means Yahoo's dates are shifted by a day.
2. Which timezone are the headline timestamps in? Articles sometimes state an event time in GMT
   ("... at 13:30 GMT"). Under the right timezone the stated time is usually just after the
   publish time, so test how often that holds under several timezone hypotheses.

Run: python -m src.timing_check
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# FRED series quoted the same way as our pair (USD per EUR = EURUSD, JPY per USD = USDJPY).
# FRED has no EURCHF series, so that pair is not checked.
FRED = {"EURUSD": "DEXUSEU", "GBPUSD": "DEXUSUK", "AUDUSD": "DEXUSAL", "USDJPY": "DEXJPUS"}


def fred_series(code: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd=2022-10-01&coed=2023-05-31"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def yahoo_vs_fred() -> pd.DataFrame:
    prices = pd.read_csv(RAW / "prices_daily.csv", parse_dates=["date"])
    rows = []
    for pair, code in FRED.items():
        y = prices[prices.ticker == pair].set_index("date")["Close"]
        fr = np.log(fred_series(code)).reindex(y.index).diff()
        yr = np.log(y).diff()
        row = {"pair": pair}
        for k in (-1, 0, 1):
            joined = pd.concat([yr, fr.shift(-k)], axis=1).dropna()
            row[f"lag{k:+d}"] = round(joined.corr().iloc[0, 1], 3)
        row["n"] = len(joined)
        rows.append(row)
    return pd.DataFrame(rows)


def stated_gmt_check() -> pd.DataFrame:
    d = pd.read_csv(RAW / "sentiment_annotated_with_texts.csv", parse_dates=["published_at"])
    t = d["text"].fillna("").str.extract(r"\b(\d{1,2}):(\d{2})\s*GMT\b").dropna().astype(int)
    pub = d.loc[t.index, "published_at"]
    stated_min = t[0] * 60 + t[1]
    pub_min = pub.dt.hour * 60 + pub.dt.minute
    rows = []
    for label, offset_h in [("UTC-5 (US Eastern)", -5), ("UTC-4 (US Eastern DST)", -4),
                            ("UTC+0 (GMT/UTC)", 0), ("UTC+1 (CET)", 1), ("UTC+2", 2)]:
        # If timestamps are in UTC+offset, publish time in GMT is pub - offset.
        delta = (stated_min - (pub_min - offset_h * 60)) % 1440
        delta = np.where(delta > 720, delta - 1440, delta)
        rows.append({"timestamps assumed to be": label, "n": len(t),
                     "stated time 0-12h after publish": round(float(((delta >= -15) & (delta <= 720)).mean()), 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Yahoo daily return vs FRED noon-NY return (correlation by lag of FRED vs Yahoo date):")
    print(yahoo_vs_fred().to_string(index=False))
    print("\nShare of 'at HH:MM GMT' mentions that fall just after the publish time, per timezone hypothesis:")
    print(stated_gmt_check().to_string(index=False))
