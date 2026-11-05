"""Reproduce the parent paper's (Fatouros et al. 2023) two analyses on our data.

1. Classification: how well FinBERT and the LLM match the human sentiment labels
   (accuracy, macro precision/recall/F1, mean absolute error on the -1/0/+1 scale).
2. Market link: per pair, sum all sentiment scores within one day, then correlate that daily score
   with the pair's daily return (Pearson) and count how often its sign matches the return's sign.
   This is a SAME-DAY comparison, as in the paper, so it describes association, not prediction.

The paper's LLM is ChatGPT (GPT-P2); ours is Qwen2.5-14B. Run: python -m src.reproduce
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models import ROOT

SCORE = {"positive": 1, "neutral": 0, "negative": -1}
SOURCES = {"human": "true_sentiment", "FinBERT": "finbert_sentiment", "LLM (Qwen)": "llm_sentiment"}


def classification(h: pd.DataFrame) -> pd.DataFrame:
    y = h["true_sentiment"].str.lower()
    rows = []
    for name in ("FinBERT", "LLM (Qwen)"):
        p = h[SOURCES[name]].str.lower()
        # The paper's recall equals its accuracy, which is the signature of support-weighted averaging.
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        mae = (y.map(SCORE) - p.map(SCORE)).abs().mean()
        rows.append({"model": name, "accuracy": accuracy_score(y, p), "precision": pr,
                     "recall": rc, "f1": f1, "s_mae": mae})
    return pd.DataFrame(rows)


def market_link(h: pd.DataFrame, prices: pd.DataFrame, by_pair: bool = False) -> pd.DataFrame:
    p = prices.sort_values(["ticker", "move_date"]).copy()
    p["ret"] = p.groupby("ticker")["Close"].transform(lambda s: np.log(s).diff())
    p = p.rename(columns={"move_date": "day"})[["ticker", "day", "ret"]].dropna()
    d = h.copy()
    d["day"] = d["published_at"].dt.normalize()
    for name, col in SOURCES.items():
        d[name] = d[col].str.lower().map(SCORE)
    daily = d.groupby(["ticker", "day"])[list(SOURCES)].sum().reset_index().merge(p, on=["ticker", "day"])
    groups = daily.groupby("ticker") if by_pair else [("all pairs", daily)]
    rows = []
    for key, g in groups:
        for name in SOURCES:
            nz = g[g[name] != 0]
            rows.append({"pair": key, "score": name, "n_days": len(g),
                         "pearson": g[name].corr(g["ret"]),
                         "directional_acc": (np.sign(nz[name]) == np.sign(nz["ret"])).mean(),
                         "n_nonzero_days": len(nz)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    h = pd.read_csv(ROOT / "data" / "processed" / "llm_labels.csv", parse_dates=["published_at"])
    prices = pd.read_csv(ROOT / "data" / "raw" / "prices_daily.csv", parse_dates=["date", "move_date"])
    print(f"Headlines: {len(h)}\n\nClassification vs human labels")
    print(classification(h).round(3).to_string(index=False))
    print("\nSame-day market link (daily summed sentiment vs daily return)")
    print(market_link(h, prices).round(3).to_string(index=False))
    print("\nBy pair")
    print(market_link(h, prices, by_pair=True).round(3).to_string(index=False))
