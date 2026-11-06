"""One-command demo for the recording: FinBERT vs a lightweight local LLM (Ollama) on forex headlines.

Prints the quantitative results the project reports, from the committed labels and saved results.
Run: python -m src.demo            (instant, no Ollama needed)
     python -m src.demo --live 5   (also labels 5 random headlines live with Ollama, ~2 seconds)
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.models import ROOT
from src.reproduce import classification, market_link

SLIDE_DISAGREEMENT = (960, 2291)  # figure on the proposal slides (paper setup, ChatGPT GPT-P2)
SCORE = {"positive": 1, "neutral": 0, "negative": -1}


def heading(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main(live: int) -> None:
    h = pd.read_csv(ROOT / "data" / "processed" / "llm_labels.csv", parse_dates=["published_at"])
    prices = pd.read_csv(ROOT / "data" / "raw" / "prices_daily.csv", parse_dates=["date", "move_date"])
    human, fin, llm = (h[c].str.lower() for c in ("true_sentiment", "finbert_sentiment", "llm_sentiment"))

    heading("1. DATA")
    print(f"{len(h):,} forex headlines, pairs: {', '.join(sorted(h['ticker'].unique()))}")
    print(f"{h['published_at'].min().date()} to {h['published_at'].max().date()}, each with a human sentiment label")
    print("Models: FinBERT (finance-tuned classifier) vs Qwen2.5-14B-Instruct run locally through Ollama")

    heading("2. HOW WELL EACH MODEL MATCHES THE HUMAN LABELS")
    c = classification(h).rename(columns={"model": "model"})
    print(c.round(3).to_string(index=False))

    heading("3. HOW OFTEN THE TWO MODELS DISAGREE")
    dis = fin != llm
    n, k = len(h), int(dis.sum())
    print(f"Our result (FinBERT vs Qwen via Ollama): {k:,} of {n:,} headlines differ = {100 * k / n:.1f}%")
    sk, sn = SLIDE_DISAGREEMENT
    print(f"Figure on the proposal slides:           {sk:,} of {sn:,} headlines differ = {100 * sk / sn:.1f}%  (paper setup, ChatGPT)")
    print(f"Difference: {100 * k / n - 100 * sk / sn:+.1f} points. Different LLM, so this is a comparison, not a replication.")
    print("\nWhere they disagree (rows = FinBERT, columns = Qwen):")
    print(pd.crosstab(fin.rename("FinBERT"), llm.rename("Qwen")).to_string())

    heading("4. DOES DISAGREEMENT SIGNAL UNRELIABLE LABELS?")
    print(f"Qwen matches the human label {100 * (llm[~dis] == human[~dis]).mean():.1f}% of the time when FinBERT agrees ({(~dis).sum():,} headlines)")
    print(f"Qwen matches the human label {100 * (llm[dis] == human[dis]).mean():.1f}% of the time when FinBERT disagrees ({dis.sum():,} headlines)")
    print(f"On the disagreements: FinBERT is right {100 * (fin[dis] == human[dis]).mean():.1f}%, Qwen is right {100 * (llm[dis] == human[dis]).mean():.1f}%")

    heading("5. SAME-DAY LINK TO PRICE MOVES (daily summed sentiment vs daily return)")
    m = market_link(h, prices)
    m = m[m["score"] != "human"].copy()
    m["pearson"] = m["pearson"].round(3)
    m["directional_acc"] = m["directional_acc"].round(3)
    print(m[["score", "n_days", "pearson", "directional_acc"]].to_string(index=False))
    print("directional_acc = share of days where the sign of the sentiment score matches the sign of the price move")

    heading("6. DOES IT PREDICT TOMORROW? (walk-forward test, 164 test rows)")
    s = pd.read_csv(ROOT / "reports" / "m4_results" / "main_logistic_summary.csv")
    s["accuracy (95% interval)"] = s.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}]", axis=1)
    print(s[["variant", "accuracy (95% interval)"]].to_string(index=False))
    print("Every interval includes 50%: no evidence that news sentiment, alone or combined, predicts next-day direction here.")

    if live:
        heading(f"7. LIVE: {live} random headlines labeled by Ollama right now")
        from src.sentiment import MODEL, ask
        sample = h.sample(live, random_state=7)
        t0 = time.time()
        for _, r in sample.iterrows():
            lab = ask(r["ticker"], r["title"])
            print(f"\n{r['ticker']}: {r['title'][:90]}\n  human={r['true_sentiment'].lower():8s} FinBERT={r['finbert_sentiment'].lower():8s} {MODEL}={lab:8s}"
                  f" (matches saved label: {lab == r['llm_sentiment']})")
        print(f"\n{live} headlines in {time.time() - t0:.1f}s ({(time.time() - t0) / live:.2f}s each)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0)
    main(ap.parse_args().live)
