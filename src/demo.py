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

EARLIER_DISAGREEMENT = (960, 2291)  # midterm-slide figure; could not be reproduced (no ChatGPT labels in the public data)
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
    sk, sn = EARLIER_DISAGREEMENT
    print(f"Earlier midterm-slide figure:            {sk:,} of {sn:,} headlines differ = {100 * sk / sn:.1f}%")
    print("That figure could not be reproduced: it needs ChatGPT labels, and the public data has none.")
    print(f"Ours is {100 * k / n - 100 * sk / sn:+.1f} points away. Different LLM, so this is a comparison, not a replication.")
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

    nov = ROOT / "reports" / "novelty_results" / "comparison_table.csv"
    if nov.exists():
        heading("7. OUR NOVEL CONTRIBUTION: HOW THE LOCAL LLM IS USED (916 held-out later headlines)")
        names = {"A0": "A0 FinBERT (parent's classical model)", "A1": "A1 Qwen zero-shot (parent's style)",
                 "A2": "A2 Qwen few-shot, 9 fixed examples", "A3": "A3 Qwen few-shot, RETRIEVED examples (novel)",
                 "A4": "A4 Stacked FinBERT + Qwen (tested, no gain)"}
        t = pd.read_csv(nov)
        t["technique"] = t["technique"].map(names)
        t["accuracy (95% interval)"] = t.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}]", axis=1)
        print(t[["technique", "accuracy (95% interval)", "weighted_f1", "s_mae"]].round(3).to_string(index=False))
        c = pd.read_csv(ROOT / "reports" / "novelty_results" / "paired_comparisons.csv")
        c = c[c["comparison"].isin(["A2 minus A1", "A3 minus A1", "A3 minus A2", "A4 minus A3"])]
        print("\nPaired accuracy differences (95% interval):")
        for _, r in c.iterrows():
            print(f"  {r['comparison']:12s} {r['acc_diff']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]")

    if live:
        heading(f"8. LIVE: {live} random headlines labeled by Ollama right now")
        import urllib.error

        from src.sentiment import MODEL, ask
        sample = h.sample(live, random_state=7)
        t0 = time.time()
        try:
            for _, r in sample.iterrows():
                lab = ask(r["ticker"], r["title"])
                same = "same as saved label" if lab == r["llm_sentiment"] else "differs from saved label (rare; every result uses the saved labels)"
                print(f"\n{r['ticker']}: {r['title'][:90]}\n  human={r['true_sentiment'].lower():8s} FinBERT={r['finbert_sentiment'].lower():8s} "
                      f"{MODEL}={lab:8s} ({same})")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"\nCould not reach the local model ({e}).\n"
                  "Start Ollama (open the Ollama app, or run `ollama serve` in a second Terminal tab), "
                  f"check `ollama list` shows {MODEL}, then rerun this command.\n"
                  "Everything above this line does not need Ollama and has already printed.")
            raise SystemExit(1)
        print(f"\n{live} headlines in {time.time() - t0:.1f}s ({(time.time() - t0) / live:.2f}s each)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0)
    main(ap.parse_args().live)
