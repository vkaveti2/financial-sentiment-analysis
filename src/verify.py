"""Check the regenerated results against the numbers reported in the paper.

Run AFTER `src.features`, `src.run_m4`, `src.novelty --evaluate-only` and `src.reproduce`.
Every expected value below is the number printed in the paper (rounded the way the paper rounds it).
Each line recomputes the value from the files on disk, so PASS means "this machine got the paper's number".

Run: python -m src.verify
"""
import sys

import pandas as pd

from src.models import ROOT

R = ROOT / "reports"
checks = []


def check(name: str, got, expected, ndigits: int) -> None:
    ok = round(float(got), ndigits) == round(float(expected), ndigits)
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} got {round(float(got), ndigits):<9} paper {expected}")


def section(t: str) -> None:
    print(f"\n{t}")


def main() -> int:
    raw = pd.read_csv(ROOT / "data" / "raw" / "sentiment_annotated_with_texts.csv")
    h = pd.read_csv(ROOT / "data" / "processed" / "llm_labels.csv")
    human, fin, llm = (h[c].str.lower() for c in ("true_sentiment", "finbert_sentiment", "llm_sentiment"))
    dis = fin != llm

    section("1. Data and labelers (all 2,291 headlines)")
    check("headlines in the public dataset", len(raw), 2291, 0)
    check("FinBERT agrees with human label", (fin == human).mean(), 0.562, 3)
    check("Qwen agrees with human label", (llm == human).mean(), 0.752, 3)
    check("FinBERT and Qwen disagree (headlines)", dis.sum(), 1009, 0)
    check("FinBERT and Qwen disagree (share)", dis.mean(), 0.440, 3)
    check("Qwen right when FinBERT agrees", (llm[~dis] == human[~dis]).mean(), 0.825, 3)
    check("Qwen right when FinBERT disagrees", (llm[dis] == human[dis]).mean(), 0.659, 3)

    section("2. Daily table")
    d = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["td"])
    e = d[d["eligible"]]
    check("eligible pair-days", len(e), 249, 0)
    check("distinct dates", e["td"].nunique(), 71, 0)
    check("share of next days that go up", e["target_up"].mean(), 0.510, 3)

    section("3. Does news predict tomorrow? (164 walk-forward test rows)")
    s = pd.read_csv(R / "m4_results" / "main_logistic_summary.csv").set_index("variant")
    pred = pd.read_csv(R / "m4_results" / "main_logistic_predictions.csv")
    check("test rows", len(pred), 164, 0)
    for v, paper in [("V0 Majority", 0.494), ("V1 Prices only", 0.518), ("V2 +FinBERT", 0.488),
                     ("V3 +LLM", 0.524), ("V4 +both", 0.506), ("V5 Full", 0.488)]:
        check(f"accuracy, {v}", s.loc[v, "accuracy"], paper, 3)
    c = pd.read_csv(R / "m4_results" / "main_logistic_comparisons.csv").set_index("comparison")
    k = "V5 Full minus V4 +both"
    check("disagreement adds (V5 minus V4), points", 100 * c.loc[k, "acc_diff"], -1.8, 1)
    check("  95% interval low", 100 * c.loc[k, "lo"], -5.9, 1)
    check("  95% interval high", 100 * c.loc[k, "hi"], 2.4, 1)

    section("4. Novel contribution (916 held-out headlines)")
    t = pd.read_csv(R / "novelty_results" / "comparison_table.csv").set_index("technique")
    for tid, name, paper in [("A0", "FinBERT", 0.547), ("A1", "Qwen zero-shot", 0.753),
                             ("A2", "Qwen few-shot, fixed examples", 0.777),
                             ("A3", "Qwen few-shot, retrieved examples", 0.809), ("A4", "stacked", 0.809)]:
        check(f"accuracy, {tid} {name}", t.loc[tid, "accuracy"], paper, 3)
    p = pd.read_csv(R / "novelty_results" / "paired_comparisons.csv").set_index("comparison")
    for k, (d_, lo, hi) in {"A3 minus A1": (5.6, 2.6, 8.5), "A3 minus A2": (3.2, 0.9, 5.3)}.items():
        check(f"{k}, points", 100 * p.loc[k, "acc_diff"], d_, 1)
        check("  95% interval low", 100 * p.loc[k, "lo"], lo, 1)
        check("  95% interval high", 100 * p.loc[k, "hi"], hi, 1)

    n_fail = len(checks) - sum(checks)
    print(f"\n{'ALL ' + str(len(checks)) + ' CHECKS PASS: this machine reproduced the paper numbers.' if not n_fail else str(n_fail) + ' of ' + str(len(checks)) + ' CHECKS FAILED: see README Troubleshooting.'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
