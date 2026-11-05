"""Run the M4 comparison exactly as specified in reports/m4_analysis_plan.md.

Run: python -m src.run_m4   (writes reports/m4_results/*.csv)
"""
import subprocess
import sys

import pandas as pd

from src.evaluate import bootstrap
from src.models import ROOT, VARIANTS, load_eligible, make_folds, walk_forward

OUT = ROOT / "reports" / "m4_results"
PROC = ROOT / "data" / "processed"
V = list(VARIANTS)
COMPARISONS = [(v, V[0]) for v in V[1:]] + [(V[5], V[4]), (V[4], V[1]), (V[5], V[1])]
SETTINGS = {  # name -> (features file, cutoff, gap, dropped tickers)
    "main": ("daily.csv", 16, 0, ()),
    "R1_gap1": ("daily_c16_g1.csv", 16, 1, ()),
    "R2_noroll": ("daily_c24_g0.csv", 24, 0, ()),
    "R3_no_EURCHF": ("daily.csv", 16, 0, ("EURCHF",)),
}


def ensure_features(fname: str, cutoff: int, gap: int) -> None:
    if not (PROC / fname).exists():
        subprocess.run([sys.executable, "-m", "src.features", "--cutoff", str(cutoff), "--gap", str(gap)],
                       check=True, cwd=ROOT)


def run(name: str, model: str = "logistic") -> None:
    fname, cutoff, gap, drop = SETTINGS[name]
    ensure_features(fname, cutoff, gap)
    d = load_eligible(PROC / fname, drop)
    test, results, chosen = walk_forward(d, model)
    summary, comp = bootstrap(test, results, COMPARISONS)
    tag = f"{name}_{model}"
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / f"{tag}_summary.csv", index=False)
    comp.to_csv(OUT / f"{tag}_comparisons.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / f"{tag}_chosen_C.csv", index=False)
    pred = test[["td", "ticker", "target_up"]].copy()
    for v in V:
        pred[v + " pred"], pred[v + " score"] = results[v]
    pred.to_csv(OUT / f"{tag}_predictions.csv", index=False)
    nf = len(make_folds(d["td"].unique()))
    print(f"\n=== {tag}: {len(d)} rows, {d['td'].nunique()} dates, {nf} folds, "
          f"{len(test)} test rows, {test['td'].nunique()} test dates, test up-rate {test['target_up'].mean():.3f}")
    s = summary.copy()
    s["accuracy"] = s.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}]", axis=1)
    s["macro_f1"] = s.apply(lambda r: f"{r.macro_f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]", axis=1)
    s["auc"] = s.apply(lambda r: f"{r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]", axis=1)
    print(s[["variant", "accuracy", "macro_f1", "auc"]].to_string(index=False))
    print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    for setting in SETTINGS:
        run(setting)
    run("main", model="boosting")  # secondary model, exploratory only
