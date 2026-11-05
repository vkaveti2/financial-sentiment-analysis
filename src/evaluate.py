"""Metrics and date-level block bootstrap (resample whole dates; pairs on one date move together)."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

B = 2000
SEED = 42


def point_metrics(y, pred, score) -> Dict[str, float]:
    auc = roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "auc": auc}


def _date_index(test: pd.DataFrame):
    dates = np.sort(test["td"].unique())
    return dates, [np.flatnonzero((test["td"] == d).values) for d in dates]


def bootstrap(test: pd.DataFrame, results: Dict[str, Tuple[np.ndarray, np.ndarray]],
              comparisons: List[Tuple[str, str]], n_boot: int = B, seed: int = SEED):
    """95% date-block bootstrap intervals for every variant and paired accuracy differences."""
    y = test["target_up"].values
    dates, idx = _date_index(test)
    D = len(dates)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, D, size=(n_boot, D))
    n_d = np.array([len(i) for i in idx], dtype=float)
    correct = {v: np.array([(results[v][0][i] == y[i]).sum() for i in idx], dtype=float) for v in results}
    tot = n_d[samples].sum(1)
    acc = {v: correct[v][samples].sum(1) / tot for v in results}

    rows = []
    for v, (pred, score) in results.items():
        pm = point_metrics(y, pred, score)
        f1s, aucs = [], []
        for s in samples:
            ii = np.concatenate([idx[j] for j in s])
            f1s.append(f1_score(y[ii], pred[ii], average="macro", zero_division=0))
            aucs.append(roc_auc_score(y[ii], score[ii]) if len(np.unique(y[ii])) > 1 else np.nan)
        rows.append({"variant": v, **pm,
                     "acc_lo": np.percentile(acc[v], 2.5), "acc_hi": np.percentile(acc[v], 97.5),
                     "f1_lo": np.nanpercentile(f1s, 2.5), "f1_hi": np.nanpercentile(f1s, 97.5),
                     "auc_lo": np.nanpercentile(aucs, 2.5), "auc_hi": np.nanpercentile(aucs, 97.5)})
    summary = pd.DataFrame(rows)

    comp = []
    for a, b in comparisons:
        diff = acc[a] - acc[b]
        point = (results[a][0] == y).mean() - (results[b][0] == y).mean()
        comp.append({"comparison": f"{a} minus {b}", "acc_diff": point,
                     "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5),
                     "share_boot_diff_gt0": float((diff > 0).mean())})
    return summary, pd.DataFrame(comp)
