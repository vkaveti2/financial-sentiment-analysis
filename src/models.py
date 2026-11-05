"""Six variants, walk-forward folds, and fitting. Design is fixed in reports/m4_analysis_plan.md."""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
C_GRID = [0.01, 0.1, 1, 10]
PRICE = ["ret_1", "ret_5", "vol_10"]
VARIANTS: Dict[str, List[str]] = {
    "V0 Majority": [],
    "V1 Prices only": PRICE,
    "V2 +FinBERT": PRICE + ["mean_finbert"],
    "V3 +LLM": PRICE + ["mean_llm"],
    "V4 +both": PRICE + ["mean_finbert", "mean_llm"],
    "V5 Full": PRICE + ["mean_finbert", "mean_llm", "disagree", "dis_x_finbert", "dis_x_llm"],
}


def load_eligible(path: Path, drop_tickers: Tuple[str, ...] = ()) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["td", "target_date"])
    d = d[d["eligible"]].copy()
    d = d[~d["ticker"].isin(drop_tickers)]
    d["dis_x_finbert"] = d["disagree"] * d["mean_finbert"]
    d["dis_x_llm"] = d["disagree"] * d["mean_llm"]
    return d.sort_values(["td", "ticker"]).reset_index(drop=True)


def make_folds(dates, min_train: int = 30, block: int = 5):
    """Expanding-window folds over sorted distinct dates. A trailing block shorter than `block`
    is merged into the previous one."""
    dates = sorted(pd.to_datetime(list(dates)))
    n = len(dates)
    starts = list(range(min_train, n, block))
    if len(starts) > 1 and n - starts[-1] < block:
        starts.pop()
    folds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n
        folds.append((dates[:s], dates[s:e]))
    return folds


def train_rows(d: pd.DataFrame, train_dates, test_start) -> pd.DataFrame:
    """Rows on training dates whose answer is known before the test block starts (embargo)."""
    return d[d["td"].isin(train_dates) & (d["target_date"] < test_start)]


def _lr(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=SEED))


def select_C(tr: pd.DataFrame, cols: List[str]) -> float:
    """Pick C by validation log loss on the last 20% of training dates; tie goes to smaller C."""
    dates = sorted(tr["td"].unique())
    k = max(1, int(np.ceil(0.2 * len(dates))))
    val_start = dates[-k]
    inner = tr[(tr["td"] < val_start) & (tr["target_date"] < val_start)]
    val = tr[tr["td"] >= val_start]
    if inner["target_up"].nunique() < 2 or val.empty:
        return 1.0
    best, best_loss = 1.0, np.inf
    for C in C_GRID:
        m = _lr(C).fit(inner[cols], inner["target_up"])
        loss = log_loss(val["target_up"], m.predict_proba(val[cols])[:, 1], labels=[0.0, 1.0])
        if loss < best_loss - 1e-12:
            best, best_loss = C, loss
    return best


def fit_predict(model: str, variant: str, tr: pd.DataFrame, te: pd.DataFrame):
    """Return (hard prediction, score for class 'up') for the test rows."""
    cols = VARIANTS[variant]
    y = tr["target_up"]
    if not cols:  # majority baseline: constant score, so AUC is 0.5 by construction
        up = float(y.mean() >= 0.5)
        return np.full(len(te), up), np.full(len(te), 0.5), None
    if model == "logistic":
        C = select_C(tr, cols)
        m = _lr(C).fit(tr[cols], y)
    else:
        C = None
        m = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                       random_state=SEED).fit(tr[cols], y)
    p = m.predict_proba(te[cols])[:, 1]
    return (p >= 0.5).astype(float), p, C


def walk_forward(d: pd.DataFrame, model: str = "logistic", min_train: int = 30, block: int = 5):
    """Run all variants over all folds. Returns (test rows, {variant: (pred, score)}, chosen C list)."""
    test_parts, preds, scores, chosen = [], {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}, []
    for train_dates, test_dates in make_folds(d["td"].unique(), min_train, block):
        te = d[d["td"].isin(test_dates)]
        tr = train_rows(d, train_dates, test_dates[0])
        test_parts.append(te)
        for v in VARIANTS:
            pr, sc, C = fit_predict(model, v, tr, te)
            preds[v].append(pr)
            scores[v].append(sc)
            if C is not None:
                chosen.append({"variant": v, "first_test_date": test_dates[0], "C": C})
    test = pd.concat(test_parts, ignore_index=True)
    out = {v: (np.concatenate(preds[v]), np.concatenate(scores[v])) for v in VARIANTS}
    return test, out, chosen
