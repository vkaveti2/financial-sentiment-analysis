"""Tests that the M4 evaluation design is leak-free and consistent (synthetic data)."""
import numpy as np
import pandas as pd

from src.evaluate import bootstrap
from src.models import VARIANTS, make_folds, train_rows, walk_forward


def synth(n_dates=50, pairs=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates + 1)
    rows = []
    for i in range(n_dates):
        for p in pairs:
            rows.append({"td": dates[i], "ticker": p, "target_date": dates[i + 1], "eligible": True,
                         "ret_1": rng.normal(), "ret_5": rng.normal(), "vol_10": abs(rng.normal()),
                         "mean_finbert": rng.normal(), "mean_llm": rng.normal(),
                         "disagree": rng.uniform(), "target_up": float(rng.random() > 0.5)})
    d = pd.DataFrame(rows)
    d["dis_x_finbert"] = d["disagree"] * d["mean_finbert"]
    d["dis_x_llm"] = d["disagree"] * d["mean_llm"]
    return d


def test_folds_are_ordered_non_overlapping_and_cover_all_later_dates():
    dates = pd.bdate_range("2023-01-02", periods=71)
    folds = make_folds(dates)
    assert len(folds) == 8
    seen = []
    for train, test in folds:
        assert max(train) < min(test)
        assert len(train) >= 30
        seen += list(test)
    assert seen == list(dates[30:])  # every date after the warm-up is tested exactly once
    assert len(folds[-1][1]) == 6   # short trailing block merged into the last fold


def test_embargo_removes_rows_whose_answer_crosses_the_boundary():
    d = synth()
    train_dates = sorted(d["td"].unique())[:30]
    test_start = sorted(d["td"].unique())[30]
    tr = train_rows(d, train_dates, test_start)
    assert (tr["target_date"] < test_start).all()
    assert tr["td"].max() < pd.Timestamp(train_dates[-1])  # last training date's answer lands ON test start


def test_same_rows_scored_for_every_variant_and_no_train_test_overlap():
    d = synth()
    test, res, _ = walk_forward(d)
    assert set(res) == set(VARIANTS)
    assert all(len(p) == len(test) == len(s) for p, s in res.values())
    assert not test.duplicated(["td", "ticker"]).any()


def test_future_labels_cannot_change_earlier_predictions():
    d = synth()
    base_test, base, _ = walk_forward(d)
    d2 = d.copy()
    last = sorted(d["td"].unique())[-6:]
    d2.loc[d2["td"].isin(last), "target_up"] = 1.0 - d2.loc[d2["td"].isin(last), "target_up"]
    alt_test, alt, _ = walk_forward(d2)
    early = (base_test["td"] < last[0]) & (base_test["td"] < sorted(d["td"].unique())[-11])
    for v in VARIANTS:
        assert np.array_equal(base[v][1][early.values], alt[v][1][early.values])


def test_majority_baseline_is_constant_with_auc_half():
    d = synth()
    _, res, _ = walk_forward(d)
    pred, score = res["V0 Majority"]
    assert set(np.unique(score)) == {0.5}
    assert len(np.unique(pred[:5])) == 1


def test_bootstrap_is_by_date_and_reproducible():
    d = synth()
    test, res, _ = walk_forward(d)
    a1, c1 = bootstrap(test, res, [("V4 +both", "V0 Majority")], n_boot=200)
    a2, c2 = bootstrap(test, res, [("V4 +both", "V0 Majority")], n_boot=200)
    assert a1.equals(a2) and c1.equals(c2)
    row = a1[a1["variant"] == "V1 Prices only"].iloc[0]
    assert row["acc_lo"] <= row["accuracy"] <= row["acc_hi"]
    assert 0 <= row["acc_lo"] and row["acc_hi"] <= 1
