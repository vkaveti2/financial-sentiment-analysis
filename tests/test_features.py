"""Alignment and leakage tests for src/features.py, on small synthetic data."""
import numpy as np
import pandas as pd

from src.features import CUTOFF_HOUR, assign_trading_date, build_daily

# 30 weekday trading dates starting Mon 2023-02-06
DATES = pd.bdate_range("2023-02-06", periods=30)


def make_prices(seed=0):
    rng = np.random.default_rng(seed)
    close = 1.07 * np.exp(np.cumsum(rng.normal(0, 0.005, len(DATES) + 1)))
    # Yahoo-style: `date` is one trading day after `move_date`; the first row's move_date is earlier.
    move_dates = pd.bdate_range(end=DATES[-1], periods=len(DATES) + 1)
    return pd.DataFrame({"ticker": "EURUSD", "date": move_dates + pd.offsets.BDay(1),
                         "move_date": move_dates, "Close": close})


def make_headlines(stamps, fin="Positive", llm="Neutral"):
    return pd.DataFrame({"published_at": pd.to_datetime(stamps), "ticker": "EURUSD",
                         "finbert_sentiment": fin, "llm_sentiment": llm})


def test_cutoff_rolls_late_headlines_to_next_day():
    ts = pd.Series(pd.to_datetime(["2023-02-08 15:59", "2023-02-08 16:00", "2023-02-08 23:30"]))
    out = assign_trading_date(ts, DATES, CUTOFF_HOUR)
    assert list(out) == [pd.Timestamp("2023-02-08"), pd.Timestamp("2023-02-09"), pd.Timestamp("2023-02-09")]


def test_weekend_and_friday_evening_roll_to_monday():
    ts = pd.Series(pd.to_datetime(["2023-02-10 20:00", "2023-02-11 09:00", "2023-02-12 22:30", "2023-02-13 08:00"]))
    out = assign_trading_date(ts, DATES, CUTOFF_HOUR)
    assert all(o == pd.Timestamp("2023-02-13") for o in out)


def test_headline_after_last_trading_date_is_dropped():
    ts = pd.Series(pd.to_datetime(["2023-03-30 20:00"]))
    assert assign_trading_date(ts, DATES, CUTOFF_HOUR).isna().all()


def test_target_date_is_strictly_after_feature_date_and_headlines():
    h = make_headlines(["2023-02-14 09:00", "2023-02-15 10:00", "2023-02-15 18:00", "2023-02-17 12:00"])
    for gap in (0, 1, 2):
        full = build_daily(h, make_prices(), gap=gap)
        expected = full.groupby("ticker")["td"].shift(-(1 + gap))  # computed before dropping tail rows
        assert full["target_date"].equals(expected) or (
            full["target_date"].fillna(pd.Timestamp(0)) == expected.fillna(pd.Timestamp(0))).all()
        d = full.dropna(subset=["target_date"])
        assert (d["target_date"] > d["td"]).all()
        used = d.dropna(subset=["last_headline_ts"])
        # every headline used was published before the cutoff on the feature date
        assert (used["last_headline_ts"] < used["td"] + pd.Timedelta(hours=CUTOFF_HOUR)).all()


def test_features_do_not_change_when_future_prices_change():
    h = make_headlines(["2023-02-14 09:00", "2023-02-15 10:00"])
    base = build_daily(h, make_prices())
    p2 = make_prices()
    p2.loc[p2["move_date"] > pd.Timestamp("2023-02-15"), "Close"] *= 1.05  # alter the future only
    alt = build_daily(h, p2)
    feats = ["ret_1", "ret_5", "vol_10", "mean_finbert", "mean_llm", "disagree", "n_headlines"]
    day = pd.Timestamp("2023-02-15")
    a, b = base[base["td"] == day].iloc[0], alt[alt["td"] == day].iloc[0]
    for c in feats:
        assert (pd.isna(a[c]) and pd.isna(b[c])) or a[c] == b[c], c
    # the target on 2023-02-15 *is* the next day, and the shock shifts it
    assert a["target_ret"] != b["target_ret"]


def test_features_do_not_change_when_late_headlines_change():
    early = ["2023-02-15 10:00"]
    base = build_daily(make_headlines(early), make_prices())
    alt = build_daily(make_headlines(early + ["2023-02-15 17:00", "2023-02-15 22:00"], fin="Negative"), make_prices())
    day = pd.Timestamp("2023-02-15")
    a, b = base[base["td"] == day].iloc[0], alt[alt["td"] == day].iloc[0]
    assert a["n_headlines"] == b["n_headlines"] == 1
    assert a["mean_finbert"] == 1  # the Negative headlines published after the cutoff did not leak in
    # ...and they land on the next trading day instead
    assert alt[alt["td"] == pd.Timestamp("2023-02-16")].iloc[0]["n_headlines"] == 2


def test_disagreement_rate():
    h = pd.concat([make_headlines(["2023-02-15 08:00"], "Positive", "Positive"),
                   make_headlines(["2023-02-15 09:00"], "Positive", "Negative"),
                   make_headlines(["2023-02-15 10:00"], "Neutral", "Positive"),
                   make_headlines(["2023-02-15 11:00"], "Negative", "Negative")])
    row = build_daily(h, make_prices()).query("td == '2023-02-15'").iloc[0]
    assert row["n_headlines"] == 4 and row["disagree"] == 0.5
    # finbert: +1 +1 0 -1 = 1 -> 0.25 ; llm: +1 -1 +1 -1 = 0 -> 0.0
    assert row["mean_finbert"] == 0.25 and row["mean_llm"] == 0.0


def test_eligible_requires_news_and_lookback_and_target():
    h = make_headlines(["2023-02-07 10:00", "2023-02-20 10:00", "2023-03-22 10:00"])
    d = build_daily(h, make_prices())
    elig = d[d["eligible"]]
    assert list(elig["td"]) == [pd.Timestamp("2023-02-20")]  # 02-07: no lookback; 03-22: no target
