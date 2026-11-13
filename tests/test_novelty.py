"""Tests for the novelty experiment's design (synthetic data, no Ollama needed)."""
import numpy as np
import pandas as pd

from src.novelty import (N_RETRIEVED, few_shot_prompt, fit_stacker, date_bootstrap, load_split,
                         retrieved_shots, stack_features, static_shots)

LABS = ["positive", "neutral", "negative"]


def synth(n=120, seed=0, tmp_path=None):
    rng = np.random.default_rng(seed)
    words = ["euro", "dollar", "yen", "pound", "ecb", "fed", "rate", "hike", "cut", "inflation", "boe", "boj"]
    rows = []
    for i in range(n):
        rows.append({"published_at": pd.Timestamp("2023-02-01") + pd.Timedelta(hours=7 * i),
                     "ticker": rng.choice(["EURUSD", "USDJPY"]),
                     "title": " ".join(rng.choice(words, 5)) + f" item{i}",
                     "true_sentiment": rng.choice(LABS).capitalize(),
                     "finbert_sentiment": rng.choice(LABS).capitalize(),
                     "finbert_sent_score": float(rng.uniform(0.3, 0.9)),
                     "llm_sentiment": rng.choice(LABS)})
    d = pd.DataFrame(rows).sample(frac=1, random_state=1)  # shuffled on disk, split must sort by time
    p = tmp_path / "labels.csv"
    d.to_csv(p, index=False)
    return load_split(p)


def test_split_is_time_ordered_disjoint_and_60_40(tmp_path):
    h = synth(tmp_path=tmp_path)
    dev, test = h[h["split"] == "dev"], h[h["split"] == "test"]
    assert len(dev) == 72 and len(test) == 48
    assert dev["published_at"].max() <= test["published_at"].min()
    assert h["published_at"].is_monotonic_increasing


def test_static_shots_are_dev_only_stratified_and_deterministic(tmp_path):
    h = synth(tmp_path=tmp_path)
    a, b = static_shots(h), static_shots(h)
    assert a == b and len(a) == 9
    assert (h.loc[a, "split"] == "dev").all()
    assert h.loc[a, "human"].value_counts().to_dict() == {"positive": 3, "neutral": 3, "negative": 3}


def test_retrieval_never_uses_test_labels_and_never_returns_itself(tmp_path):
    h = synth(tmp_path=tmp_path)
    retr = retrieved_shots(h)
    for i, shots in retr.items():
        assert len(shots) == N_RETRIEVED and len(set(shots)) == N_RETRIEVED
        assert all(h.loc[s, "split"] == "dev" for s in shots)  # only dev examples are ever shown
        assert i not in shots                                   # a dev row cannot see its own label


def test_retrieval_returns_the_most_similar_headline_last(tmp_path):
    h = synth(tmp_path=tmp_path)
    dev_i = h.index[h["split"] == "dev"][5]
    h.loc[h.index[h["split"] == "test"][0], "title"] = h.loc[dev_i, "title"] + " extra"  # near-copy of a dev row
    retr = retrieved_shots(h)
    assert retr[h.index[h["split"] == "test"][0]][-1] == dev_i


def test_prompt_contains_labeled_examples_and_the_query_but_not_its_answer(tmp_path):
    h = synth(tmp_path=tmp_path)
    shots = h.loc[static_shots(h)]
    q = h[h["split"] == "test"].iloc[0]
    p = few_shot_prompt(shots, q["ticker"], q["title"])
    assert p.count("Answer:") == 10 and p.rstrip().endswith("Answer:")
    assert all(t in p for t in shots["title"])
    assert q["title"] in p and f"Headline ({q['ticker']}): {q['title']}\nAnswer: {q['human']}" not in p


def test_stacker_trains_on_dev_only_and_predicts_valid_labels(tmp_path):
    h = synth(tmp_path=tmp_path)
    dev, test = h[h["split"] == "dev"], h[h["split"] == "test"]
    m = fit_stacker(dev, "zero_shot")
    pred = m.predict(stack_features(test, "zero_shot"))
    assert set(pred) <= set(LABS) and len(pred) == len(test)
    assert stack_features(test, "zero_shot").shape[1] == 8


def test_date_bootstrap_is_reproducible_and_brackets_the_estimate(tmp_path):
    h = synth(n=300, tmp_path=tmp_path)
    test = h[h["split"] == "test"]
    preds = {"X": test["zero_shot"], "Y": test["finbert"]}
    a1, c1 = date_bootstrap(test, preds, [("X", "Y")], n_boot=300)
    a2, c2 = date_bootstrap(test, preds, [("X", "Y")], n_boot=300)
    assert a1.equals(a2) and c1.equals(c2)
    acc = (test["zero_shot"] == test["human"]).mean()
    r = a1[a1["technique"] == "X"].iloc[0]
    assert r["acc_lo"] <= acc <= r["acc_hi"]
