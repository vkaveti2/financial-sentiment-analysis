"""Novel contribution: compare ways of using a lightweight local LLM (Ollama) to label forex headlines.

Design is fixed in reports/novelty_plan.md (committed before any of this ran).

Techniques (all scored on the TEST split, the last 40% of headlines by time):
  A0  FinBERT                              labels shipped with the dataset
  A1  Qwen zero-shot                       the parent paper's style: one prompt, no examples
  A2  Qwen few-shot, static                9 fixed labeled dev examples in the prompt
  A3  Qwen few-shot, retrieved             the 6 most similar labeled dev headlines in the prompt  <-- NOVEL
  A4  Stacked FinBERT + best Qwen variant  logistic regression fit on dev only (tested; added nothing)

Outcome (reports/novelty_results.md): A3 beat zero-shot by 5.6 points; A4 reproduced A3 exactly.
Labels for A2/A3 are cached in data/labels/novelty_cache.jsonl, so `--evaluate-only` needs no Ollama.

Run: python -m src.novelty [--limit N] [--evaluate-only]
"""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.sentiment import LABELS, MODEL, PROMPT, URL

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "labels" / "llm_labels.csv"
CACHE = ROOT / "cache" / "novelty.jsonl"
SHARED_CACHE = ROOT / "data" / "labels" / "novelty_cache.jsonl"  # committed copy of the same labels
OUT = ROOT / "reports" / "novelty_results"
DEV_FRAC = 0.6
N_STATIC = 9          # 3 per class
N_RETRIEVED = 6
SEED = 42
B = 2000
SCORE = {"positive": 1, "neutral": 0, "negative": -1}

INSTRUCTION = (
    "You are a forex trading expert. For each news headline, decide whether it is likely to push "
    "the named currency pair's exchange rate UP in the short term (positive), DOWN (negative), or "
    "have no clear short-term effect (neutral). Judge the impact on that pair, not the tone of the "
    "words. Labeled examples from human annotators are given first. Answer with exactly one word: "
    "positive, negative, or neutral."
)


# ----------------------------------------------------------------------------------------------
# data and split
# ----------------------------------------------------------------------------------------------
def load_split(path: Path = SRC) -> pd.DataFrame:
    """All headlines sorted by time; first 60% = dev, last 40% = test."""
    h = pd.read_csv(path, parse_dates=["published_at"])
    h = h.sort_values("published_at", kind="stable").reset_index(drop=True)
    k = int(round(DEV_FRAC * len(h)))
    h["split"] = ["dev"] * k + ["test"] * (len(h) - k)
    h["human"] = h["true_sentiment"].str.lower()
    h["finbert"] = h["finbert_sentiment"].str.lower()
    h["zero_shot"] = h["llm_sentiment"].str.lower()
    return h


def static_shots(h: pd.DataFrame, n_per_class: int = 3, seed: int = SEED) -> List[int]:
    """Stratified random dev examples (row indices), fixed by seed."""
    dev = h[h["split"] == "dev"]
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    for lab in ("positive", "neutral", "negative"):
        pool = dev.index[dev["human"] == lab].to_numpy()
        picks += list(rng.choice(pool, size=n_per_class, replace=False))
    order = rng.permutation(len(picks))  # mix classes so order carries no signal
    return [int(picks[i]) for i in order]


def retrieved_shots(h: pd.DataFrame, k: int = N_RETRIEVED) -> Dict[int, List[int]]:
    """For every row, the k most similar DEV headlines (TF-IDF cosine), most similar LAST.

    A dev row never retrieves itself. Test rows can only retrieve dev rows, so no test label is used.
    """
    dev_idx = h.index[h["split"] == "dev"].to_numpy()
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    dev_m = vec.fit_transform(h.loc[dev_idx, "title"])  # fit on dev titles only
    all_m = vec.transform(h["title"])
    sims = (all_m @ dev_m.T).toarray()
    pos_in_dev = {int(i): j for j, i in enumerate(dev_idx)}
    for i, j in pos_in_dev.items():
        sims[i, j] = -np.inf  # exclude self
    out: Dict[int, List[int]] = {}
    for i in range(len(h)):
        top = np.argsort(-sims[i], kind="stable")[:k]
        out[i] = [int(dev_idx[j]) for j in top[::-1]]  # most similar last
    return out


# ----------------------------------------------------------------------------------------------
# prompts and the LLM call
# ----------------------------------------------------------------------------------------------
def fmt_example(r: pd.Series) -> str:
    return f"Headline ({r['ticker']}): {r['title']}\nAnswer: {r['human']}"


def few_shot_prompt(shots: pd.DataFrame, pair: str, title: str) -> str:
    ex = "\n\n".join(fmt_example(r) for _, r in shots.iterrows())
    return f"{INSTRUCTION}\n\n{ex}\n\nHeadline ({pair}): {title}\nAnswer:"


def _key(prompt: str) -> str:
    return hashlib.sha1(f"{MODEL}\n{prompt}".encode()).hexdigest()


def ask_prompt(prompt: str) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": SEED, "num_predict": 4}})
    req = urllib.request.Request(URL, data=body.encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.loads(r.read())["response"].strip().lower()
    word = text.split()[0].strip(".,!\"'") if text else ""
    return word if word in LABELS else "invalid"


def label_all(prompts: Dict[Tuple[str, int], str], workers: int = 4) -> Dict[Tuple[str, int], str]:
    """Label every prompt, resuming from the on-disk cache. Keys are (technique, row index)."""
    CACHE.parent.mkdir(exist_ok=True)
    if not CACHE.exists() and SHARED_CACHE.exists():
        CACHE.write_text(SHARED_CACHE.read_text())
    done: Dict[str, str] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec["label"]
    todo = [(k, p) for k, p in prompts.items() if _key(p) not in done]
    print(f"  {len(prompts) - len(todo)} cached, {len(todo)} to label")
    with CACHE.open("a") as f, ThreadPoolExecutor(workers) as ex:
        for n, ((k, p), lab) in enumerate(zip(todo, ex.map(lambda t: ask_prompt(t[1]), todo)), 1):
            done[_key(p)] = lab
            f.write(json.dumps({"key": _key(p), "label": lab}) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"    {n}/{len(todo)}", flush=True)
    return {k: done[_key(p)] for k, p in prompts.items()}


def build_prompts(h: pd.DataFrame, limit: Optional[int] = None):
    shots = static_shots(h)
    retr = retrieved_shots(h)
    rows = range(len(h)) if limit is None else list(range(limit)) + list(range(len(h) - limit, len(h)))
    a2, a3 = {}, {}
    static_df = h.loc[shots]
    for i in rows:
        r = h.loc[i]
        a2[("A2", i)] = few_shot_prompt(static_df, r["ticker"], r["title"])
        a3[("A3", i)] = few_shot_prompt(h.loc[retr[i]], r["ticker"], r["title"])
    return {**a2, **a3}, shots, retr


# ----------------------------------------------------------------------------------------------
# stacking (A4)
# ----------------------------------------------------------------------------------------------
def stack_features(d: pd.DataFrame, qwen_col: str) -> np.ndarray:
    cols = []
    for lab in ("positive", "neutral", "negative"):
        cols.append((d["finbert"] == lab).astype(float))
        cols.append((d[qwen_col] == lab).astype(float))
    cols.append(d["finbert_sent_score"].astype(float))
    cols.append((d["finbert"] == d[qwen_col]).astype(float))  # agree flag; disagreement is its complement
    return np.column_stack(cols)


def fit_stacker(dev: pd.DataFrame, qwen_col: str, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(stack_features(dev, qwen_col), dev["human"])


# ----------------------------------------------------------------------------------------------
# metrics and date-block bootstrap
# ----------------------------------------------------------------------------------------------
def metrics(y: pd.Series, p: pd.Series) -> Dict[str, float]:
    s_mae = (y.map(SCORE) - p.map(SCORE).fillna(0)).abs().mean()  # invalid answers score 0 (neutral)
    return {"accuracy": accuracy_score(y, p),
            "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "s_mae": s_mae, "n_invalid": int((~p.isin(LABELS)).sum())}


def date_bootstrap(test: pd.DataFrame, preds: Dict[str, pd.Series], comparisons: List[Tuple[str, str]],
                   n_boot: int = B, seed: int = SEED):
    y = test["human"].values
    days = test["published_at"].dt.normalize().values
    uniq = np.unique(days)
    idx = [np.flatnonzero(days == d) for d in uniq]
    n_d = np.array([len(i) for i in idx], dtype=float)
    correct = {k: np.array([(p.values[i] == y[i]).sum() for i in idx], dtype=float) for k, p in preds.items()}
    s = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    tot = n_d[s].sum(1)
    acc = {k: correct[k][s].sum(1) / tot for k in preds}
    rows = [{"technique": k, "acc_lo": np.percentile(a, 2.5), "acc_hi": np.percentile(a, 97.5)} for k, a in acc.items()]
    comp = []
    for a, b in comparisons:
        diff = acc[a] - acc[b]
        point = (preds[a].values == y).mean() - (preds[b].values == y).mean()
        comp.append({"comparison": f"{a} minus {b}", "acc_diff": point, "lo": np.percentile(diff, 2.5),
                     "hi": np.percentile(diff, 97.5), "share_boot_gt0": float((diff > 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(comp)


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def evaluate(h: pd.DataFrame, a2: Dict[int, str], a3: Dict[int, str], shots: List[int],
             retr: Dict[int, List[int]]) -> None:
    h = h.copy()
    h["A0"] = h["finbert"]
    h["A1"] = h["zero_shot"]
    h["A2"] = pd.Series(a2)
    h["A3"] = pd.Series(a3)
    dev = h[(h["split"] == "dev") & ~h.index.isin(shots)]  # the 9 static shots saw their own answer
    test = h[h["split"] == "test"]

    dev_acc = {t: (dev[c] == dev["human"]).mean() for t, c in [("A1", "A1"), ("A2", "A2"), ("A3", "A3")]}
    best_q = max(dev_acc, key=dev_acc.get)
    stacker = fit_stacker(dev, best_q)
    h["A4"] = pd.Series(stacker.predict(stack_features(h, best_q)), index=h.index)
    test = h[h["split"] == "test"]
    print(f"dev accuracy (excluding the 9 static shots): " + ", ".join(f"{k} {v:.3f}" for k, v in dev_acc.items())
          + f"  -> A4 uses {best_q}")

    techs = ["A0", "A1", "A2", "A3", "A4"]
    preds = {t: test[t] for t in techs}
    best_single = max(["A0", "A1", "A2", "A3"], key=lambda t: {"A0": (dev["finbert"] == dev["human"]).mean(), **dev_acc}[t])
    comps = [("A2", "A1"), ("A3", "A1"), ("A3", "A2"), ("A4", best_single)] + [(t, "A0") for t in techs[1:]] + [("A3", "A4")]
    comps = list(dict.fromkeys(comps))
    ci, comp = date_bootstrap(test, preds, comps)
    tab = pd.DataFrame([{"technique": t, **metrics(test["human"], test[t])} for t in techs]).merge(ci, on="technique")
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "comparison_table.csv", index=False)
    comp.to_csv(OUT / "paired_comparisons.csv", index=False)
    test[["published_at", "ticker", "title", "human"] + techs].to_csv(OUT / "test_predictions.csv", index=False)

    # exploratory diagnostics (labeled as such in the write-up)
    dup = test["title"].isin(h.loc[h["split"] == "dev", "title"])
    diag = {"best_single_by_dev": best_single, "dev_accuracy": {k: float(v) for k, v in dev_acc.items()},
            "test_rows": int(len(test)), "test_dates": int(test["published_at"].dt.normalize().nunique()),
            "test_rows_with_exact_title_in_dev": int(dup.sum()),
            "A3_acc_without_exact_title_dups": float((test.loc[~dup, "A3"] == test.loc[~dup, "human"]).mean()),
            "A1_acc_without_exact_title_dups": float((test.loc[~dup, "A1"] == test.loc[~dup, "human"]).mean()),
            "A4_stacker_coef_shape": list(stacker.coef_.shape)}
    (OUT / "diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))
    by_pair = test.groupby("ticker")[techs + ["human"]].apply(
        lambda g: pd.Series({t: (g[t] == g["human"]).mean() for t in techs} | {"n": len(g)}))
    by_pair.to_csv(OUT / "accuracy_by_pair.csv")
    dis = test["A0"] != test["A1"]
    by_agree = pd.DataFrame({t: [(test.loc[~dis, t] == test.loc[~dis, "human"]).mean(),
                                 (test.loc[dis, t] == test.loc[dis, "human"]).mean()] for t in techs},
                            index=[f"FinBERT and Qwen agree (n={int((~dis).sum())})",
                                   f"FinBERT and Qwen disagree (n={int(dis.sum())})"])
    by_agree.to_csv(OUT / "accuracy_by_agreement.csv")

    t = tab.copy()
    t["accuracy (95% CI)"] = t.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}]", axis=1)
    print(f"\nTEST split: {len(test)} headlines, {diag['test_dates']} dates")
    print(t[["technique", "accuracy (95% CI)", "weighted_f1", "macro_f1", "s_mae", "n_invalid"]].round(3).to_string(index=False))
    print("\nPaired accuracy differences (date-block bootstrap):")
    print(comp.round(3).to_string(index=False))
    print("\nAccuracy by pair:\n", by_pair.round(3).to_string())
    print("\nAccuracy by agreement:\n", by_agree.round(3).to_string())
    print("\nDiagnostics:", json.dumps(diag, indent=1, default=str))


def main(limit: Optional[int], evaluate_only: bool) -> None:
    h = load_split()
    prompts, shots, retr = build_prompts(h, limit)
    print(f"split: dev {int((h['split'] == 'dev').sum())}, test {int((h['split'] == 'test').sum())}; "
          f"static shots (row idx): {shots}")
    labels = label_all(prompts) if not evaluate_only else label_all(prompts, workers=1)
    if limit:
        print("smoke test only; first labels:", {k: v for k, v in list(labels.items())[:6]})
        return
    a2 = {i: labels[("A2", i)] for i in range(len(h))}
    a3 = {i: labels[("A3", i)] for i in range(len(h))}
    evaluate(h, a2, a3, shots, retr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="smoke test: label only the first and last N rows")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    main(a.limit, a.evaluate_only)
