"""Label forex headlines with a local LLM served by Ollama (replaces the paper's GPT-P2).

Prompt follows the Fatouros et al. idea: judge the headline's likely short-term impact on the
given pair, not its wording. Every response is cached to disk so reruns are free and repeatable.
Run: python -m src.sentiment [--limit N]
"""
import argparse
import json
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sentiment_annotated_with_texts.csv"
CACHE = ROOT / "cache" / "llm_labels.jsonl"
OUT = ROOT / "data" / "processed" / "llm_labels.csv"
MODEL = "qwen2.5:14b-instruct"
URL = "http://localhost:11434/api/generate"
LABELS = {"positive", "negative", "neutral"}

PROMPT = (
    "You are a forex trading expert. Decide whether this news headline is likely to push the "
    "{pair} exchange rate UP in the short term (positive), DOWN (negative), or have no clear "
    "short-term effect (neutral). Judge the impact on {pair}, not the tone of the words. "
    "Answer with exactly one word: positive, negative, or neutral.\n\nHeadline: {title}\nAnswer:"
)


def ask(pair: str, title: str) -> str:
    body = json.dumps({"model": MODEL, "prompt": PROMPT.format(pair=pair, title=title),
                       "stream": False, "options": {"temperature": 0, "seed": 42, "num_predict": 4}})
    req = urllib.request.Request(URL, data=body.encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read())["response"].strip().lower()
    word = text.split()[0].strip(".,!\"'") if text else ""
    return word if word in LABELS else "invalid"


def main(limit: Optional[int] = None) -> None:
    df = pd.read_csv(RAW)
    if limit:
        df = df.head(limit)
    CACHE.parent.mkdir(exist_ok=True)
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["model"], rec["ticker"], rec["title"])] = rec["label"]
    with CACHE.open("a") as f:
        for _, row in df.iterrows():
            key = (MODEL, row.ticker, row.title)
            if key not in done:
                done[key] = ask(row.ticker, row.title)
                f.write(json.dumps({"model": MODEL, "ticker": row.ticker, "title": row.title,
                                    "label": done[key]}) + "\n")
                f.flush()
    df["llm_sentiment"] = [done[(MODEL, r.ticker, r.title)] for r in df.itertuples()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["published_at", "ticker", "title", "true_sentiment", "finbert_sentiment",
        "finbert_sent_score", "llm_sentiment"]].to_csv(OUT, index=False)
    print(df.llm_sentiment.value_counts().to_dict())
    print("invalid:", (df.llm_sentiment == "invalid").sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
