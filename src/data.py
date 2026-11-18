"""Fetch the public inputs so a fresh clone can rebuild everything.

- Headlines: Zenodo doi:10.5281/zenodo.7976208 (CC-BY-4.0, Fatouros et al.), md5-checked.
- LLM labels: a copy of our Qwen2.5-14B labels is committed in data/labels/ (derived from the
  CC-BY dataset, attribution in README) so nobody has to re-run ~15 minutes of Ollama labeling.
- Prices: a frozen snapshot of Yahoo Finance daily bars (pulled 2026-10-29) is committed in
  data/snapshots/ and restored to data/raw/ with a checksum, so results do not depend on what
  Yahoo returns on the day you run this. `python -m src.prices --refresh` re-downloads (may differ).

Run: python -m src.data
"""
import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sentiment_annotated_with_texts.csv"
URL = "https://zenodo.org/api/records/7976208/files/sentiment_annotated_with_texts.csv/content"
MD5 = "95ba5134ca6203fee889378dcc9a42c6"
SHARED_LABELS = ROOT / "data" / "labels" / "llm_labels.csv"
LABELS = ROOT / "data" / "processed" / "llm_labels.csv"
PRICES_SNAPSHOT = ROOT / "data" / "snapshots" / "prices_daily.csv"
PRICES = ROOT / "data" / "raw" / "prices_daily.csv"
PRICES_MD5 = "9e7ac1f27d371e44ec1ef26d0bcadd4f"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_headlines() -> Path:
    if RAW.exists() and md5(RAW) == MD5:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        RAW.write_bytes(r.read())
    if md5(RAW) != MD5:
        raise RuntimeError("Zenodo file checksum mismatch; do not use this download.")
    return RAW


def restore_labels() -> Path:
    if not LABELS.exists():
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SHARED_LABELS, LABELS)
    return LABELS


def restore_prices() -> Path:
    if PRICES.exists() and md5(PRICES) == PRICES_MD5:
        return PRICES
    if md5(PRICES_SNAPSHOT) != PRICES_MD5:
        raise RuntimeError("Committed prices snapshot does not match its checksum; re-clone the repo.")
    PRICES.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PRICES_SNAPSHOT, PRICES)
    return PRICES


if __name__ == "__main__":
    print("headlines  :", download_headlines(), "(md5 OK)")
    print("llm labels :", restore_labels())
    print("prices     :", restore_prices(), "(md5 OK, frozen snapshot)")
    print("next: python -m src.features")
