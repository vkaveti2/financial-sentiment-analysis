# Combining FinBERT and ChatGPT for Forex Prediction

DS 340W project (Harshi Chinnam, Varun Kaveti).

**Research question:** can using both FinBERT and ChatGPT sentiment on forex headlines improve next-trading-day direction prediction compared with using either model alone?

## Status

Early setup (M0). The pipeline below is planned, not built yet. Numbers quoted in the midterm slides (2,291 headlines, 41.9% model disagreement) come from Fatouros et al. (2023) and have **not** been reproduced in this repo yet; reproducing them is milestone M1.

| Milestone | What | State |
|---|---|---|
| M0 | Repo structure, environment | in progress |
| M1 | Download dataset, verify 2,291 / 41.9%, pull prices | todo |
| M2 | FinBERT + GPT-P2 labels per headline | todo |
| M3 | Daily feature table with leakage tests | todo |
| M4 | Six-variant ablation (walk-forward, bootstrap CIs) | todo |
| M5 | Disagreement test + error analysis | todo |
| M6 | Final slides / report | todo |

## Layout

```
legacy/      earlier Kaggle RoBERTa notebook + data.csv (general sentiment, not part of the forex pipeline)
src/         data, sentiment, features, splits, models, evaluate
notebooks/   EDA and final figures
data/        raw/ and processed/ (gitignored; re-download via src/data.py)
reports/     tables and figures
tests/       leakage, alignment, and split tests
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY only if re-running GPT labels
```

## Data

Headlines: Fatouros et al., *Forex news annotated dataset for sentiment analysis*, Zenodo, doi:10.5281/zenodo.7976208 (5 pairs, Jan to May 2023). Check the Zenodo license before committing any data. Prices come from a separate source chosen in M1.

## References

- Fatouros et al. (2023). Transforming sentiment analysis in the financial domain with ChatGPT. *Machine Learning with Applications* 14, 100508.
- Moradi-Kamali et al. (2025). Market-Derived Financial Sentiment Analysis. arXiv:2502.14897.
- Dong, Fan and Peng (2024). FNSPID. KDD 2024.
