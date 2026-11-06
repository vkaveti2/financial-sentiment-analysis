# Combining FinBERT and ChatGPT for Forex Prediction

DS 340W project (Harshi Chinnam, Varun Kaveti).

**Research question:** can using both FinBERT and ChatGPT sentiment on forex headlines improve next-trading-day direction prediction compared with using either model alone?

## Status

M0 to M4 are done; M5 and M6 are next. The midterm slides quote 2,291 headlines and 41.9% model disagreement from Fatouros et al. (2023). The headline count is confirmed. The 41.9% cannot be reproduced because the dataset ships no GPT-P2 labels. With a free local LLM (Qwen2.5-14B via Ollama) standing in for ChatGPT, FinBERT and the LLM disagree on 44.0% of headlines. Details in `reports/data_audit.md`; a running log for the presentation is in `reports/project_log.md`.

| Milestone | What | State |
|---|---|---|
| M0 | Repo structure, environment | done |
| M1 | Download dataset, check 2,291 / 41.9%, pull prices, label with local LLM | done |
| M2 | FinBERT + LLM labels per headline (FinBERT ships in the dataset; LLM labels done) | mostly done |
| M3 | Daily feature table with leakage tests | done (249 eligible pair-days; see `reports/features_m3.md`) |
| M4 | Six-variant ablation (walk-forward, bootstrap CIs) | done: no evidence of benefit; see `reports/m4_results.md` (plan fixed in `reports/m4_analysis_plan.md` before running) |
| M5 | Disagreement test + error analysis | todo |
| M6 | Final slides / report | todo |

## Demo (for the recording)

After setup, one command prints every number the project reports:

```
python -m src.data && python -m src.prices   # fetch inputs (first time only)
python -m src.demo                           # instant, no Ollama needed
python -m src.demo --live 5                  # also labels 5 headlines live (needs `ollama serve`)
```

It shows: FinBERT vs the local LLM against human labels (56.2% vs 75.2%), FinBERT-vs-LLM disagreement (44.0% here, compared with the 41.9% on the proposal slides, which used ChatGPT), whether disagreement signals unreliable labels, the same-day link to price moves, and the walk-forward prediction test. Tests: `python -m pytest tests`.

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
brew install ollama && ollama serve   # separate terminal
ollama pull qwen2.5:14b-instruct
python -m src.prices      # daily FX prices
python -m src.sentiment   # LLM labels (~15 min, cached)
```

## Data

One command rebuilds inputs on a fresh clone: `python -m src.data` (downloads the Zenodo file with an md5 check and restores our Qwen labels from `data/labels/llm_labels.csv`). That file is derived from the CC-BY-4.0 Zenodo dataset (Fatouros et al.), reused here with attribution.

Headlines: Fatouros et al., *Forex news annotated dataset for sentiment analysis*, Zenodo, doi:10.5281/zenodo.7976208 (5 pairs, Jan to May 2023). Check the Zenodo license before committing any data. Daily prices come from Yahoo Finance via `yfinance` (`src/prices.py`). Daily features: `python -m src.features`. Tests: `python -m pytest tests`. Full comparison: `python -m src.run_m4`. Parent-paper reproduction: `python -m src.reproduce` (results in `reports/reproduction_vs_parent_paper.md`). Yahoo's rows are dated one trading day late, so use the `move_date` column; see `reports/timing_decision.md`.

## References

- Fatouros et al. (2023). Transforming sentiment analysis in the financial domain with ChatGPT. *Machine Learning with Applications* 14, 100508.
- Moradi-Kamali et al. (2025). Market-Derived Financial Sentiment Analysis. arXiv:2502.14897.
- Dong, Fan and Peng (2024). FNSPID. KDD 2024.
