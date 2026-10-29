# Project log (for the final presentation)

Plain-language record of what was done, what was found, and what changed from the midterm slides. Newest entries last. Detail lives in `data_audit.md` and `timing_decision.md`.

## M0: Setup (2026-10-28)
- Moved the unrelated Kaggle RoBERTa notebook and `data.csv` into `legacy/` (history kept).
- Added `src/`, `tests/`, `notebooks/`, `data/`, `reports/`, `.gitignore` (secrets, raw data, caches), `requirements.txt`.
- Found that the slides' "completed analysis" had not been done in code. The 2,291 headlines and 41.9% disagreement came from the Fatouros paper, so the README says so.

## M1: Data and labels (2026-10-28 to 10-29)
- Dataset: Zenodo 7976208, CC-BY-4.0, 2,291 headlines, 5 pairs, 2023-01-12 to 2023-05-04. FinBERT labels ship with it; GPT labels and prices do not.
- Prices: daily OHLC from Yahoo Finance via `yfinance` (`src/prices.py`).
- ChatGPT replacement: no API access, so `qwen2.5:14b-instruct` runs locally through Ollama (`src/sentiment.py`, temperature 0, seed 42, prompt written by us, no tuning against human labels).
- Results on all 2,291 headlines: LLM vs human 75.2%, FinBERT vs human 56.2%, LLM vs FinBERT disagreement 44.0% (1,009 headlines). Human-label accuracy of the LLM is 82.5% when the models agree and 65.9% when they disagree.
- Sample is small: 293 pair-days with news; January has only 12 headlines.

## M3 prerequisite: timing (2026-10-29)
- Compared Yahoo daily returns with FRED noon-New-York rates. Yahoo's rows are dated one trading day late (sign match 83 to 85% after re-dating, chance level before). Fixed with a `move_date` column in `src/prices.py`.
- Headline timestamps are consistent with UTC, not US Eastern (83% vs 23 to 28% on 47 articles that state a GMT event time).
- An earlier exploratory correlation looked "predictive" but was contemporaneous because of the date error. It is not to be quoted.
- Late-day headlines correlate most strongly with the move, which suggests news partly reports price moves that already happened. This is a hypothesis to test in M4/M5, and is the main threat to claiming real forecasting skill.

## Changes the slides need
| Slide | Change |
|---|---|
| 2, 3, 6, 7, 10, 12 | "ChatGPT" becomes "an open-weights LLM (Qwen2.5-14B)". Slide 3 still correctly describes the paper's ChatGPT work. |
| 9, 12 | 41.9% is the paper's figure. Our measured FinBERT vs LLM disagreement is 44.0%. Remove "completed analysis of saved predictions". |
| 8, 11 | Mention the verified timing rule (UTC dates, Yahoo re-dating) and that the 70/20/10 split is replaced by walk-forward evaluation because the sample is small. |

## Presentation talking points so far
- We checked our own pipeline against an independent data source and caught a one-day date error before it produced fake results.
- Models disagree on 44% of headlines, and the LLM is more reliable when FinBERT agrees with it.
- Honest limits: one open-weights LLM instead of ChatGPT, about 300 pair-days, labels framed as price impact.

## Next
M3 daily feature table with leakage tests, then the six-variant comparison (M4) and error analysis (M5).
