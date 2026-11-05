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

## M3: daily feature table (2026-10-29)
- Built `src/features.py`: one row per (pair, trading date) with mean FinBERT score, mean LLM score, disagreement rate, headline count, and price features. Headlines at or after 16:00 UTC (noon New York) roll to the next day so news written after a price move cannot sit beside that move.
- 8 tests pass, including: changing only future prices cannot change any feature, and headlines after the cutoff cannot leak into that day.
- Sample: 249 eligible pair-days (not the ~300 estimated), target up-rate 51.0%.
- **Finding: on corrected dates, daily sentiment has no detectable correlation with next-day returns** (FinBERT -0.02, LLM -0.06; every 95% interval includes zero; same with an extra day of gap).
- Supporting evidence for the "news reports the move" explanation: letting late-day headlines into the same day raises the LLM's correlation with that day's move from 0.10 to 0.28.
- Detail: `reports/features_m3.md`.

## M4: six-variant comparison (2026-11-05)
- Wrote and committed the analysis plan before running anything (`m4_analysis_plan.md`, commit `06310e3`), then ran it unchanged.
- Added `src/data.py` (one-command rebuild with checksum) and committed our Qwen labels so a fresh clone does not need 15 minutes of labeling.
- Walk-forward by date, 8 folds, 164 test rows, date-block bootstrap intervals, 14 tests passing.
- **Result: no variant beats the baseline or prices-only. Disagreement did not help (-1.8 points, interval [-5.9, +2.4]).** Best-looking variant (+LLM) is +3.0 points over baseline with interval [-8.4, +14.7].
- The majority baseline scored 49.4% (below 50%) because the test period was 54.9% UP and the training periods leaned DOWN. Explained in `m4_results.md`.
- Robustness (gap, no-roll, drop EURCHF, boosting) agrees: nothing above chance.
- Detail: `reports/m4_results.md`.

## Changes the slides need
| Slide | Change |
|---|---|
| 2, 3, 6, 7, 10, 12 | "ChatGPT" becomes "an open-weights LLM (Qwen2.5-14B)". Slide 3 still correctly describes the paper's ChatGPT work. |
| 9, 12 | 41.9% is the paper's figure. Our measured FinBERT vs LLM disagreement is 44.0%. Remove "completed analysis of saved predictions". |
| 10, 11 | Slide 10 needs the results table (use `m4_results.md`). Slide 11's 70/20/10 split was replaced by walk-forward: 8 folds, 164 test rows, date-level bootstrap. |
| 8, 11 | Update slide 8's timing note (it says timing needs verification: it is now verified, rule on `features_m3.md`). Mention the verified timing rule (UTC dates, Yahoo re-dating) and that the 70/20/10 split is replaced by walk-forward evaluation because the sample is small. |

## Presentation talking points so far
- We checked our own pipeline against an independent data source and caught a one-day date error before it produced fake results.
- Models disagree on 44% of headlines, and the LLM is more reliable when FinBERT agrees with it.
- On corrected dates, news sentiment does not predict next-day direction in this sample; the earlier apparent link was the date error.
- Combining the two models, and adding their disagreement, did not improve forecasts; with 164 test rows we can rule out only large effects (about 10 points or more).
- Honest limits: one open-weights LLM instead of ChatGPT, about 300 pair-days, labels framed as price impact.

## Next
M5 error analysis (where do the models disagree and who is right; does disagreement predict the size of the next move, not its direction), then M6 slides and report. Deadline: Sunday 2026-11-15.
