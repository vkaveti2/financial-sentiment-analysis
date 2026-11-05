# M4 analysis plan (written and committed BEFORE any model was run)

Purpose: fix every design choice in advance so the final comparison cannot be tuned after seeing results. Anything not listed here that is added later must be labeled "exploratory" in the write-up. The git commit time of this file is the record.

## Question
Does adding FinBERT sentiment, LLM sentiment, both, or both plus their disagreement improve next-trading-day direction prediction over price features alone and over a majority-class baseline?

## Data
`data/processed/daily.csv`, rows with `eligible == True`: 249 (pair, date) rows over 71 distinct dates, up-rate 51.0%. Target `target_up`. Main setting: 16:00 UTC cutoff, gap 0.

## Variants (all fit and scored on exactly the same rows)
| # | Name | Features |
|---|---|---|
| 0 | Majority baseline | none; predicts the most common direction in the training rows |
| 1 | Prices only | ret_1, ret_5, vol_10 |
| 2 | Prices + FinBERT | 1 + mean_finbert |
| 3 | Prices + LLM | 1 + mean_llm |
| 4 | Prices + both | 1 + mean_finbert + mean_llm |
| 5 | Full | 4 + disagree + disagree×mean_finbert + disagree×mean_llm |

`n_headlines` is not used. Features are standardized using training rows only.

## Models
- Primary: L2 logistic regression. Regularization C chosen from {0.01, 0.1, 1, 10} using only the training rows of each fold (last 20% of the training dates held out as validation, then refit on all training rows).
- Secondary (reported separately, not used for the main conclusion): gradient boosting with 50 trees, depth 2, learning rate 0.1, no tuning.
- Fixed random seed 42.

## Evaluation: walk-forward by date
- Sort the 71 dates. First 30 dates are the minimum training window. Then test blocks of 5 consecutive dates, training on all earlier dates (expanding window). About 8 folds, about 160 test rows.
- Splits are by **date**, never by row, so all pairs on one date are in the same fold.
- Embargo: drop any training row whose `target_date` is on or after the first date of the test block (slide 11's rule that a future answer must not cross a boundary).
- Predictions from all folds are pooled and scored once.
- This replaces slide 11's 70/20/10 split, which would leave about 25 test rows.

## Metrics
Accuracy (primary), macro-F1, ROC-AUC, on pooled test predictions.

## Uncertainty
Block bootstrap over **test dates** (2,000 resamples, seed 42): resample whole dates with all their pairs, because pairs move together on a date. 95% percentile intervals. For comparisons, the paired difference in accuracy on the same resampled dates.

## Pre-specified comparisons (the only confirmatory ones)
- C1: each of variants 1 to 5 against the majority baseline (accuracy difference).
- C2 (key test): variant 5 against variant 4, i.e. does disagreement add anything.
- Because five variants are compared against the baseline, a single interval excluding zero would be weak evidence; we will say so. Everything else (per-pair, per-month, secondary model) is exploratory.

## Robustness runs (reported regardless of outcome)
- R1: gap = 1 (`daily_c16_g1.csv`): headlines from date D must predict the move one extra day later. Real forecasting skill should survive; leakage should not.
- R2: no roll (`--cutoff 24`), where late-day headlines sit beside the day's own move. Reported as a leakage demonstration, not as a result to rely on.
- R3: drop EURCHF (its dates are unverified against FRED).

## What counts as each outcome
- "Combining helps": variant 4 or 5 beats both the baseline and variant 1 with a paired interval that excludes zero, and the gain survives R1 and R3.
- "No evidence": intervals include zero. This is a valid result and will be reported as such, with the interval widths, so the reader can see what size of effect we could not have detected.

## Known limits (stated up front)
- 71 dates, one 4-month window, 5 correlated pairs: low power. With about 160 test rows, accuracy is uncertain by roughly +/- 8 points.
- One open-weights LLM (Qwen2.5-14B), one prompt, not tuned.
- Yahoo bar window undocumented; UTC and the 16:00 cutoff are assumptions.
