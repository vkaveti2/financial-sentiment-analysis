# M4 results: does combining FinBERT and the LLM help predict next-day direction?

Design followed `m4_analysis_plan.md` exactly (committed as `06310e3` before any model was run). Reproduce with `python -m src.run_m4`; raw outputs are in `reports/m4_results/`. 14 tests pass (`python -m pytest tests`).

## Bottom line
**No evidence that any version, including the full model with disagreement, predicts next-day direction better than price features alone or than the baseline.** Every accuracy interval contains 50%, every pre-specified comparison interval contains zero, and the disagreement features did not help (if anything they slightly hurt). This is the outcome the plan labeled "no evidence": a valid negative result, with limited power to detect small effects.

## Setup in one paragraph
249 (pair, date) rows over 71 dates. Walk-forward evaluation: train on all earlier dates, test on the next 5 dates, 8 folds, 164 pooled test rows over 41 dates. Rows whose answer lands on or after a test block are removed from training. Logistic regression with strength chosen only on training data. Intervals are 95% date-block bootstraps (whole dates resampled, 2,000 times).

## Main result (logistic regression, 16:00 UTC cutoff, gap 0)
Accuracy with 95% interval; test up-rate is 54.9%.

| Variant | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| V0 Majority baseline | 0.494 [0.401, 0.587] | 0.480 | 0.500 |
| V1 Prices only | 0.518 [0.432, 0.596] | 0.513 | 0.517 [0.404, 0.627] |
| V2 + FinBERT | 0.488 [0.405, 0.567] | 0.480 | 0.506 [0.399, 0.609] |
| V3 + LLM | 0.524 [0.438, 0.604] | 0.519 | 0.533 [0.421, 0.640] |
| V4 + both | 0.506 [0.421, 0.585] | 0.499 | 0.518 [0.413, 0.622] |
| V5 Full (+ disagreement) | 0.488 [0.401, 0.568] | 0.484 | 0.496 [0.391, 0.601] |

Pre-specified comparisons (accuracy difference, 95% interval):

| Comparison | Difference | Interval |
|---|---|---|
| C2, key test: V5 Full minus V4 + both | -0.018 | [-0.059, +0.024] |
| V4 + both minus V1 Prices only | -0.012 | [-0.053, +0.025] |
| V5 Full minus V1 Prices only | -0.030 | [-0.071, +0.012] |
| V1 Prices only minus baseline | +0.024 | [-0.094, +0.145] |
| V3 + LLM minus baseline (best-looking) | +0.030 | [-0.084, +0.147] |
| V4 + both minus baseline | +0.012 | [-0.101, +0.123] |
| V2 + FinBERT / V5 Full minus baseline | -0.006 / -0.006 | both include 0 |

- V3 (+LLM) has the highest accuracy, but the gain over the baseline is 3.0 points with an interval from -8.4 to +14.7. With five variants compared, a best-of-five number like this is expected by chance.
- **Disagreement (the project's key idea) did not help**: V5 minus V4 is -1.8 points, interval includes zero.

## Why the baseline scores below 50%
The baseline predicts the more common direction in the training dates. Training periods were close to 50/50 and mostly leaned DOWN, while the test period was 54.9% UP, so it was wrong in the folds where the market turned. A benchmark of "always UP" would have scored 54.9%, but that is only known in hindsight, so it is not a fair baseline. Nothing, including any model, reached 54.9%. Read "beats the baseline" as weak evidence either way: the baseline is itself unstable here.

| Fold starts | Train up-rate | Baseline says | Test up-rate | Baseline accuracy |
|---|---|---|---|---|
| Mar 9 | 0.432 | DOWN | 0.636 | 0.364 |
| Mar 16 | 0.451 | DOWN | 0.571 | 0.429 |
| Mar 23 | 0.504 | UP | 0.591 | 0.591 |
| Mar 30 | 0.493 | DOWN | 0.364 | 0.636 |
| Apr 6 | 0.491 | DOWN | 0.571 | 0.429 |
| Apr 13 | 0.495 | DOWN | 0.545 | 0.455 |
| Apr 20 | 0.497 | DOWN | 0.550 | 0.450 |
| Apr 27 | 0.500 | UP | 0.560 | 0.560 |

(This table was looked at after the main run, as a diagnostic. It changes no conclusion.)

## Robustness runs (all pre-specified)
| Run | What changed | Result |
|---|---|---|
| R1: gap 1 | Headlines must predict the move one extra day later | Best variant V4 at 0.506 [0.416, 0.601]; V4 minus baseline +0.037 [-0.038, +0.118]. Nothing separable from chance. Full model is again not better than V4 (-0.024, [-0.053, 0.000]). |
| R2: no roll | Late-day headlines sit beside the day's own move (leakage demo) | Every variant's accuracy rises by 2.5 to 6.5 points versus the main run (prices only 0.518 to 0.560, +FinBERT 0.488 to 0.553), yet all intervals still include zero and no sentiment variant beats prices only (V4 minus V1 is -1.9 points). Prices-only has identical features in both runs, so its 4-point jump comes only from one extra row shifting the fold boundaries. That shows how fragile single accuracy numbers are at this sample size. |
| R3: drop EURCHF | Pair with unverified dates removed (216 rows, 113 test rows) | No variant above 0.558. V5 minus V4 is -0.035 [-0.069, -0.009]: the one interval that excludes zero, in the direction of the full model being worse. |
| Boosting (exploratory) | Gradient boosting instead of logistic | Accuracy 0.451 to 0.518, no variant beats baseline, full model worst at 0.451. |

R3's V5-minus-V4 interval excluding zero is one interval out of many (about 30 across the tables) and is exploratory. It is consistent with the simple explanation that three extra features add noise on about 200 rows.

## What the data could and could not detect
- Intervals on accuracy are about +/-9 points wide (+/-12 on differences against the baseline). Only improvements of roughly 10 points or more over the baseline could have been seen. Effects of 1 to 3 points, typical in real forecasting, are invisible at this sample size.
- Gains against price-only (V4 minus V1) are measured more tightly, within about +/-4 points, and the estimate is slightly negative.
- So the supportable statement is: "we found no evidence of a benefit, and we can rule out a large one," not "news cannot help."

## Findings usable in the presentation
1. News sentiment from either model, alone or combined, did not improve next-day direction prediction in this sample.
2. Model disagreement did not add information for prediction, though it does track label reliability (LLM agrees with the human label 82.5% of the time when FinBERT agrees with it, 65.9% when not; see `data_audit.md`).
3. The earlier apparent link between news and returns came from a one-day date error, and from late-day headlines describing moves that already happened. The leakage tests and the gap/no-roll runs document this.
4. The negative result is specific: 249 rows, 71 dates, one 4-month window, one open-weights LLM (not ChatGPT) with one untuned prompt.

## Limits
- 71 dates in a single regime (Jan to May 2023); pairs are correlated, which date-level bootstrapping accounts for but cannot fix.
- Yahoo's bar window is undocumented; UTC and 16:00 are assumptions. EURCHF dates are unchecked.
- Qwen2.5-14B replaces ChatGPT, so the result says nothing directly about ChatGPT.
- Many intervals were examined; no correction for multiple comparisons beyond the pre-specified list.
