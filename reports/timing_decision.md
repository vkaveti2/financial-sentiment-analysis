# Timing decision (M3 prerequisite)

Why this matters: the project predicts the *next* trading day's direction. If a price row is dated one day off, a "next-day" target silently becomes the same-day move, and news that reacts to that move looks predictive. Two checks were run (`src/timing_check.py`; independent price source is the Federal Reserve's noon New York rates from FRED).

## Finding 1: Yahoo daily rows are dated one trading day late

Correlation between Yahoo daily returns and FRED noon-NY returns, by how FRED is shifted (n = 130 days per pair):

| Pair | FRED one day earlier | Same date | FRED one day later |
|---|---|---|---|
| EURUSD | **0.837** | 0.128 | -0.087 |
| GBPUSD | **0.875** | 0.118 | -0.226 |
| AUDUSD | **0.861** | 0.010 | -0.083 |
| USDJPY | **0.918** | 0.079 | -0.073 |

After re-dating each Yahoo row to the previous trading date, the sign of the daily move matches FRED on 82.3% to 85.4% of days. With Yahoo's raw dates it matches on 49.6% to 55.7%, which is chance. Spot check: the three biggest USDJPY down days are 12-21, 11-11, 01-13 by Yahoo's dates and 12-20, 11-10, 12-01 by FRED's.

- EURCHF could not be checked (FRED has no series). It comes from the same Yahoo feed, so the same re-dating is applied by assumption.
- Correlations are 0.84 to 0.92, not 1.0, because Yahoo's bar window and FRED's noon-NY snapshot are not the same clock. Yahoo does not document its cutoff, so the exact window of each bar is unknown.

**Decision:** `src/prices.py` now writes `move_date` (the previous trading date). All features and targets use `move_date`, never `date`.

## Finding 2: Headline timestamps look like GMT/UTC, not US Eastern

47 articles state an event time in GMT ("... at 13:30 GMT"). Share where the stated time falls 0 to 12 hours after the publish time, by assumed timestamp timezone:

| Assumed timezone | Share |
|---|---|
| US Eastern (UTC-5) | 23.4% |
| US Eastern DST (UTC-4) | 27.7% |
| UTC / GMT | **83.0%** |
| UTC+1 | 80.9% |
| UTC+2 | 80.9% |

US Eastern is ruled out. UTC+0 to +2 cannot be told apart with 47 samples, so we assume UTC+0 and test sensitivity. The small weekend cluster (Sunday 20:00 to 23:00, 49 headlines) fits UTC, since the FX market opens Sunday around 22:00 UTC.

## Effect on the earlier exploratory result

The earlier check (human-label sentiment vs return) looked as if sentiment correlated *more* with the "next-day" return (0.316) than the same-day return (0.256). Because of Finding 1, Yahoo's "next-day" was really the same-day move. So that number is contemporaneous correlation, not prediction, and must not be quoted as predictive. It will be recomputed on corrected dates.

Same finding, second reading: headlines published late in the day correlated most with the supposed next-day return (0.391 for 14:00 to 24:00 UTC). A likely explanation (a hypothesis, not tested) is that those headlines describe a move that has already happened. If so, sentiment partly *reflects* price, and the model's apparent skill would not be real forecasting.

## Rule for M3 (proposed, to be sensitivity-tested)

- Assign each headline to its UTC calendar date D.
- Features for pair P at date D use headlines dated D or earlier and price information through the move stamped D.
- Target: direction of the move stamped by the next trading date after D.
- Weekend headlines roll forward to the next trading date.
- Robustness run: add a one-day gap (headlines from D predict the move stamped two trading dates later) and a variant that drops headlines from the last 6 hours of D. If results only exist without the gap, they are probably leakage.
- A test in `tests/` will assert that no feature uses data dated after the prediction time, and that the target date is strictly after every headline used.

## Limits
- Yahoo's bar window is undocumented, so some hours may overlap even after re-dating.
- UTC+0 is an assumption supported by 47 articles.
- Sample is small (293 pair-days with news), so conclusions stay exploratory.
