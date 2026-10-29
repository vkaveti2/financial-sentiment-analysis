# M3: daily feature table

Code: `src/features.py`. Tests: `tests/test_features.py` (8 tests, all pass). Output: `data/processed/daily.csv` (gitignored; rebuild with `python -m src.features`).

## What one row is
One (pair, trading date) with these inputs, all known by the end of that date:

| Feature | Meaning |
|---|---|
| `mean_finbert`, `mean_llm` | Average of +1 / 0 / -1 labels over that day's headlines, kept separate per model |
| `disagree` | Share of that day's headlines where FinBERT and the LLM gave different labels |
| `n_headlines` | Number of headlines assigned to the day |
| `ret_1`, `ret_5`, `vol_10` | 1-day return, 5-day return sum, 10-day return standard deviation (all stamped by `move_date`) |

Target: whether the return stamped on the next trading date is positive (`target_up`). Days with an exactly zero target return are dropped.

## Timing rule as built
- Headline timestamps are treated as UTC. A headline published at or after **16:00 UTC** rolls to the next day, because 16:00 UTC is noon New York (daylight saving) and the returns are noon-to-noon. Without the roll, a headline written after the day's price move would sit on the same row as that move.
- Weekend and Friday-evening headlines roll to Monday.
- A row is eligible only if it has at least one headline, a target, and enough price history for all features.

## Tests (what each one protects)
1. Cutoff boundary: 15:59 stays, 16:00 rolls.
2. Weekend and Friday evening land on Monday; headlines after the last price date are dropped.
3. Target date is strictly after the feature date for gaps of 0, 1 and 2, and every headline used was published before the cutoff on its feature date.
4. Changing only *future* prices changes the target but none of the features.
5. Changing only *late* headlines (after the cutoff) leaves that day's features unchanged and moves them to the next day.
6. Disagreement rate and mean scores are computed correctly on a hand-checked case.
7. Eligibility drops rows with no lookback, no target, or no news.

## Resulting sample (main setting: 16:00 UTC cutoff, gap 0)
- All 2,291 headlines are assigned to a trading date.
- **249 eligible pair-days**, split by pair: USDJPY 55, GBPUSD 55, AUDUSD 54, EURUSD 52, EURCHF 33.
- By month: Jan 7, Feb 51, Mar 101, Apr 73, May 17. Median 9 headlines per row.
- Target up-rate **51.0%**, so the majority-class baseline scores about 51%.
- The old plan's 70/20/10 split would leave about 25 test rows. M4 uses walk-forward evaluation instead.

## First look: does sentiment line up with the next-day move?
Correlation of the day's mean sentiment with the target return, n = 249, with 95% bootstrap intervals (2,000 resamples):

| Setting | FinBERT | LLM |
|---|---|---|
| Main (cutoff 16h, gap 0) | -0.020 [-0.13, +0.09] | -0.058 [-0.17, +0.04] |
| One extra day of gap | -0.084 [-0.19, +0.03] | -0.102 [-0.21, +0.02] |
| No roll at all (cutoff 24h) | -0.035 [-0.14, +0.06] | -0.079 [-0.18, +0.02] |

Every interval includes zero. **On these corrected dates, daily news sentiment shows no usable link to the next day's direction.** This replaces the earlier exploratory correlations (up to +0.32), which were same-day moves mislabeled as next-day (see `timing_decision.md`).

The same-day relationship points the other way, which supports the "news reports the move" explanation:

| Correlation with the move already in `ret_1` | FinBERT | LLM |
|---|---|---|
| With the 16:00 UTC roll | +0.080 | +0.101 |
| Without the roll (late headlines included) | +0.143 | +0.277 |

Letting late-day headlines into the same row nearly triples the LLM's correlation with that day's move (0.101 to 0.277). That is the signature of news describing a move that had already happened, and it is the reason the cutoff exists.

## What this means for the project
- The research question stays valid, but a likely honest answer is "no model variant beats the majority baseline". The slides should be framed as testing whether news helps, not assuming it does (slide 5 already says this).
- These are raw correlations. M4 still has to test combinations, disagreement and interactions properly before any conclusion.
- The result depends on the timing assumptions (UTC, 16:00 cutoff, Yahoo bar window). The gap and no-roll columns above are the sensitivity check, and none of them gives a positive link.

## Known limits
- 249 rows, one 4-month period, five correlated pairs (a dollar move hits several pairs at once, so rows are not independent).
- 33 EURCHF rows with the price dates unverified against FRED.
- The LLM prompt and the UTC/16:00 cutoff were fixed before looking at returns and have not been tuned.
