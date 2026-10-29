# Data audit (M1, in progress), 2026-10-28

## Source
Zenodo doi:10.5281/zenodo.7976208, license CC-BY-4.0 (open). Single file `sentiment_annotated_with_texts.csv` (4.3 MB), md5 `95ba5134ca6203fee889378dcc9a42c6` (matches Zenodo). Downloaded to `data/raw/` (gitignored).

## What the file contains
2,291 rows. Columns: `published_at`, `ticker`, `true_sentiment` (human), `title`, `author`, `url`, `source`, `text`, `finbert_sentiment`, `finbert_sent_score`.
- **FinBERT output is included.** No GPT-P2 output is included, so ChatGPT labels have to be generated.
- **No price data is included.**
- Timestamps carry no timezone. Dates run 2023-01-12 to 2023-05-04 (86 distinct dates).

## Sample size (this limits every conclusion)
- Tickers: EURUSD 758, GBPUSD 518, USDJPY 499, AUDUSD 461, EURCHF 55.
- Only **293 pair-days** have any news (median 8 headlines per pair-day), before dropping weekends and rows lost to the next-day target.
- January has 12 headlines; March has 1,078. Coverage is uneven.
- 26 duplicated titles.

## Slide claims vs. what this file shows
| Slide claim | Finding |
|---|---|
| 2,291 headlines | Confirmed: 2,291 rows. |
| 960 of 2,291 (41.9%) differ between FinBERT and GPT-P2 | **Not verifiable yet**: no GPT-P2 labels exist locally. |
| (not on slides) FinBERT vs human label | 1,004 of 2,291 differ (43.8%); FinBERT accuracy vs human is 56.2%. This is a different comparison from the slide's, and close in size, so do not mix them up. |

Slides 9 and 12 should say the 41.9% is taken from the literature or is pending reproduction until GPT labels are generated and compared.

## Prices (done)
`src/prices.py` pulled daily OHLC from Yahoo Finance (`yfinance`, `PAIR=X`) for 2022-11-01 to 2023-05-19: 144 trading days per pair, no nulls, no duplicate ticker-dates, weekdays only. Daily-return std is 0.34% (EURCHF) to 0.84% (USDJPY). Up-days run from 45.5% (EURCHF) to 56.6% (GBPUSD), so the majority-class baseline sits near 51-57%.
72 of 2,291 headlines fall on days with no price bar (weekends), so they need a roll-forward rule in M3.
Caveat: Yahoo's daily bar cutoff is not documented here, so the timing rule in M3 must be checked against it.

## Open
- GPT-P2 replacement: Ollama install was blocked, so `src/sentiment.py` is written but has not run.
- GPT-P2 replacement (see README / chat): a free local LLM is proposed, which changes what "ChatGPT" means on the slides.
