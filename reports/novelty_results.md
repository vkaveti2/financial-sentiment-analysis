# Novel contribution results: how to use a lightweight local LLM to label forex headlines

Plan: `novelty_plan.md` (commit `beeaa3b`, written before any of this ran). Code: `src/novelty.py`. Reproduce with `python -m src.novelty --evaluate-only` (no Ollama needed; the 4,582 Qwen answers are saved in `data/labels/novelty_cache.jsonl`). 21 tests pass.

## Setup
2,291 headlines sorted by time. **Dev** = first 60% (1,375, through 2023-03-29). **Test** = last 40% (916 headlines, 32 dates), scored once. Few-shot examples, the choice of best Qwen variant, and the stacker all come from dev only. The model is Qwen2.5-14B-Instruct through Ollama, temperature 0, seed 42. Intervals are 95% bootstraps over test **dates** (2,000 resamples).

## Main table (test split, n = 916)

| ID | Technique | Accuracy [95% CI] | Weighted F1 | S-MAE (lower is better) |
|---|---|---|---|---|
| A0 | FinBERT (parent's classical model) | 0.547 [0.516, 0.579] | 0.542 | 0.558 |
| A1 | Qwen zero-shot (parent's style, one prompt, no examples) | 0.753 [0.719, 0.788] | 0.754 | 0.272 |
| A2 | Qwen few-shot, 9 fixed human-labeled examples | 0.777 [0.750, 0.805] | 0.779 | 0.237 |
| **A3** | **Qwen few-shot, 6 retrieved similar labeled headlines** | **0.809 [0.773, 0.844]** | **0.809** | **0.207** |
| A4 | Stacked FinBERT + Qwen (logistic regression) | 0.809 [0.773, 0.844] | 0.809 | 0.207 |

No invalid answers in any run. A5 (smaller 7B model) from the plan was optional and was **not run**.

## The four pre-specified comparisons

| Hypothesis | Result (paired accuracy difference, 95% CI) | Verdict |
|---|---|---|
| H1a: A2 beats A1 | +2.4 points [-0.6, +5.4] | **Not supported** (interval includes 0) |
| H1b: A3 beats A1 | **+5.6 points [+2.6, +8.5]** | **Supported** |
| H2: A3 beats A2 | **+3.2 points [+0.9, +5.3]** | **Supported**, with a lower bound close to zero |
| H3: A4 beats best single technique (A3) | +0.0 points [0.0, 0.0] | **Not supported** |

All techniques also beat FinBERT by 20 to 26 points (every interval far from zero). With four confirmatory comparisons, A3 vs A1 is the sturdy result; A3 vs A2 is modest.

Predictions versus outcomes (from the plan): A0 about 0.55, observed 0.547. A1 about 0.75, observed 0.753. A2 predicted +1 to +4, observed +2.4 (inside the range, but not distinguishable from zero). A3 predicted +2 to +6, observed +5.6. **A4 predicted +1 to +3, observed 0.**

## Why A4 added nothing
The stacker (trained on dev) reproduces A3's label on 100% of dev rows and 100% of test rows. It learned to copy the Qwen label and ignore FinBERT, because on dev the retrieval-prompted Qwen is right far more often than FinBERT. So combining the two models, at the labeling stage, gives no gain over Qwen alone once Qwen sees good examples. (FinBERT's label and an agree/disagree flag were available to it.)

## Where the gain comes from (exploratory, looked at after the main result)

| Test subset | n | A0 FinBERT | A1 zero-shot | A3 retrieved |
|---|---|---|---|---|
| FinBERT and Qwen agree | 506 | 0.812 | 0.812 | 0.838 |
| FinBERT and Qwen disagree | 410 | 0.220 | 0.680 | **0.773** |

- Most of A3's gain is on the headlines where the two models disagreed (+9.3 points over A1, versus +2.6 where they agree). On disagreements FinBERT is right only 22% of the time.
- By pair (A1 to A3): AUDUSD 0.781 to 0.839, GBPUSD 0.764 to 0.829, USDJPY 0.757 to 0.817, EURUSD 0.722 to 0.770, EURCHF 0.900 to 0.900 (10 headlines only).

## Checks on whether the gain is real
- **Exact repeats:** 20 test headlines have an identical title in dev. Without them A3 scores 0.805 and A1 0.750, so the gain is the same (+5.5).
- **Near-duplicates:** the gain is similar for test headlines with low, middle and high similarity to any dev headline (+6.2, +4.3, +6.2 points), so it does not depend on copying a nearly identical earlier headline. Only 2.3% of test headlines have cosine similarity of 0.9 or more to a dev headline.
- Both checks were run after seeing the results and are labeled exploratory.

## Limits and honest caveats
- **Same annotators.** The human labels judge short-term price impact and come from one annotation team. Retrieval shows the model how those annotators labeled similar headlines, so part of the gain is learning their conventions, not general understanding. That is still useful for this dataset, but may not carry to other labelers.
- **Templated headlines.** FXStreet headlines follow repeated templates ("extends bounce off ... on upbeat ... PMI"); retrieval is well suited to them. Gains on other news styles are untested.
- **One model, one prompt family, one 4-month window.** 916 test rows over 32 dates resolve differences of roughly 3 to 4 points.
- **Not tested downstream.** We did not check whether better labels improve next-day price prediction. Labeling earlier days' headlines with human labels from later days would leak the future, and M4 already found no prediction benefit from zero-shot labels. Do not claim improved forecasting.
- **Not comparable to the paper's 0.790.** That figure is ChatGPT (GPT-P2) on all 2,291 headlines; ours is Qwen on a later 40% subset. Do not claim we beat ChatGPT. (Paper numbers were read from the arXiv HTML, not the PDF.)
- Dev accuracy of A1/A2/A3 (0.750/0.773/0.799) shows the same ordering as test, so the ranking is not a quirk of the test split.
