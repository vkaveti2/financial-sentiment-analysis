# Reproducing the parent paper (Fatouros et al. 2023) with our code

Code: `src/reproduce.py` (run `python -m src.reproduce`). Paper: arXiv:2308.07935, Machine Learning with Applications 14, 100508. Paper values below were read from the paper's arXiv HTML through a fetch tool, **not** checked against the PDF; verify each against the PDF before putting it on a slide.

## 1. Sentiment classification vs human labels (paper Table 4), n = 2,291

| Model | Accuracy | Precision | Recall | F1 | S-MAE |
|---|---|---|---|---|---|
| FinBERT, paper | 0.561 | 0.560 | 0.562 | 0.556 | 0.540 |
| **FinBERT, ours** | **0.562** | **0.560** | **0.562** | **0.556** | **0.540** |
| GPT-P1, paper | 0.730 | 0.760 | 0.730 | 0.725 | 0.300 |
| GPT-P2, paper | 0.790 | 0.797 | 0.790 | 0.790 | 0.227 |
| **Qwen2.5-14B, ours** | **0.752** | 0.768 | 0.752 | 0.752 | 0.277 |

- **FinBERT reproduces the paper to three decimals on every metric.** (The 0.001 accuracy difference is 2 or 3 headlines out of 2,291; the dataset file ships FinBERT's labels, so this confirms we read the data and compute the metrics the way the authors did.)
- Our LLM is not ChatGPT, so there is nothing to match exactly. Qwen sits between the paper's GPT-P1 and GPT-P2 on every metric, and far above FinBERT, the same ordering the paper reports (LLM prompts clearly beat FinBERT).
- Metric definition: precision, recall and F1 are support-weighted averages, because the paper's recall equals its accuracy. We chose this after seeing the paper's table. With macro averaging, our FinBERT numbers would not match (recall 0.568, F1 0.559). S-MAE is the mean absolute error on the -1/0/+1 scale.

## 2. Link to market moves (paper Tables 7 to 9, same-day), our data

Method as in the paper: sum all sentiment scores for a pair within one day, then compare with that day's return. Directional accuracy (DA) is the share of days where the sign of the score matches the sign of the return, counting only days with a nonzero score.

| | Paper | Ours |
|---|---|---|
| Human-label score vs return, Pearson | "around 0.49" | 0.438 (247 pair-days) |
| FinBERT directional accuracy, all pairs | 0.599 | 0.621 |
| Best-matching LLM DA, all pairs | GPT-P2: 0.644 | Qwen: 0.643 |

Per pair (DA, paper Table 9 vs ours):

| Pair | FinBERT paper / ours | GPT-P2 paper / Qwen ours |
|---|---|---|
| AUDUSD | 0.518 / 0.543 | 0.648 / 0.571 |
| EURCHF | 0.250 / 0.292 | 0.437 / 0.476 |
| EURUSD | 0.654 / 0.705 | 0.576 / 0.688 |
| GBPUSD | 0.698 / 0.659 | 0.679 / 0.652 |
| USDJPY | 0.714 / 0.745 | 0.678 / 0.735 |

- **Similar, not identical**, as expected: the paper does not say where its prices came from, which time zone it used, or how days with a zero score were handled, and our LLM differs. The overall pattern matches: around 60 to 65% same-day agreement, USDJPY and GBPUSD strongest, **EURCHF weak for both models (below 50%) in both studies**.
- Pearson by model is only in the paper's Figure 8 (an image), so only the human-label figure could be compared.

## 3. What did not reproduce

| Paper item | Status |
|---|---|
| GPT-P2 numbers (0.790 accuracy) | Cannot be reproduced: needs ChatGPT; we used Qwen2.5-14B instead. |
| 41.9% FinBERT/GPT-P2 disagreement on slide 9 | Not in the paper's text that we could read; cannot be reproduced without GPT-P2 labels. We measured 44.0% with Qwen. |
| "Filtered data" (Table 6: headlines that do not name the pair) | Not reproduced. Our guess at the filter (pair code absent from the title, 489 headlines) gives FinBERT 0.485 against the paper's 0.543, so our filter is evidently not theirs. Not claimed. |
| "35% better classification, 36% higher correlation" | The 35% is roughly reproducible (mean of GPT-P1 to P4 accuracy 0.760 divided by 0.561), but the paper does not show the formula; the 36% needs Figure 8 values. Not claimed. |
| Predictive power | The paper itself says same-day alignment "does not necessarily imply predictive power". Our M4 tests prediction and finds none (see `m4_results.md`). |

## 4. Honest summary
The paper's FinBERT classification result reproduces exactly, our substitute LLM reproduces the paper's qualitative finding, and the same-day market association reproduces approximately. The ChatGPT-specific numbers do not reproduce, by design. All of our departures (Qwen instead of ChatGPT, Yahoo prices re-dated by one day, UTC dates) are documented.
