# Novel contribution plan (written and committed BEFORE any of these techniques was run)

Assignment: change one modular block of the parent pipeline with a technique we predict improves on it, and report a comparison table. As with M4, every choice below is fixed in advance; anything added later is labeled exploratory. The git commit time of this file is the record.

## What is and is not novel (be precise in the paper)
The parent paper (Fatouros et al. 2023) already compared FinBERT with ChatGPT prompts (GPT-P1 to P6; GPT-P2 reaches 0.790 accuracy against FinBERT's 0.561). So "using an LLM" is NOT our novelty. Our contributions are:
1. **Lightweight, local, free LLM** (Qwen2.5-14B-Instruct through Ollama, temperature 0, fixed seed, every response cached) in place of a paid ChatGPT API. Anyone can rerun it.
2. **Using both models together**, including their disagreement, as inputs, where the paper compares them one against the other.
3. **Forecasting with leakage controls** (UTC/16:00 cutoff, embargo, date-block bootstrap), where the paper reports only same-day association. Result so far: no next-day predictive benefit (`m4_results.md`).
4. **The modified block tested here:** how the LLM is used to label headlines (techniques A1 to A5 below).

## The modified block
Headline sentiment labeling: given a headline and its currency pair, output positive, negative or neutral for the pair's short-term price impact. The parent block is one zero-shot prompt.

## Split (fixed now)
Sort all 2,291 headlines by timestamp. First 60% (1,375 rows, 2023-01-12 to 2023-03-29 07:34) = **dev**. Last 40% (916 rows, from 2023-03-29 07:35) = **test**. All prompts, examples, model choices and stacker weights come from dev only. Test is scored once.

## Techniques compared
| ID | Technique | Uses human labels? | Prediction (stated before running) |
|---|---|---|---|
| A0 | FinBERT (parent's classical baseline, labels shipped in the dataset) | no | about 0.55 accuracy |
| A1 | Qwen zero-shot (our current prompt; labels already saved) | no | about 0.75 |
| A2 | Qwen few-shot, static: 9 dev examples (3 per class, stratified random, seed 42) added to the prompt | dev only | +1 to +4 points over A1 |
| A3 | Qwen few-shot, retrieved: the 6 most similar dev headlines (TF-IDF cosine) with their human labels. For dev rows the row itself is excluded. | dev only | +2 to +6 points over A1; the largest gain |
| A4 | Stacked: logistic regression on dev combining FinBERT score and label, and the best Qwen variant's label (best = highest dev accuracy among A1 to A3). Disagreement is implicit in the inputs. | dev only | +1 to +3 points over the best single Qwen variant |
| A5 | (optional, only if the model is pulled) qwen2.5:7b zero-shot, to test how small the LLM can be | no | 3 to 8 points below A1 |

Few-shot examples are only used as prompt text; no model weights are trained. The 9 static examples are excluded when scoring A2 on dev.

## Metrics and uncertainty
Accuracy (primary), support-weighted F1 (to match `reproduce.py` and the paper's Table 4), macro-F1, and S-MAE on the -1/0/+1 scale. 95% intervals by bootstrap over **test dates** (2,000 resamples, seed 42; headlines on one day are correlated). Comparisons are paired differences on the same test rows.

## Pre-specified comparisons (the only confirmatory ones)
- H1: A2 and A3 each beat A1 in accuracy.
- H2: A3 beats A2.
- H3: A4 beats the best single technique among A0 to A3.
- Every technique is also reported against A0 (the parent's classical model).
- Four comparisons were chosen, so one interval excluding zero is weak evidence, and we will say so.

## Outcomes
"Helps" means a paired interval excluding zero in the predicted direction. If intervals include zero the result is "no evidence of improvement", reported with interval widths. The test set of 916 rows, 32 dates can resolve differences of roughly 3 to 4 points.

## What we will NOT claim
- No claim that any technique beats ChatGPT/GPT-P2 (not run).
- No claim about price prediction from this block. The downstream next-day test stays as in M4. Re-running it with few-shot labels would need human labels from later dates to label earlier headlines, which is a time leak, so it is excluded unless labels are made time-safe.
- Human labels judge short-term price impact, so few-shot examples carry that framing. A gain therefore partly reflects task alignment.

## Known limits
Single prompt family, one 4-month window, one LLM family plus one optional size comparison, human labels from a small annotation team.
