# Section: Proposed Novel Approach (draft for the paper)

*Replace [bracketed] items. Numbers come from `reports/novelty_results.md`.*

## X. Proposed Novel Approach: Retrieval-Augmented Labeling with a Lightweight Local LLM

### X.1 Motivation
The parent study (Fatouros et al., 2023) showed that zero-shot ChatGPT prompts label forex headlines far better than FinBERT (their best prompt, GPT-P2, reaches 0.790 accuracy against FinBERT's 0.561). Two practical limits remain. First, a paid, closed API is hard to reproduce and cannot be rerun by readers at no cost. Second, a zero-shot prompt cannot show the model how the human annotators actually judged similar headlines, even though the annotation task (short-term impact on a specific currency pair) differs from ordinary sentiment.

### X.2 What we change
We replace the parent's single zero-shot ChatGPT prompt with a **lightweight, open-weights LLM (Qwen2.5-14B-Instruct) run locally through Ollama** at temperature 0 with a fixed seed, so every result is repeatable and free to rerun. We then modify the labeling block with three techniques that use the LLM differently:

- **Few-shot, static (A2).** Nine fixed human-labeled headlines (three per class) are placed in the prompt.
- **Few-shot, retrieved (A3, our main contribution).** For each headline, the six most similar earlier human-labeled headlines (TF-IDF cosine similarity, restricted to the development period) are placed in the prompt, most similar last. The model is shown how annotators labeled nearby cases.
- **Stacked (A4).** A logistic regression combines FinBERT's label and score with the best Qwen variant's label and an agree/disagree flag.

We compare them against FinBERT (A0) and the parent-style zero-shot prompt (A1). A0 and A1 mirror the parent's two kinds of labeler (a classical finance model and a zero-shot LLM prompt), with Qwen standing in for ChatGPT in A1. No model is fine-tuned; the only fitted component is the stacker's eight-input logistic regression.

### X.3 Hypotheses (committed before running)
H1: few-shot prompts improve on zero-shot. H2: retrieved examples improve on fixed examples. H3: stacking FinBERT with the best Qwen variant improves on the best single technique. The design, split and metrics were fixed in a committed plan (`reports/novelty_plan.md`) before any technique was run.

### X.4 Experimental design
Headlines are sorted by time. The first 60% (1,375) form a development set from which all examples, selections and weights come; the last 40% (916 headlines, 32 dates) are scored once. We report accuracy, weighted F1 (the paper's averaging, where recall equals accuracy), macro-F1, and the mean absolute error on the -1/0/+1 scale. Uncertainty is a 95% bootstrap that resamples whole test dates, because headlines on one day are related.

### X.5 Results

| Technique | Accuracy [95% CI] | Weighted F1 | S-MAE |
|---|---|---|---|
| A0 FinBERT | 0.547 [0.516, 0.579] | 0.542 | 0.558 |
| A1 Qwen zero-shot | 0.753 [0.719, 0.788] | 0.754 | 0.272 |
| A2 Qwen few-shot, static | 0.777 [0.750, 0.805] | 0.779 | 0.237 |
| **A3 Qwen few-shot, retrieved** | **0.809 [0.773, 0.844]** | **0.809** | **0.207** |
| A4 Stacked FinBERT + Qwen | 0.809 [0.773, 0.844] | 0.809 | 0.207 |

- **H1 is partly supported.** Retrieved few-shot beats zero-shot by 5.6 points (95% CI +2.6 to +8.5). Static few-shot gains 2.4 points but its interval includes zero (-0.6 to +5.4).
- **H2 is supported.** Retrieved beats static by 3.2 points (CI +0.9 to +5.3), a modest margin.
- **H3 is not supported.** The stacker reproduced A3's labels exactly; it learned to rely on Qwen and ignore FinBERT, so combining the two models added nothing at this stage.
- Every Qwen variant beats FinBERT by 20 to 26 points.
- Most of the retrieval gain comes from headlines on which FinBERT and Qwen disagree (A1 0.680 to A3 0.773, n = 410), where FinBERT is right only 22% of the time. This breakdown was examined after the main result and is exploratory. The gain is similar for headlines with low and high similarity to earlier ones, and is unchanged (0.805 vs 0.750) after removing 20 test headlines whose exact title appears in the development set.

### X.6 Limitations
The human labels judge price impact and come from one annotation team; retrieval partly teaches the model their conventions, which may not carry to other labelers or to headline styles that are less templated than FXStreet's. One LLM family, one prompt family and one four-month window were tested, and 916 test rows detect differences of roughly 3 to 4 points. Our figures are not comparable to the parent's 0.790, which used ChatGPT on all 2,291 headlines, so we make no claim to beat it. A 7B model (planned as an option) was not run. We did not test whether the improved labels help next-day forecasting: earlier work in this project found no forecasting benefit from zero-shot labels, and labeling earlier days with later human labels would leak future information.

### X.7 Contribution summary
(1) A reproducible, free, local LLM labeling pipeline in place of a paid API. (2) Retrieval-augmented few-shot labeling, which improves accuracy by 5.6 points over the parent-style zero-shot prompt on later, held-out headlines. (3) A negative finding that stacking FinBERT with the retrieval-prompted LLM adds nothing. (4) A leakage-controlled evaluation with a pre-committed plan, a time-based split and date-level intervals.

Code and data pointers: [GitHub URL], `src/novelty.py`, `reports/novelty_results.md`.
