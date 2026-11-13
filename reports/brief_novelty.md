# Brief: Novel Contribution (for your document)

*Written for: your instructor and the TAs, as the new-approach section of the project brief. Edit the bracketed items.*

---

**Novel contribution: retrieval-augmented labeling with a lightweight local language model**

**Problem.** The parent paper (Fatouros et al., 2023) shows that zero-shot ChatGPT prompts label forex news headlines far better than FinBERT, reaching 0.790 accuracy against FinBERT's 0.561. That approach depends on a paid, closed API, and a single zero-shot prompt cannot show the model how the human annotators judged similar headlines.

**Our change.** We modify the headline-labeling block of the pipeline in two ways. First, we replace the paid ChatGPT API with a free, open-weights model, Qwen2.5-14B-Instruct, run locally through Ollama with temperature 0 and a fixed seed, so every result is repeatable. Second, we change how the model is prompted: in addition to a zero-shot prompt (the parent's style), we test few-shot prompts that include human-labeled example headlines, either nine fixed examples or the six most similar earlier labeled headlines, retrieved by TF-IDF cosine similarity. We also test a stacked model that combines FinBERT and the language model.

**Prediction.** Before running any experiment we committed a plan predicting that few-shot prompts would improve on zero-shot, that retrieved examples would improve on fixed ones, and that stacking would add a smaller gain.

**Evaluation.** We sorted the 2,291 headlines by time. The first 60% provided every example and every choice, and the last 40% (916 headlines over 32 days) was scored once. We report accuracy, weighted F1 and mean absolute error with 95% intervals from a bootstrap over days.

**Results.**

| Technique | Accuracy [95% CI] | Weighted F1 |
|---|---|---|
| FinBERT | 0.547 [0.516, 0.579] | 0.542 |
| Qwen zero-shot | 0.753 [0.719, 0.788] | 0.754 |
| Qwen few-shot, 9 fixed examples | 0.777 [0.750, 0.805] | 0.779 |
| **Qwen few-shot, retrieved examples** | **0.809 [0.773, 0.844]** | **0.809** |
| Stacked FinBERT + Qwen | 0.809 [0.773, 0.844] | 0.809 |

Retrieved examples improved accuracy by 5.6 points over zero-shot (95% CI +2.6 to +8.5) and by 3.2 points over fixed examples (+0.9 to +5.3). Fixed examples alone gave 2.4 points, not distinguishable from zero. Stacking added nothing, because the stacker learned to copy the language model's label.

**Honest limits.** One model, one prompt family, one four-month window, and one annotation team, so part of the gain is learning the annotators' conventions. Our numbers are not comparable to the parent's 0.790, which used ChatGPT on all headlines. We did not test whether better labels improve next-day price forecasting; separate leakage-controlled tests found no forecasting benefit from news sentiment in this sample.

**Reproducibility.** Code and results: [GitHub URL]. Run `python -m src.novelty --evaluate-only` to regenerate the table from saved model answers.
