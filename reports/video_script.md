# Recording script (solo, about 6 to 7 minutes)

Written for: you, recording alone, for the Week 11 "novelty" assignment.

**Before you hit record**
1. Terminal tab 1: `cd financial_sentiment && source .venv/bin/activate`. Tab 2: `ollama serve` (only needed for the live demo in Part 4; skip it and the live step if you want).
2. Open in your editor: `src/novelty.py` (scroll position at the top), `reports/novelty_results.md`, and the GitHub page https://github.com/vkaveti2/financial-sentiment-analysis in a browser tab.
3. Increase your terminal and editor font size. Do a 20-second test recording.
4. Rehearse once with `python -m src.demo` so the numbers are not a surprise.

**On teamwork.** The assignment asks for a recording of how the team worked together. You are alone on camera, so say plainly who did what. I left `[Harshi: ...]` for you to fill in with what she actually did. Do not claim contributions that did not happen.

---

## Part 1: The idea (0:00 to 1:15). Screen: slide 6 or the README

> "Hi, I'm Varun Kaveti. This is our DS 340W project with Harshi Chinnam: can two interpretations of financial news, FinBERT and a language model, help us read forex headlines and predict price moves.
>
> The parent paper, Fatouros et al., showed that ChatGPT labels forex headlines much better than FinBERT. Two things bothered us. It depends on a paid API that nobody can rerun for free, and a zero-shot prompt never shows the model how the human annotators actually judged these headlines.
>
> So our contribution is to use a small, free language model that runs on a laptop, Qwen 2.5 with 14 billion parameters through Ollama, and to change how we use it. The modified block is the headline labeling step."

## Part 2: The novelty in the code (1:15 to 3:45). Screen: `src/novelty.py`

> "Everything is in one file, `src/novelty.py`. The docstring at the top lists the five techniques."

**Scroll to the top docstring (lines 1 to 20).**
> "A0 is FinBERT. A1 is zero-shot Qwen, which mirrors the parent paper's style. A2 is few-shot with nine fixed examples. A3, the main idea, is few-shot with retrieved examples. A4 stacks FinBERT and Qwen. I wrote this plan and committed it before running anything, so I could not tune it after seeing the results."

**Line 38, `DEV_FRAC = 0.6`, and `load_split` (line 57).**
> "First, honesty about evaluation. I sort every headline by time. The first 60 percent is development, the last 40 percent, 916 headlines, is the test set I scored exactly once. Every example and every choice comes from the earlier period only."

**`retrieved_shots`, line 81. Spend the most time here.** Point at lines 87 to 97.
> "This is the core idea. For each headline, I turn all the development headlines into TF-IDF vectors and find the six most similar ones by cosine similarity. Line 93 sets a headline's similarity to itself to minus infinity, so a development headline can never see its own answer. And because I only fit on development titles, a test headline can only ever retrieve development examples, never another test label."

**`few_shot_prompt`, line 108.**
> "Those six headlines and their human labels get written into the prompt, most similar last, followed by the new headline. So the model is shown how the annotators labeled nearby cases, then asked to label this one."

**Compare with `src/sentiment.py` line 23, the original `PROMPT`.**
> "Here is what we had before: one instruction, no examples. The only change in the call is the prompt text. Temperature zero and a fixed seed keep it repeatable, and every answer is cached, which is why a teammate can rerun this without waiting."

**`stack_features` and `fit_stacker`, lines 165 to 180.**
> "The stacked model is a small logistic regression on FinBERT's label and score, Qwen's label, and whether they agree. I tested it because disagreement was my original hypothesis."

**`date_bootstrap`, line 190.**
> "Last piece: the confidence intervals. Headlines on the same day are related, so I resample whole days rather than individual headlines. Otherwise the intervals would look tighter than they really are."

## Part 3: The results (3:45 to 5:30). Screen: terminal

Run: `python -m src.demo` and scroll to section 7 (or `python -m src.novelty --evaluate-only`).

> "On the 916 held-out later headlines: FinBERT gets 54.7 percent. Qwen zero-shot gets 75.3. Static few-shot gets 77.7. Retrieved few-shot gets 80.9.
>
> The paired comparison matters more than the raw numbers. Retrieved beats zero-shot by 5.6 points, with a 95 percent interval from 2.6 to 8.5, so that is a real gain. Retrieved beats the fixed examples by 3.2 points, interval 0.9 to 5.3, a smaller gain, close to the edge. Fixed examples versus zero-shot is 2.4 points but the interval crosses zero, so I won't claim it."

> "Two things did not go the way I predicted. First, stacking FinBERT with Qwen added exactly nothing. The stacker learned to copy Qwen's label and ignore FinBERT, and I checked that it reproduces every test prediction. Second, I did not test a smaller model. I'm reporting both."

> "Where does the gain come from? When FinBERT and Qwen disagree, which is 410 of the 916 headlines, FinBERT is right only 22 percent of the time and zero-shot Qwen 68 percent. Retrieval raises that to 77 percent. That breakdown was looked at after the main result, so treat it as exploratory."

## Part 4: Live demo, optional (5:30 to 6:00). Needs `ollama serve`

Run: `python -m src.demo --live 3`
> "This labels three random headlines on my laptop right now with the local model, about a second and a half each, and compares to the human label and FinBERT."

## Part 5: Teamwork and close (6:00 to 7:00). Screen: GitHub page, then `reports/project_log.md`

> "On how we worked: I built the pipeline, [fill in: the data checks, the timing fix, the labeling experiments]. Harshi [fill in what she did: slides, paper sections, literature, review]. The repo is public, and the project log records every step.
>
> Along the way we caught a real error: Yahoo's daily prices were dated one day late, which would have created a fake correlation between news and next-day returns. After fixing it, none of our models, with or without disagreement, predicted next-day direction better than chance. That is a negative result and we are reporting it as one.
>
> So our conclusion is narrow but solid: a small, free, local language model, shown a few similar human-labeled headlines, labels forex news about 5.6 points better than the same model asked cold, and far better than FinBERT. Limits: one model, one four-month window, one annotation team, and we did not test whether better labels improve forecasting. Thank you."

---

## Code-highlight cheat sheet (line numbers from `src/novelty.py`)
| Moment | Where | What to point at |
|---|---|---|
| Plan and techniques | lines 1 to 20 | The "NOVEL" markers and the outcome note |
| No peeking at test | line 38 `DEV_FRAC`, `load_split` line 57 | Time-sorted 60/40 split |
| **The main novelty** | `retrieved_shots`, lines 81 to 97 | TF-IDF similarity, line 93 (no self-retrieval), "dev titles only" on line 88 |
| How the LLM is used | `few_shot_prompt`, line 108 | Examples first, then the query |
| What it replaced | `src/sentiment.py` line 23 | Original zero-shot `PROMPT` |
| Tested but no gain | `stack_features` / `fit_stacker`, lines 165 to 180 | The agree flag |
| Honest uncertainty | `date_bootstrap`, line 190 | Resamples whole days |
| Evidence the code is careful | `tests/test_novelty.py` | Run `python -m pytest tests -q`: 21 pass, including "test labels never retrieved" |

Line numbers shift if the file is edited; re-check with `grep -n "^def " src/novelty.py` before recording.

## Things not to say
- Do not say the parent paper used only FinBERT: it compared FinBERT with six ChatGPT prompts. Say "the parent used a paid ChatGPT API; we use a free local model."
- Do not say we beat ChatGPT or the paper's 0.790. Different model, different data subset.
- Do not say better labels improve price prediction. We did not test it.
