# Recording checklist: Week 12 technical demo (due Fri 2026-11-20, 8:00 pm)

Everything the assignment asks for, mapped to what to do and say. The recording is on **Harshi's laptop**, which has never run this code. The other teammate joins by webcam on their own machine and must not run anything there during the recording.

## Before the day (does NOT run our code)

- [ ] Harshi runs the four Mac checks from README section 2 (`sw_vers -productVersion`, `uname -m`, `sysctl -n hw.memsize`, `df -h ~`). Want: macOS 14+, `arm64`, 16 GB or more RAM (our estimate for the 14B model), 15 GB free.
- [ ] Both teammates can log into PSU Zoom and have tested a webcam. Both must be on camera for the whole session.
- [ ] Decide how to handle the 9 GB model download. Installing Apple's developer tools, the Ollama app, and pulling the model beforehand is not running our code, but the assignment says the computer must not have *run your code*, so this is our reading, not a rule we have confirmed. If unsure, ask the instructor or a TA. Safest: do it on camera and let it run while the other person explains code.
- [ ] Do NOT let Harshi test-run `src.*` commands on her laptop before recording. Our own clean-machine rehearsal was done on another computer.
- [ ] Have the numbers from the paper open in another window for step 7.

## The seven steps

| # | Assignment step | What to do |
|---|---|---|
| 1 | Use a computer you have not coded on | Harshi's laptop. Say so on camera. |
| 2 | Log into PSU Zoom | Both teammates. |
| 3 | Start recording, share screen, webcams on | Start the recording first, then share the entire screen. Both faces visible. |
| 4 | Show downloading the code from GitHub | Open https://github.com/vkaveti2/financial-sentiment-analysis in a browser to show it is public, then in Terminal: `git clone https://github.com/vkaveti2/financial-sentiment-analysis.git` |
| 5 | Follow the README, error free | Follow README sections 3 (Steps 1 to 8) in order. Read each "You should see" aloud. |
| 6 | Everyone takes turns explaining code | See the code tour below. |
| 7 | Show the paper's results | `python -m src.verify` prints `ALL 41 CHECKS PASS`, and README section 4 lists the same numbers as the paper. Show `git status` is clean. |

Suggested order on camera: steps 1 to 3 (2 min), clone and setup (Steps 1 to 4 of the README, about 5 min), tests (1 min), recreate results (Step 6, about 1.5 min of running), code tour while Ollama downloads (about 5 min), demo and live LLM (3 min), `src.verify` and the paper numbers (2 min).

## Code tour: two speakers, both explain code

A suggested split (change it however you like; do not claim contributions that did not happen). Open files from Terminal on the shared screen with the stock macOS tools, so nothing extra needs installing:

```bash
sed -n '57,100p' src/novelty.py          # prints lines 57 to 100
open -e src/novelty.py                   # or open the whole file in TextEdit
```

Line numbers below are from the committed `src/novelty.py`. Re-check with `grep -n "^def " src/novelty.py` if the file changes.

**Speaker A: from headlines to a daily table**
- `src/data.py`: why the download is checksummed, and why prices are a frozen snapshot.
- `src/prices.py` lines 23 to 26 and `reports/timing_decision.md`: Yahoo's dates are one trading day late; `move_date` fixes it. Sign of the daily move matched an independent source on 82 to 85% of days after the fix, versus chance before.
- `src/features.py`: the 16:00 UTC cutoff in `assign_trading_date` and the target built in `build_daily`.
- `tests/test_features.py`: "changing only future prices cannot change any feature."

**Speaker B: models, the novel method, and the check**
- `src/models.py`: `make_folds` (walk-forward by date) and `train_rows` (the embargo).
- `src/novelty.py`: `load_split` (line 57, time-based 60/40), `retrieved_shots` (line 81; the self-exclusion and dev-only retrieval), `few_shot_prompt` (line 108), `stack_features` and `fit_stacker` (lines 165 to 180), `date_bootstrap` (line 190).
- `src/sentiment.py` line 23: the original zero-shot prompt, to show what changed.
- `src/verify.py`: how each PASS recomputes a number from the files on disk.

Both: take turns on `src/demo.py` output sections.

## Things to say, and not say

- Say: "free local model", "a ChatGPT-style model". Do not say we used ChatGPT or that we beat ChatGPT or the parent paper's 0.790.
- Say: "no evidence in this sample that news predicts next-day direction." Do not say news cannot help, and do not say better labels improve forecasting (we did not test it).
- The earlier slide figure of 41.9% disagreement could not be reproduced from the public data (it has no ChatGPT labels). Do not say where it came from.

## If something goes wrong on camera

Do not stop and restart silently. Read the error, use README section 6, and keep recording. Common ones: environment not activated (`source .venv/bin/activate`), Ollama not running (`ollama serve` in a second Terminal tab). If the recording is unusable, it is fine to start a new one; the assignment allows unlimited attempts.

## After recording

- [ ] Check the recording shows: the GitHub page, the clone, every README command, both speakers, `ALL 41 CHECKS PASS`, and a clean `git status`.
- [ ] Submit the recording link on the assignment page before 8:00 pm Friday.
