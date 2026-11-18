# Reading Forex News with FinBERT and a Free Local LLM

DS 340W project by Harshi Chinnam and Varun Kaveti, Penn State.

This guide takes you from a brand-new Mac to every number in our paper. You do not need to know Python. Copy each command exactly, one box at a time, and compare what you see with the "You should see" notes.

---

## 1. What this project is

**The question.** Financial news arrives all day. Can software read forex (currency) headlines well enough to help predict whether a currency pair, such as EUR/USD, will go up or down the next day?

**The two "readers".**
- **FinBERT** is an older AI model trained on finance text. Its labels come with the public dataset.
- **Qwen2.5-14B** is a free, open language model (a ChatGPT-style model) that runs on your own laptop through an app called **Ollama**. It costs nothing and needs no account or API key.

Each reader labels every headline **positive**, **negative** or **neutral** for its currency pair. We compare them with human labels and test whether their labels help predict the next day's price direction.

**What we found** (all numbers are recreated by this guide):
1. **The free local LLM reads forex news far better than FinBERT.** On all 2,291 headlines it agrees with the human label 75.2% of the time; FinBERT agrees 56.2%.
2. **Showing the LLM similar, already-labeled headlines makes it better.** On 916 held-out later headlines, accuracy goes from 75.3% (asked cold) to 80.9% (shown 6 similar labeled examples), a gain of 5.6 points (95% interval +2.6 to +8.5).
3. **But reading the news is not the same as predicting the market.** In a leakage-controlled walk-forward test on 164 rows, no combination of FinBERT, the LLM, and their disagreement beat chance at predicting next-day direction. Every accuracy interval includes 50%. This is a negative result, and we report it as one.

The dataset is 2,291 forex headlines (Jan to May 2023, five currency pairs) from Fatouros et al. (2023), CC-BY-4.0.

---

## 2. Before you start

| You need | Details |
|---|---|
| A Mac | Apple chip (M1 or newer) recommended. Ollama needs macOS 14 (Sonoma) or newer. |
| Free disk space | About 15 GB (the model is 9 GB, the Python packages about 0.25 GB). |
| Memory (RAM) | 16 GB or more recommended for the 14B model. This is our estimate; the Ollama page only lists download sizes. |
| Internet | Needed for the downloads. The 9 GB model is the slow one. |
| Time | About 20 minutes of hands-on work, plus the model download. |

Check your Mac. Open **Terminal** (press Cmd+Space, type "Terminal", press Enter) and run:

```bash
sw_vers -productVersion        # want 14 or higher
uname -m                       # "arm64" means Apple chip
sysctl -n hw.memsize           # bytes of RAM; 17179869184 = 16 GB
df -h ~                        # check "Avail" has at least 15G
```

---

## 3. Step-by-step setup

### Step 1. Install Apple's developer tools (gives you `git` and Python)

```bash
xcode-select --install
```

A window pops up. Click **Install**, wait for it to finish, then close Terminal and open a new one. If it says the tools are already installed, that is fine.

**Check:**
```bash
git --version
python3 --version
```
You should see a git version and `Python 3.9.x` or newer. Python 3.9.6 is what Apple provides, and it is what we tested.

### Step 2. Download the code from GitHub

```bash
cd ~/Documents
git clone https://github.com/vkaveti2/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

You should now be inside a folder called `financial-sentiment-analysis`. Every later command is run from this folder. If you open a new Terminal window, run `cd ~/Documents/financial-sentiment-analysis` first.

### Step 3. Create the Python environment

This makes a private box of packages for the project so nothing else on your Mac is touched.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You should see `(.venv)` at the start of your Terminal line, then a list of packages installing, ending with `Successfully installed ...`. The environment uses about 240 MB of disk. In our test the install took about 10 seconds with a warm download cache, so expect one to a few minutes on a new laptop.

**Every time you open a new Terminal window, run `source .venv/bin/activate` again** before any command below.

### Step 4. Get the dataset

```bash
python -m src.data
```

You should see:
```
headlines  : .../data/raw/sentiment_annotated_with_texts.csv (md5 OK)
llm labels : .../data/processed/llm_labels.csv
prices     : .../data/raw/prices_daily.csv (md5 OK, frozen snapshot)
next: python -m src.features
```

What this did:
- **Headlines:** downloaded from Zenodo (doi:10.5281/zenodo.7976208, CC-BY-4.0) and checked against a fingerprint (md5 `95ba5134ca6203fee889378dcc9a42c6`). If the fingerprint is wrong, the script stops instead of using bad data.
- **LLM labels:** copied from `data/labels/llm_labels.csv`, so you do not have to wait about 15 minutes for the model to relabel everything.
- **Prices:** copied from a frozen snapshot of Yahoo Finance daily prices (pulled 2026-10-29), also fingerprint-checked. We froze it so results never depend on what Yahoo returns on the day you run this.

### Step 5. Run the tests

```bash
python -m pytest tests -q
```

**You should see:** `21 passed` (plus a warning line, which is harmless). These tests check that no future information leaks into the features, that the train/test splits are clean, and that the novel method never looks at test labels.

### Step 6. Recreate every result

Run these four commands in order.

```bash
python -m src.features                    # builds the daily table (about 1 second)
python -m src.run_m4                      # the six-model prediction test (about 45 seconds)
python -m src.novelty --evaluate-only     # the novel few-shot comparison (about 2 seconds)
python -m src.reproduce                   # metrics from the parent paper on our data (about 1 second)
```

Then check them against the paper:

```bash
python -m src.verify
```

**You should see** a list of `PASS` lines, ending with:
```
ALL 41 CHECKS PASS: this machine reproduced the paper numbers.
```

Last, confirm nothing changed:

```bash
git status
```
**You should see** `nothing to commit, working tree clean`. The result files you just regenerated are identical, byte for byte, to the ones we committed. If `git status` lists modified files under `reports/`, see Troubleshooting.

### Step 7. Get the local LLM (Ollama)

Steps 1 to 6 did not need this. The saved model answers (4,582 of them) are in the repo, so results recreate without Ollama. You need it only to watch the model label headlines live.

1. Go to **https://ollama.com/download**, download the macOS app, open the `.dmg`, and drag **Ollama** into **Applications**.
2. Open **Ollama** from Applications. On first launch it may ask permission to install its command-line tool (a link in `/usr/local/bin`). Allow it and enter your Mac password if asked.
3. In Terminal, download the model. This is **9 GB**, so use a fast connection:

```bash
ollama pull qwen2.5:14b-instruct
```

4. Check it:

```bash
ollama list
```
**You should see** a line for `qwen2.5:14b-instruct`, ID `7cdf5a0187d5`, size `9.0 GB`.

**If you see `could not connect to ollama server`**, the server is not running. Open the Ollama app, or run this in a **second Terminal tab** and leave it open:

```bash
ollama serve
```

### Step 8. Run the demo

```bash
python -m src.demo
```

This prints the whole story in seven sections: data, how well each model matches the humans, how often the two models disagree (1,009 of 2,291 = 44.0%), whether disagreement signals unreliable labels, the same-day link to price moves, the next-day prediction test, and our novel contribution.

To watch the local LLM label headlines **live** (needs Step 7, with Ollama running):

```bash
python -m src.demo --live 3
```

It labels 3 random headlines on your laptop and shows the human label, FinBERT's label, and the LLM's label side by side. The LLM runs at temperature 0, so it is nearly always identical to the saved answers, but a rare headline can differ on different hardware. The printed line says so when it happens, and every result in this guide uses the saved answers.

---

## 4. The results you should see

| What | Number | Where it comes from |
|---|---|---|
| Qwen agrees with human labels (2,291 headlines) | **75.2%** | `src.verify`, `src.demo` |
| FinBERT agrees with human labels | **56.2%** | same |
| FinBERT and Qwen give different labels | **1,009 of 2,291 = 44.0%** | `src.demo` section 3 |
| Qwen is right when FinBERT agrees / disagrees | **82.5% / 65.9%** | `src.demo` section 4 |
| Usable rows for next-day prediction | **249 pair-days over 71 dates** | `src.features` |
| Next-day prediction, highest-scoring variant (prices + Qwen); not distinguishable from chance | **52.4%**, interval [43.8%, 60.4%] | `src.run_m4` |
| Does adding disagreement help? | **-1.8 points, interval [-5.9, +2.4]**: no | `src.run_m4` |
| Novel method on 916 held-out headlines: FinBERT / Qwen cold / fixed examples / **retrieved examples** | 54.7% / 75.3% / 77.7% / **80.9%** | `src.novelty` |
| Retrieved examples vs Qwen cold | **+5.6 points, interval [+2.6, +8.5]** | `src.novelty` |

Details and caveats: [`reports/m4_results.md`](reports/m4_results.md), [`reports/novelty_results.md`](reports/novelty_results.md), [`reports/novelty_plan.md`](reports/novelty_plan.md) (written and committed before the experiment ran).

**Things we do not claim:** that we beat ChatGPT, that better labels improve price forecasting, or that news can never help. We found no evidence of a benefit in this sample, and our test set is small enough to rule out only large effects (about 10 points or more).

---

## 5. How the code fits together

```
Zenodo headlines (2,291) ---> FinBERT labels (shipped with the data)
                         \--> Qwen labels (src/sentiment.py, saved in data/labels/)
Yahoo prices (frozen snapshot, dates fixed with move_date)
        |
        v
src/features.py   one row per currency pair per day: average FinBERT score, average Qwen score,
                  how often they disagree, recent price changes; target = did tomorrow go up?
        |
        v
src/models.py + src/evaluate.py + src/run_m4.py    six variants, walk-forward folds, bootstrap intervals
src/novelty.py                                      novel contribution: few-shot labeling with retrieved examples
        |
        v
src/verify.py     checks 41 numbers against the paper         src/demo.py   prints the whole story
```

| File | What it does |
|---|---|
| `src/data.py` | Downloads the headlines (checksummed) and restores labels and the price snapshot. |
| `src/sentiment.py` | Asks the local LLM to label headlines (one-time, about 15 minutes; results are saved). |
| `src/features.py` | Builds the daily table. Headlines at or after 16:00 UTC roll to the next day so news written after a price move cannot sit beside it. |
| `src/models.py`, `src/evaluate.py`, `src/run_m4.py` | The six-variant prediction test with walk-forward folds and date-level bootstrap. |
| `src/novelty.py` | The novel contribution: zero-shot vs fixed examples vs retrieved examples vs stacking. |
| `src/reproduce.py` | The parent paper's own metrics, computed on our data. |
| `src/timing_check.py` | The audit that found Yahoo's dates are one day late (needs internet). |
| `src/verify.py` | PASS/FAIL against the paper's numbers. |
| `src/demo.py` | One-command summary for presentations. |
| `tests/` | 21 automated tests (leakage, splits, the novel method). |
| `data/labels/` | Our saved Qwen answers. `data/snapshots/` holds the frozen prices. |
| `reports/` | Write-ups: data audit, timing decision, results, plans. |
| `legacy/` | An earlier, unrelated Kaggle notebook. Not part of the pipeline. |

---

## 6. Troubleshooting

| What you see | What it means | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or `pandas`) | The Python environment is not active. | `source .venv/bin/activate`, then rerun. |
| `No module named 'src'` | You ran a file directly. | Use the `python -m src.NAME` form exactly as written, from the project folder. |
| `KeyError: 'true_sentiment'` or other odd data errors | A data file was overwritten or damaged. | Delete `data/processed/llm_labels.csv`, run `python -m src.data`, rerun the step. |
| `git status` shows modified files under `reports/` | Your package versions differ from ours. | Run `pip install -r requirements.txt` again inside the active environment. The `src.verify` PASS lines are the real check. |
| `Could not reach the local model` / `could not connect to ollama server` | Ollama is not running. | Open the Ollama app, or run `ollama serve` in a second Terminal tab. Then `ollama list`. |
| `zsh: command not found: ollama` | The Ollama command-line tool is not installed. | Open the Ollama app once and allow the command-line install. (Expected from Ollama's documentation; not reproduced in our testing.) |
| The model download is very slow or stalls | 9 GB over a slow connection. | Rerun `ollama pull qwen2.5:14b-instruct`; it resumes. |
| `python3: command not found` or an Xcode install prompt | Developer tools are not installed. | Redo Step 1. |

The first four rows were reproduced in testing. The last four are expected from documentation and were not reproduced.

**Optional, not for a demo:** `python -m src.sentiment` relabels all 2,291 headlines with Ollama (about 15 minutes) and overwrites `data/processed/llm_labels.csv` with fresh answers that can differ slightly from ours, which can make `src.verify` fail. To undo it, delete that file and run `python -m src.data`. `python -m src.prices --refresh` likewise re-downloads prices (needs `pip install -r requirements-refresh.txt`) and can change results.

---

## 7. Credits and references

- Data: Fatouros et al., *Forex news annotated dataset for sentiment analysis*, Zenodo, doi:10.5281/zenodo.7976208, CC-BY-4.0. Our saved labels (`data/labels/`) are derived from it and reused with attribution.
- Prices: Yahoo Finance daily bars via `yfinance`, pulled 2026-10-29 and stored as a frozen snapshot. Used to cross-check dates: Federal Reserve (FRED) noon New York rates.
- Fatouros et al. (2023). Transforming sentiment analysis in the financial domain with ChatGPT. *Machine Learning with Applications* 14, 100508.
- Moradi-Kamali et al. (2025). Market-Derived Financial Sentiment Analysis. arXiv:2502.14897.
- Dong, Fan and Peng (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. KDD 2024.
- FinBERT: Araci (2019). Qwen2.5: Qwen Team. Ollama: ollama.com.
