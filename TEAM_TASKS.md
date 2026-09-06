# Team tasks — step by step

Two tasks, for whoever picks them up. Both are independent of the modeling
work currently in progress (feature/hyperparameter iteration) — safe to
start right away without waiting.

Background docs if you want the *why*: [`PLAN.md`](./PLAN.md) (modeling
rationale, banned techniques, validation design) and
[`STATUS.md`](./STATUS.md) (current scores, what's done, what's next).

---

## Task 1 — Reproduce the pipeline locally

Goal: confirm you get the *same* walk-forward CV numbers I got, on your own
machine. This is also literally the reproducibility requirement for a
top-15 finish, so doing it now (not at the deadline) de-risks that.

### 1.1 Clone the repo

You're already a collaborator on `AhsanulHoque22/react-2026-datathon`
(private repo), so:

```bash
git clone https://github.com/AhsanulHoque22/react-2026-datathon.git
cd react-2026-datathon
```

(If HTTPS asks for a password and rejects it, GitHub no longer accepts
account passwords for git over HTTPS — use a
[personal access token](https://github.com/settings/tokens) as the password,
or clone via SSH instead: `git clone git@github.com:AhsanulHoque22/react-2026-datathon.git`.)

### 1.2 Set up the Python environment

Needs **Python 3.12** (`python3 --version` to check). Every dependency
version is pinned in `requirements.txt` — install exactly those, not
whatever `pip` would resolve on its own, so your results match:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 Get the competition data

`data/raw/` is gitignored (raw competition data isn't committed). Set up
your own Kaggle API token if you haven't already:

```bash
# Kaggle account -> Settings -> "Create New Token" downloads kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# accept the competition rules on kaggle.com first if you haven't, then:
kaggle competitions download -c react-2026-datathon -p data/raw
cd data/raw && unzip -o react-2026-datathon.zip && rm react-2026-datathon.zip && cd ../..
```

### 1.4 Build the feature frame

```bash
source venv/bin/activate
python scripts/01_build_features.py
```

Expected tail of the output (takes ~60-70s):

```
Loading raw data...
train=(731942, 13) test=(262648, 12)  (~5s)
Building time-sorted combined frame...
combined=(994590, 14)  (~12s)
Building features...
featurized=(994590, 67)  (~65s)
Running leakage assertions...
OK.
Final shape: (994590, 67), columns: 67
```

If your row/column counts don't match this, stop and report it before
going further — something's different in the data or environment.

### 1.5 Run walk-forward CV and compare numbers

```bash
python scripts/02_cv_eval.py
```

This takes **~15-20 minutes** (trains 4 LightGBM models). Expected result
(as of the current `main`, with merchant/device trailing windows added):

```
Fold 0 [2026-05-21->2026-06-04] ... PR-AUC=0.7496
Fold 1 [2026-06-04->2026-06-18] ... PR-AUC=0.7648
Fold 2 [2026-06-18->2026-07-02] ... PR-AUC=0.7573
Fold 3 [2026-07-02->2026-07-15] ... PR-AUC=0.4973

mean=0.6923 min=0.4973 max=0.7648 std=0.1127
```

Small differences in the 3rd-4th decimal are fine (thread-count-dependent
floating point). If a fold is off by more than ~0.01, or a different fold
number is affected, that's worth flagging — could mean a library version
drifted from `requirements.txt`, or a local code change.

### 1.6 (Optional) Quick unit test

A fast correctness check on synthetic data (~1s, no real data needed):

```bash
python scripts/test_features_small.py
```

Should end with `All assertions passed.` / `Manual spot-checks passed.`

### 1.7 Report back

Post in the team chat (or a commit message / STATUS.md note) whether your
numbers matched. If you find a discrepancy, check `git log` and
`pip freeze | diff - requirements.txt` before assuming it's a bug — version
drift is the most likely cause.

---

## Task 2 — Package a reproducible Kaggle Notebook

Goal: the competition rulebook requires, **if we finish top 15**, pointing
organizers to a specific **Kaggle Notebook Version History entry** that
reproduces our scoring submission — not just GitHub code, and not just the
notebook's current/latest state. Building this incrementally now avoids a
scramble in the ~10h window between the leaderboard closing
(2026-09-07 23:59:59) and the notebook+summary deadline
(2026-09-08 10:00).

### 2.1 Create the notebook

1. Go to `kaggle.com/competitions/react-2026-datathon/code`
2. Click **New Notebook** — creating it from the competition's Code tab
   auto-attaches the competition's dataset, so `train.csv`/`test.csv` are
   already available at `/kaggle/input/react-2026-datathon/` inside the
   notebook without a separate upload.
3. In notebook Settings (right sidebar): confirm the **Accelerator** is
   "None" (we don't need a GPU — LightGBM on this data trains on CPU in
   minutes) to avoid burning GPU quota unnecessarily.

### 2.2 Get the pipeline code into the notebook

The repo is private, so a plain `!git clone` from inside the notebook will
fail without credentials. Two options — pick whichever is easier for you:

**Option A (recommended, no credentials needed in the notebook):** copy the
file contents directly into notebook cells. The pipeline is four small
files (574 lines total) — paste each into its own cell, wrapped so it
defines the functions without needing `sys.path` tricks:

- Cell 1: contents of `src/config.py` (but change `RAW_DIR`/paths to point
  at `/kaggle/input/react-2026-datathon/` instead of `data/raw/` — the
  Kaggle-attached dataset path is different from our local layout)
- Cell 2: contents of `src/data.py`
- Cell 3: contents of `src/features.py`
- Cell 4: contents of `src/model.py`
- Cell 5: the training + prediction logic from `scripts/04_train_final_and_submit.py`
  (skip the CLI-submit step — Kaggle scores the notebook's output CSV
  automatically if this ever becomes a code competition; for now we just
  need the Version History entry with a `submission.csv` in `/kaggle/working/`)

**Option B (if you'd rather not copy-paste):** add a GitHub personal access
token as a Kaggle Secret (notebook Settings → Secrets → Add Secret), then:

```python
import os
token = os.environ["GITHUB_TOKEN"]  # the secret you added
!git clone https://{token}@github.com/AhsanulHoque22/react-2026-datathon.git
import sys; sys.path.insert(0, "react-2026-datathon")
from src.data import load_raw, build_combined_frame
from src.features import build_features
```
(This also needs internet enabled in notebook Settings, and Kaggle's
competition-notebook internet policy may restrict this — check the toggle
under Settings first.)

### 2.3 Run it end to end

Whichever option you used, the notebook should, in order:
1. Load `train.csv`/`test.csv` from the attached dataset path
2. Build the combined time-sorted frame + all features (`build_features`)
3. Train the final LightGBM model on all of train.csv
4. Predict on test.csv
5. Run the same assertions as `scripts/04_train_final_and_submit.py`
   (probabilities in `[0,1]`, no raw ID columns, row order matches
   `sample_submission.csv`)
6. Write `submission.csv` to `/kaggle/working/`

### 2.4 Verify it matches our current best submission

Compare the notebook's output against `submissions/submission_final.csv`
from the repo (same row order, and scores should match closely — not
necessarily bit-identical if LightGBM's threading differs, but the public
LB score should land at essentially the same 0.513 when/if submitted from
this notebook).

### 2.5 Save a Version — this is the actual requirement

In the notebook: **File → Save Version → Save & Run All (Commit)**. This is
what creates a citable **Version History** entry — the current/latest
notebook state is *not* sufficient per the rulebook; organizers need the
specific committed version that produced the scoring `submission.csv`.

Note down:
- The notebook's URL
- The Version number/timestamp of the saved version

Post both in `STATUS.md` or the team chat so whoever finalizes the
submission later knows which version to point to.

### 2.6 Sharing with organizers (only needed if shortlisted)

The rulebook says: *"The notebook must be shared with the organizing
committee's Kaggle account(s) as a collaborator (private share) — do not
make it fully public."* **We don't yet know the exact organizer Kaggle
account(s) to add.** Check the competition's Team/Rules tab closer to the
deadline, or ask in the competition's Discussion tab — don't guess or
share publicly in the meantime, since publishing it before final ranking is
confirmed is explicitly what the rule warns against.

### 2.7 Keep it updated

Every time the modeling pipeline changes meaningfully (new features,
retrained model), re-run the notebook and save a new version, so it never
falls far behind whatever we'd actually submit. Don't leave this until the
last hour before the deadline.
