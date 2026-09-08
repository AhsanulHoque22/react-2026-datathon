"""Generate notebooks/REACT_2026_best_model.ipynb -- the reproducible artifact
for teammates (and for the rulebook's reproducibility review).

Built from src/ via build_kaggle_kernel.build_common(), so it cannot drift from
the code that actually produced our scores. Regenerate after any src/ change:

    python scripts/build_team_notebook.py
"""
import json
from pathlib import Path

from scripts.build_kaggle_kernel import build_common, V8C_MAIN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "REACT_2026_v9c_reproducibility.ipynb"

INTRO = """# REACT 2026 Datathon — reproducibility notebook

**Team: Overfit & Overcaffeinated**

Reproduces `CANDIDATE_v9c_compliant_0.5280.csv` (public LB **0.55368**), the
strictly-past alternative to our final submission, end to end.

## The submissions

| role | file | public LB | what it is |
|---|---|---|---|
| final private evaluation | `submission.csv` (Sanzid Islam) | 0.56548 | Propagated Tree-Neural Champion — reproduced separately by its author |
| strictly-past alternative | **`CANDIDATE_v9c_compliant_0.5280.csv`** | **0.55368** | **this notebook** |

## Rules compliance

Every engineered feature here uses only information **strictly before t**, per
the organiser's rule. This is enforced mechanically, not by inspection:

- `leakage_assertions()` — first-occurrence rows must carry no prior statistics
- `assert_strictly_past()` — fails the run if any forward-looking column reaches
  a feature matrix; called on the feature list and again on the test matrix
- `prepare_lgb_frame()` — asserts raw `customer_id` / `merchant_id` /
  `device_id` / `transaction_id` are absent, in any encoding

No target encoding, no per-entity or per-subgroup label aggregation at any grain
finer than the whole training set, no external data, no random K-fold (all
validation is time-based / expanding-window).

`docs/METHODOLOGY_DISCLOSURE.md` documents the one judgement call in our final
submission (a post-inference propagation step that reads a symmetric
±30min window). **That step is absent here.**

## The model

197 strictly-prior behavioural features plus a backward-rich family (uniform
window ladder, past-only rate acceleration, second-order gaps) = **269 features**.

Two LightGBM models, each early-stopped independently on a Jul 1–15 holdout and
averaged over 5 seeds:

- **full-train** (`lr=0.02, num_leaves=127, min_data_in_leaf=50, ff=0.85`)
- **90-day specialist** (`lr=0.02, num_leaves=63, min_data_in_leaf=100, ff=0.75`)

blended **60/40**. Held-out Jul 1–15: full 0.5263, specialist 0.5267, blend **0.5280**.

## Two findings worth recording

**No `scale_pos_weight`.** It was prescribed from the start for the 1.76%
imbalance and every early sweep held it fixed. PR-AUC is a *rank* metric;
upweighting positives distorts the ranking it scores. Removing it was worth
**+0.026** on the leaderboard — our single largest gain.

**Local gains amplify ~2x on the leaderboard.** Four calibration points:
0.5204→0.53948, 0.5322→0.56492, 0.5263→0.55262, 0.5280→0.55368. A two-month
test window rewards ranking improvements far more than a two-week fold shows.
"""

BEST_MAIN = V8C_MAIN

OUTRO = """## Before anyone submits this

Submissions are **5 per day for the whole team**, not per person, and they reset
at **00:00 UTC = 06:00 Dhaka** — not local midnight. Coordinate in the team chat
before spending one; a slot burned by accident is gone for the day.

Sanity-check any `submission.csv` first:

```python
import pandas as pd
sub = pd.read_csv("submission.csv")
sample = pd.read_csv("sample_submission.csv")
assert list(sub.transaction_id) == list(sample.transaction_id)
assert sub.fraud.between(0, 1).all() and sub.fraud.notna().all()
print(len(sub), sub.fraud.mean())   # expect 262648 rows, mean ~0.0135
```

## If you want to try an improvement

Work against the acceptance rule above, not a single fold — that is the whole
reason we stopped chasing 0.002 deltas. `STATUS.md` has a **"Settled — do not
revisit"** table (CatBoost, recency weighting, ensembling, calibration, log
transforms); those are closed with evidence, so please read it before spending
compute re-testing them.
"""


def cell(kind, text):
    src = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def main():
    nb = {
        "cells": [
            cell("markdown", INTRO),
            cell("markdown", "## Setup: inlined `src/` (generated — do not edit here)"),
            cell("code", build_common().rstrip() + "\n"),
            cell("markdown", "## Train and write the submission"),
            cell("code", BEST_MAIN.strip() + "\n"),
            cell("markdown", OUTRO),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
