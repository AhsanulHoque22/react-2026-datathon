"""Generate notebooks/REACT_2026_best_model.ipynb -- the reproducible artifact
for teammates (and for the rulebook's reproducibility review).

Built from src/ via build_kaggle_kernel.build_common(), so it cannot drift from
the code that actually produced our scores. Regenerate after any src/ change:

    python scripts/build_team_notebook.py
"""
import json
from pathlib import Path

from scripts.build_kaggle_kernel import build_common

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "REACT_2026_best_model.ipynb"

INTRO = """# REACT 2026 Datathon — best model

**Team: Overfit & Overcaffeinated**

Reproduces our best model end-to-end and writes `submission.csv`. Everything
below is generated from `src/` by `scripts/build_team_notebook.py` — don't edit
this notebook, edit `src/` and regenerate.

## Which model is this

| | local (Jul 1–15 tail) | public LB |
|---|---|---|
| **This notebook — "arm C"** | **0.5252** | not submitted |
| Previous config (git tag `best-0.53948`) | 0.5204 | **0.53948** |

Arm C is our best *model*; 0.53948 is our best *submitted result*. Those two
numbers are on different scales — the local score is a held-out slice of
`train.csv`, the LB score is the real test set — so **do not compare them
directly**. The like-for-like comparison is the local column: 0.5252 vs 0.5204.

## What it does

LightGBM on ~200 leakage-safe behavioural features. Every feature for a row at
time *t* uses only rows strictly before *t*. No raw entity IDs reach the model,
and nothing except the label column touches `fraud`.

Configuration, and why each part is there:

- **No `scale_pos_weight`.** The single biggest win of the competition (+0.026
  on the LB). PR-AUC is a *rank* metric; upweighting positives 55× distorts the
  ranking it scores. Measured monotone on fold 3: spw=55 → 0.5045, spw=10 →
  0.5069, spw=7.4 → 0.5091, spw=1 → 0.5116.
- **`lr=0.02`, `num_leaves=127`, `min_data_in_leaf=50`, `feature_fraction=0.85`.**
  Re-swept after removing the class weighting (the original sweep ran under
  spw=55 and every conclusion it drew was void). A second sweep over
  `min_data_in_leaf` × `feature_fraction` confirmed this cell is the optimum —
  though the whole grid spans only 0.0028, so treat it as a plateau, not a peak.
- **No graph/component features.** Removing them cost −0.0004, i.e. nothing.
  Dropped on parsimony, not because they were shown to hurt.
- **Rounds from a Jul 1–15 probe, ×1.1, then 5-seed averaging.**

## Measurement discipline — read before trusting any change

**Seed-noise std on the recent windows is 0.0020.** This was measured only
after seven consecutive experiments had produced deltas of 0.0015–0.002 — all
at or below the noise floor, all meaningless. A change counts only if it clears
**+0.004** on the recent-window mean, improves at least two of three windows,
and drops none by more than 0.003.

**Select on the recent windows, never the 4-fold mean.** The 4-fold mean (~0.73)
is dominated by three easy-regime folds that look nothing like the test period.

## Known limits

- Test runs **Jul 16 – Sep 15**, up to 62 days past the end of training, and
  accuracy decays **−0.019 AP over 60 days** of staleness. Our Jul 1–15 probe is
  a gap-0 window, so it is the most optimistic estimate available and **it
  overstates what late test rows will score**.
- The test period sits entirely after the early-July regime change, and we hold
  only ~1–2 weeks of post-change training data out of 6.5 months.

## Running it

Runs anywhere `train.csv` / `test.csv` / `sample_submission.csv` are present —
locally, or on Kaggle after attaching the competition. Path detection handles
both Kaggle mount layouts. CPU only, roughly 40–60 min end to end.
"""

BEST_MAIN = '''
# ---- reproduce the best model and write submission.csv --------------------
t0 = time.time()
train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
df = build_combined_frame(train, test)
assert_frame_sane(df)
del train, test
gc.collect()

df = build_features(df)
leakage_assertions(df)
feature_cols, cat_cols = get_feature_columns(df)

# arm C: everything except the graph/component family
cols = [c for c in feature_cols if "component_size_prior" not in c]
cc = [c for c in cat_cols if c in cols]
print(f"featurized={df.shape}, {len(cols)} model features ({time.time()-t0:.0f}s)", flush=True)

labeled = df[~df["is_test"]]
test_df = df[df["is_test"]]
frame_sorted = bool(df[TIME_COL].is_monotonic_increasing)
del df
gc.collect()

BEST_PARAMS = dict(
    objective="binary", metric="None", verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)

# 1) probe on a held-out Jul 1-15 tail to pick the round count
hold_start = pd.Timestamp("2026-07-01")
fit = labeled.loc[labeled[TIME_COL] < hold_start]
hold = labeled.loc[labeled[TIME_COL] >= hold_start]
X_f, y_f = prepare_lgb_frame(fit, cols, cc), fit[LABEL_COL].astype(int)
X_h, y_h = prepare_lgb_frame(hold, cols, cc), hold[LABEL_COL].astype(int)
probe = lgb.train(
    dict(BEST_PARAMS, seed=0, bagging_seed=0, feature_fraction_seed=0),
    lgb.Dataset(X_f, label=y_f, categorical_feature=cc, free_raw_data=False),
    num_boost_round=4000,
    valid_sets=[lgb.Dataset(X_h, label=y_h, categorical_feature=cc, free_raw_data=False)],
    feval=make_pr_auc_feval(y_h.values, seed=0),
    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)],
)
tail_ap = average_precision_score(y_h, probe.predict(X_h, num_iteration=probe.best_iteration))
rounds = int(round(probe.best_iteration * 1.1))
print(f"held-out(Jul01-15) PR-AUC = {tail_ap:.4f}   rounds -> {rounds}", flush=True)
print("   (expect ~0.5252; +/- 0.0020 is seed noise, so 0.523-0.527 reproduces)")
del X_f, X_h, probe
gc.collect()

# 2) refit on ALL labelled data at that round count, averaged over 5 seeds
X_full, y_full = prepare_lgb_frame(labeled, cols, cc), labeled[LABEL_COL].astype(int)
X_test = prepare_lgb_frame(test_df, cols, cc)
ids = test_df["transaction_id"].values
ds = lgb.Dataset(X_full, label=y_full, categorical_feature=cc, free_raw_data=False)
preds = np.mean([
    lgb.train(dict(BEST_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s),
              ds, num_boost_round=rounds).predict(X_test)
    for s in (0, 1, 2, 3, 4)
], axis=0)

# 3) checks that must hold before this file is worth submitting
assert frame_sorted, "combined frame must be time-sorted for the prior-only features to be valid"
assert LABEL_COL not in X_test.columns
for c in ID_COLS:
    assert c not in X_test.columns, f"banned raw ID column {c} reached the model"
assert not np.isnan(preds).any() and np.all((preds >= 0) & (preds <= 1))

sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": ids, "fraud": preds})
sub = sub.set_index("transaction_id").loc[sample["transaction_id"]].reset_index()
assert list(sub["transaction_id"]) == list(sample["transaction_id"]), "row order must match sample_submission"
assert len(sub) == len(sample)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\\nwrote submission.csv  rows={len(sub)}  mean={sub['fraud'].mean():.5f}  "
      f"({time.time()-t0:.0f}s)")
'''

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
