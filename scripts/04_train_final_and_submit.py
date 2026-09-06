"""Train final LightGBM on all of train.csv (recency-weighting tested and
rejected -- no measured benefit, see scripts/03_recency_experiment.py), predict
on test.csv, run the full pre-submission assertions checklist, then submit via
Kaggle CLI.

Iteration count: early-stopped on a held-out tail (Jul 1-15, matching CV fold 3)
to pick best_iteration, then refit on the FULL train set (including that tail)
with a fixed round count -- standard practice to use all available data for
the model that actually generates the submission.
"""
import sys, time, subprocess, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from src.config import (
    PROCESSED_DIR, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV, TIME_COL, LABEL_COL,
    SEED, ID_COLS,
)
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval

t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
print(f"Loaded {df.shape}, {len(feature_cols)} features ({len(cat_cols)} categorical)  ({time.time()-t0:.1f}s)")

# Captured before `df` is freed below, so the pre-submission assertion can
# still check it without holding the whole frame in memory.
frame_time_sorted = bool(df[TIME_COL].is_monotonic_increasing)

labeled = df[~df["is_test"]]
test_df = df[df["is_test"]]

# --- Step 1: find a good iteration count via a held-out tail ---
holdout_start = pd.Timestamp("2026-07-01")
fit_df = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

X_fit = prepare_lgb_frame(fit_df, feature_cols, cat_cols)
y_fit = fit_df[LABEL_COL].astype(int)
X_hold = prepare_lgb_frame(hold_df, feature_cols, cat_cols)
y_hold = hold_df[LABEL_COL].astype(int)

# No scale_pos_weight -- see src/model.py; it measurably hurt this rank metric.
base_params = dict(
    objective="binary", metric="None",
    seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
SEEDS = [0, 1, 2, 3, 4]  # seed-averaged final fit: seed std measured at 0.0020

print("Finding iteration count on held-out tail (Jul 1-15)...")
t1 = time.time()
train_set = lgb.Dataset(X_fit, label=y_fit, categorical_feature=cat_cols, free_raw_data=False)
hold_set = lgb.Dataset(X_hold, label=y_hold, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)
booster_probe = lgb.train(
    base_params, train_set, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
best_iter = booster_probe.best_iteration
probe_preds = booster_probe.predict(X_hold, num_iteration=best_iter)
probe_pr_auc = average_precision_score(y_hold, probe_preds)
final_num_rounds = int(round(best_iter * 1.1))
print(f"Held-out PR-AUC={probe_pr_auc:.4f}  best_iter={best_iter} -> final_num_rounds={final_num_rounds}  ({time.time()-t1:.1f}s)")

# --- Step 2: refit on ALL of train.csv with the fixed round count ---
print(f"Refitting on full train.csv, averaged over {len(SEEDS)} seeds...")
t1 = time.time()
# Free the probe-stage frames before building the full ones -- this box has
# 7GB RAM and an earlier run was OOM-killed at exactly this point.
del X_fit, X_hold, train_set, hold_set, booster_probe
gc.collect()

X_train_full = prepare_lgb_frame(labeled, feature_cols, cat_cols)
y_train_full = labeled[LABEL_COL].astype(int)
X_test = prepare_lgb_frame(test_df, feature_cols, cat_cols)
del df
gc.collect()

full_train_set = lgb.Dataset(X_train_full, label=y_train_full, categorical_feature=cat_cols, free_raw_data=False)
seed_preds = []
final_booster = None
for s in SEEDS:
    p = dict(base_params, seed=s, bagging_seed=s, feature_fraction_seed=s)
    bst = lgb.train(p, full_train_set, num_boost_round=final_num_rounds)
    seed_preds.append(bst.predict(X_test))
    final_booster = bst  # keep the last for feature-importance reporting
    print(f"  seed {s} done ({time.time()-t1:.0f}s)", flush=True)
preds = np.mean(seed_preds, axis=0)
print(f"Refit done  ({time.time()-t1:.1f}s)")

# --- Pre-submission assertions checklist (PLAN.md) ---
assert frame_time_sorted, "frame not time-sorted"
assert LABEL_COL not in X_test.columns, "fraud column leaked into inference feature matrix"
for c in ID_COLS:
    assert c not in X_test.columns, f"banned raw ID column {c} leaked into feature matrix"
assert np.all((preds >= 0) & (preds <= 1)), "predictions out of [0,1] range"
assert not np.isnan(preds).any(), "NaN predictions"

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": preds})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
assert list(sub["transaction_id"]) == list(sample_sub["transaction_id"]), "row order mismatch vs sample_submission"
assert sub.shape == sample_sub.shape
assert sub["fraud"].between(0, 1).all()
assert not sub["fraud"].isna().any()
assert (sub["fraud"] != sub["fraud"].round().astype(int)).any(), "predictions look thresholded to hard 0/1 labels"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
out_path = SUBMISSIONS_DIR / "submission_final.csv"
sub.to_csv(out_path, index=False)
print(f"\nAll assertions passed. Wrote {out_path}  shape={sub.shape}  (total {time.time()-t0:.1f}s)")
print(sub.head())
print(sub["fraud"].describe())

# --- Feature importance (for the write-up / sanity check against the banned rule) ---
imp = pd.Series(final_booster.feature_importance(importance_type="gain"), index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features by gain:")
print(imp.head(15))
imp.to_csv(PROCESSED_DIR / "feature_importance_final.csv")
