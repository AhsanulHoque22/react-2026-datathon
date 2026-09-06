"""Generate the test submission for the 0.5288 champion tuned blend."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]

labeled = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()

holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

X_full = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test = prepare_lgb_frame(test_df, selected_cols, selected_cats)

# 1. Full Train Model
ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
print("Fitting Full Train model (iter=882)...")
t0 = time.time()
b_full = lgb.train(p_full, ds_full, num_boost_round=882)
preds_full_hold = b_full.predict(X_hold)
preds_full_test = b_full.predict(X_test)
print(f"Full Train: PR-AUC = {average_precision_score(y_hold, preds_full_hold):.4f} ({time.time()-t0:.1f}s)")

# 2. Tuned 90d Specialist
mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d = X_full.loc[mask_90d]
y_90d = y_full.loc[mask_90d]
ds_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)

p_90d = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
print("Fitting Tuned 90d Specialist (iter=433)...")
t0 = time.time()
b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
preds_90d_hold = b_90d.predict(X_hold)
preds_90d_test = b_90d.predict(X_test)
print(f"Tuned 90d:   PR-AUC = {average_precision_score(y_hold, preds_90d_hold):.4f} ({time.time()-t0:.1f}s)")

# Rank Blend
r_full_hold = rankdata(preds_full_hold) / len(preds_full_hold)
r_90d_hold = rankdata(preds_90d_hold) / len(preds_90d_hold)
blend_hold = 0.48 * r_full_hold + 0.52 * r_90d_hold
score = average_precision_score(y_hold, blend_hold)
print(f"\nFinal Blended PR-AUC = {score:.4f}")

r_full_test = rankdata(preds_full_test) / len(preds_full_test)
r_90d_test = rankdata(preds_90d_test) / len(preds_90d_test)
final_test_preds = 0.48 * r_full_test + 0.52 * r_90d_test

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_preds})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

out_path = SUBMISSIONS_DIR / f"CANDIDATE_tuned_horizon_{score:.4f}.csv"
sub.to_csv(out_path, index=False)
print(f"Saved candidate submission: {out_path} (shape={sub.shape})")
