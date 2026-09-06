"""Test blend of Full Train (0.5269) + Tuned 90d Specialist (0.5272)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]

labeled = df[~df["is_test"]].copy()
holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

X_full = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold = prepare_lgb_frame(hold_df, selected_cols, selected_cats)

# 1. Full Train Baseline
ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
ds_hold = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_full, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
print("Fitting Full Train (iter=882)...")
b_full = lgb.train(p_full, ds_full, num_boost_round=882)
preds_full = b_full.predict(X_hold)
sc_full = average_precision_score(y_hold, preds_full)
print(f"Full Train: PR-AUC = {sc_full:.4f}")

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
b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
preds_90d = b_90d.predict(X_hold)
sc_90d = average_precision_score(y_hold, preds_90d)
print(f"Tuned 90d:   PR-AUC = {sc_90d:.4f}")

r_full = rankdata(preds_full) / len(preds_full)
r_90d = rankdata(preds_90d) / len(preds_90d)

print("\n--- Blend Sweep (w * Full + (1-w) * Tuned 90d) ---")
best_sc = 0.0
best_w = 0.5
for w in np.linspace(0.2, 0.8, 25):
    blend = w * r_full + (1.0 - w) * r_90d
    sc = average_precision_score(y_hold, blend)
    if sc > best_sc:
        best_sc = sc
        best_w = w
    print(f"  w={w:.3f} Full + {1-w:.3f} 90d: PR-AUC = {sc:.4f}")

print(f"\n>>> Best Tuned Blend: PR-AUC = {best_sc:.4f} (w={best_w:.2f}) <<<")
print(f"Delta vs previous best 0.5282: {best_sc - 0.528157:+.4f}")
