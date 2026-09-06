"""Hyperparameter Tuning for the 90-Day Specialist Model.

Tests combinations of num_leaves, min_data_in_leaf, and feature_fraction
specifically calibrated for the 346k-row 90-day window (Apr 01 -> Jul 01)
to maximize held-out PR-AUC (Jul 01 -> Jul 15).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("Loading features and Top 160 core...")
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

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d = X_full.loc[mask_90d]
y_90d = y_full.loc[mask_90d]

train_set_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
hold_set = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=train_set_90d, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

grid = [
    {"num_leaves": 127, "min_data_in_leaf": 50,  "feature_fraction": 0.85, "lr": 0.02, "tag": "Baseline 90d (L=127, M=50, FF=0.85)"},
    {"num_leaves": 95,  "min_data_in_leaf": 80,  "feature_fraction": 0.80, "lr": 0.02, "tag": "Regularized A (L=95, M=80, FF=0.80)"},
    {"num_leaves": 63,  "min_data_in_leaf": 100, "feature_fraction": 0.75, "lr": 0.02, "tag": "Regularized B (L=63, M=100, FF=0.75)"},
    {"num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.75, "lr": 0.02, "tag": "Deep Reg (L=127, M=100, FF=0.75)"},
    {"num_leaves": 95,  "min_data_in_leaf": 50,  "feature_fraction": 0.75, "lr": 0.02, "tag": "Medium Leaves (L=95, M=50, FF=0.75)"},
    {"num_leaves": 63,  "min_data_in_leaf": 50,  "feature_fraction": 0.85, "lr": 0.02, "tag": "Compact Leaves (L=63, M=50, FF=0.85)"},
    {"num_leaves": 95,  "min_data_in_leaf": 80,  "feature_fraction": 0.80, "lr": 0.015, "tag": "Lower LR (L=95, M=80, FF=0.80, lr=0.015)"},
]

results = []
print(f"\nEvaluating {len(grid)} hyperparameter candidates on 90-Day Specialist (346k rows)...")
for i, g in enumerate(grid):
    t0 = time.time()
    params = dict(
        objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
        feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=g["lr"], num_leaves=g["num_leaves"],
        feature_fraction=g["feature_fraction"],
        bagging_fraction=0.85, bagging_freq=1,
        min_data_in_leaf=g["min_data_in_leaf"],
    )
    booster = lgb.train(
        params, train_set_90d, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = booster.predict(X_hold, num_iteration=booster.best_iteration)
    sc = average_precision_score(y_hold, preds)
    dt = time.time() - t0
    print(f"[{i+1}/{len(grid)}] {g['tag']:42s} | PR-AUC: {sc:.4f} | best_iter: {booster.best_iteration:4d} | time: {dt:.1f}s")
    results.append({**g, "pr_auc": sc, "best_iter": booster.best_iteration, "time": dt})

res_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n" + "="*80)
print("90-DAY SPECIALIST TUNING RANKING:")
print("="*80)
print(res_df[["tag", "pr_auc", "best_iter", "time"]].to_string(index=False))
res_df.to_csv(PROCESSED_DIR / "tune_90d_specialist_results.csv", index=False)
