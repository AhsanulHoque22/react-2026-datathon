"""Hyperparameter tuning on Full Train model using Top 160 features."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("Loading Top 160 features...")
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

ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
ds_hold = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_full, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

grid = [
    {"num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.85, "lr": 0.020, "tag": "Incumbent (L=127, M=50, FF=0.85, lr=0.02)"},
    {"num_leaves": 95,  "min_data_in_leaf": 80, "feature_fraction": 0.80, "lr": 0.020, "tag": "Reg A (L=95, M=80, FF=0.80, lr=0.02)"},
    {"num_leaves": 150, "min_data_in_leaf": 80, "feature_fraction": 0.80, "lr": 0.020, "tag": "Deep Reg (L=150, M=80, FF=0.80, lr=0.02)"},
    {"num_leaves": 127, "min_data_in_leaf": 80, "feature_fraction": 0.80, "lr": 0.015, "tag": "Lower LR (L=127, M=80, FF=0.80, lr=0.015)"},
    {"num_leaves": 63,  "min_data_in_leaf": 80, "feature_fraction": 0.85, "lr": 0.020, "tag": "Compact (L=63, M=80, FF=0.85, lr=0.02)"},
]

results = []
print(f"\nEvaluating {len(grid)} Full Train candidates...")
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
        params, ds_full, num_boost_round=3000, valid_sets=[ds_hold], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = booster.predict(X_hold, num_iteration=booster.best_iteration)
    sc = average_precision_score(y_hold, preds)
    dt = time.time() - t0
    print(f"[{i+1}/{len(grid)}] {g['tag']:46s} | PR-AUC: {sc:.4f} | best_iter: {booster.best_iteration:4d} | time: {dt:.1f}s")
    results.append({**g, "pr_auc": sc, "best_iter": booster.best_iteration, "time": dt})

res_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n" + "="*80)
print("FULL TRAIN TUNING RANKING:")
print("="*80)
print(res_df[["tag", "pr_auc", "best_iter", "time"]].to_string(index=False))
res_df.to_csv(PROCESSED_DIR / "tune_full_train_results.csv", index=False)
