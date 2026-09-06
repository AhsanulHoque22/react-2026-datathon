"""Systematic Feature Pruning Search across Feature Count Brackets.

Evaluates subsets:
- All features (~219)
- Top 160 features
- Top 120 features
- Top 90 features
- Top 70 features
- Top 50 features

Target: Strip dead/diluting features, restore deep tree growth, and test if
PR-AUC improves toward the >= 0.5400 quality gate.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("Loading features.pkl and feature_importance_final.csv...")
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
imp_df = pd.read_csv(PROCESSED_DIR / "feature_importance_final.csv", index_col=0)

# Extract features ranked by gain
ranked_features = [f for f in imp_df.index if f in df.columns]
all_feature_cols, all_cat_cols = get_feature_columns(df)
# make sure any unranked features are appended
for f in all_feature_cols:
    if f not in ranked_features:
        ranked_features.append(f)

print(f"Total available features: {len(ranked_features)}")

# Probe split: train < 2026-07-01, validation >= 2026-07-01 (Jul 1-15 held-out tail)
holdout_start = pd.Timestamp("2026-07-01")
labeled = df[~df["is_test"]].copy()
fit_df = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_fit = fit_df[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)
print(f"Fit rows: {len(fit_df):,} | Holdout rows: {len(hold_df):,}\n")

# Candidate brackets to search
brackets = [
    ("All Features", len(ranked_features)),
    ("Top 160", 160),
    ("Top 120", 120),
    ("Top 90", 90),
    ("Top 70", 70),
    ("Top 50", 50),
    ("Top 35", 35),
]

results = []

for name, n_feat in brackets:
    selected_cols = ranked_features[:n_feat]
    selected_cats = [c for c in CAT_COLS if c in selected_cols]
    
    X_fit = prepare_lgb_frame(fit_df, selected_cols, selected_cats)
    X_hold = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
    
    train_set = lgb.Dataset(X_fit, label=y_fit, categorical_feature=selected_cats, free_raw_data=False)
    hold_set = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=train_set, free_raw_data=False)
    feval = make_pr_auc_feval(y_hold.values, seed=SEED)
    
    params = dict(
        objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
        feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=0.02, num_leaves=127,
        # As feature count drops, increase feature_fraction to avoid missing key signals
        feature_fraction=0.85 if n_feat > 100 else 0.95,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )
    
    t0 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = booster.predict(X_hold, num_iteration=booster.best_iteration)
    pr_auc = average_precision_score(y_hold, preds)
    elapsed = time.time() - t0
    
    results.append({
        "bracket": name,
        "n_features": n_feat,
        "n_categoricals": len(selected_cats),
        "best_iter": booster.best_iteration,
        "pr_auc": pr_auc,
        "seconds": round(elapsed, 1),
    })
    print(f"[{name:14s}] features={n_feat:3d} | PR-AUC = {pr_auc:.4f} | best_iter = {booster.best_iteration:4d} | time = {elapsed:4.1f}s", flush=True)

res_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FEATURE PRUNING SEARCH SUMMARY")
print("="*70)
print(res_df.to_string(index=False))

best_row = res_df.sort_values("pr_auc", ascending=False).iloc[0]
print(f"\n>>> Best Subset: {best_row['bracket']} ({best_row['n_features']} features) with PR-AUC = {best_row['pr_auc']:.4f} (best_iter={best_row['best_iter']}) <<<")

# Save selected best features list
best_features = ranked_features[:int(best_row["n_features"])]
pd.Series(best_features).to_csv(PROCESSED_DIR / "optimal_pruned_features.csv", index=False)
res_df.to_csv(PROCESSED_DIR / "feature_pruning_results.csv", index=False)
print(f"Saved optimal features list to {PROCESSED_DIR / 'optimal_pruned_features.csv'}")
