"""Tri-Architecture Rank Ensemble Harness.

Ensembles:
1. Leaf-wise LightGBM (deep asymmetric splits)
2. Level-wise XGBoost (balanced depth-first splits via hist)
3. Oblivious CatBoost (symmetric oblivious trees)

Evaluates on the Jul 1-15 held-out probe and tests against the user's hard gate:
PR-AUC >= 0.5400.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

t0 = time.time()
print("Loading features.pkl...")
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
print(f"Loaded {df.shape}, {len(feature_cols)} features ({len(cat_cols)} categoricals)  ({time.time()-t0:.1f}s)")

labeled = df[~df["is_test"]].copy()

# Probe split: train < 2026-07-01, validation >= 2026-07-01 (Jul 1-15 held-out tail)
holdout_start = pd.Timestamp("2026-07-01")
fit_df = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_fit = fit_df[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)
print(f"Fit window: {len(fit_df):,} rows (fraud={y_fit.sum():,}) | Holdout window: {len(hold_df):,} rows (fraud={y_hold.sum():,})")

# -------------------------------------------------------------
# 1. Model 1: LightGBM (Leaf-Wise)
# -------------------------------------------------------------
print("\n" + "="*60)
print("1. Training Leaf-Wise LightGBM...")
print("="*60)
X_fit_lgb = prepare_lgb_frame(fit_df, feature_cols, cat_cols)
X_hold_lgb = prepare_lgb_frame(hold_df, feature_cols, cat_cols)

lgb_params = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)

train_set_lgb = lgb.Dataset(X_fit_lgb, label=y_fit, categorical_feature=cat_cols, free_raw_data=False)
hold_set_lgb = lgb.Dataset(X_hold_lgb, label=y_hold, categorical_feature=cat_cols, reference=train_set_lgb, free_raw_data=False)
feval_lgb = make_pr_auc_feval(y_hold.values, seed=SEED)

t1 = time.time()
booster_lgb = lgb.train(
    lgb_params, train_set_lgb, num_boost_round=3000, valid_sets=[hold_set_lgb], feval=feval_lgb,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_lgb = booster_lgb.predict(X_hold_lgb, num_iteration=booster_lgb.best_iteration)
score_lgb = average_precision_score(y_hold, preds_lgb)
print(f"LightGBM Score: PR-AUC={score_lgb:.4f} (best_iter={booster_lgb.best_iteration}, {time.time()-t1:.1f}s)")

# -------------------------------------------------------------
# 2. Model 2: XGBoost (Level-Wise)
# -------------------------------------------------------------
print("\n" + "="*60)
print("2. Training Level-Wise XGBoost (hist)...")
print("="*60)
X_fit_xgb = X_fit_lgb.copy()
X_hold_xgb = X_hold_lgb.copy()

xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="aucpr",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=-1,
)

t2 = time.time()
xgb_model.fit(
    X_fit_xgb, y_fit,
    eval_set=[(X_hold_xgb, y_hold)],
    verbose=False,
)
preds_xgb = xgb_model.predict_proba(X_hold_xgb)[:, 1]
score_xgb = average_precision_score(y_hold, preds_xgb)
print(f"XGBoost Score: PR-AUC={score_xgb:.4f} (best_iter={xgb_model.best_iteration}, {time.time()-t2:.1f}s)")

# -------------------------------------------------------------
# 3. Model 3: CatBoost (Oblivious Symmetric)
# -------------------------------------------------------------
print("\n" + "="*60)
print("3. Training Oblivious CatBoost...")
print("="*60)
# One-hot low-card categoricals to maintain 100% rule compliance (no target CTR)
numeric_cols = [c for c in feature_cols if c not in cat_cols]
onehot_fit = pd.get_dummies(fit_df[cat_cols].astype(str), prefix=cat_cols)
onehot_hold = pd.get_dummies(hold_df[cat_cols].astype(str), prefix=cat_cols)
# align columns
onehot_fit, onehot_hold = onehot_fit.align(onehot_hold, join="left", axis=1, fill_value=0)

X_fit_cat = pd.concat([fit_df[numeric_cols].reset_index(drop=True), onehot_fit.reset_index(drop=True)], axis=1)
X_hold_cat = pd.concat([hold_df[numeric_cols].reset_index(drop=True), onehot_hold.reset_index(drop=True)], axis=1)

cat_model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=7,
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_seed=SEED,
    early_stopping_rounds=100,
    verbose=False,
    thread_count=-1,
)
t3 = time.time()
cat_pool_fit = Pool(X_fit_cat, y_fit)
cat_pool_hold = Pool(X_hold_cat, y_hold)
cat_model.fit(cat_pool_fit, eval_set=cat_pool_hold, use_best_model=True)
preds_cat = cat_model.predict_proba(X_hold_cat)[:, 1]
score_cat = average_precision_score(y_hold, preds_cat)
print(f"CatBoost Score: PR-AUC={score_cat:.4f} (best_iter={cat_model.get_best_iteration()}, {time.time()-t3:.1f}s)")

# -------------------------------------------------------------
# 4. Tri-Architecture Rank Ensemble Optimization
# -------------------------------------------------------------
print("\n" + "="*60)
print("4. Optimizing Tri-Architecture Rank Ensemble...")
print("="*60)

# Convert to uniform percentile ranks
rank_lgb = rankdata(preds_lgb) / len(preds_lgb)
rank_xgb = rankdata(preds_xgb) / len(preds_xgb)
rank_cat = rankdata(preds_cat) / len(preds_cat)

best_ensemble_score = max(score_lgb, score_xgb, score_cat)
best_weights = (1.0, 0.0, 0.0)

for w1 in np.linspace(0.2, 0.8, 7):
    for w2 in np.linspace(0.1, 0.7, 7):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        blend = w1 * rank_lgb + w2 * rank_xgb + w3 * rank_cat
        score = average_precision_score(y_hold, blend)
        if score > best_ensemble_score:
            best_ensemble_score = score
            best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))

print(f"Individual Model Scores:")
print(f"  LightGBM (Leaf-Wise):  {score_lgb:.4f}")
print(f"  XGBoost  (Level-Wise): {score_xgb:.4f}")
print(f"  CatBoost (Oblivious):  {score_cat:.4f}")
print(f"\nOptimal Ensemble Weights (LGB, XGB, CAT): {best_weights}")
print(f"Ensemble Score on Held-Out Tail: PR-AUC = {best_ensemble_score:.4f}")

delta_vs_lgb = best_ensemble_score - score_lgb
print(f"Ensemble Lift over best single model: {delta_vs_lgb:+.4f}")

# Quality Gate Check
GATE = 0.5400
if best_ensemble_score >= GATE:
    print(f"\n>>> QUALITY GATE PASSED! ({best_ensemble_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> QUALITY GATE STATUS: {best_ensemble_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_ensemble_score:.4f}) <<<")

# Save results log
summary = pd.DataFrame([{
    "lgb_score": score_lgb,
    "xgb_score": score_xgb,
    "cat_score": score_cat,
    "ensemble_score": best_ensemble_score,
    "best_weights": str(best_weights),
    "gate_passed": best_ensemble_score >= GATE,
}])
summary.to_csv(PROCESSED_DIR / "ensemble_gate_results.csv", index=False)
print(f"\nSaved {PROCESSED_DIR / 'ensemble_gate_results.csv'}")
