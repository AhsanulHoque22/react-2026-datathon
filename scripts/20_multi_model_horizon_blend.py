"""Multi-Model and Multi-Horizon Ensembling Experiment.

Evaluates LightGBM and XGBoost across two horizons:
1. Full Train (Jan 01 -> Jul 01)
2. 90-Day Specialist (Apr 01 -> Jul 01)

Tests whether ensembling across BOTH model architectures and time horizons
breaks past the 0.5282 mark toward the >= 0.5400 quality gate.
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

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("Loading features.pkl and optimal_pruned_features.csv...")
t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]
print(f"Loaded {df.shape} in {time.time()-t0:.1f}s | Top 160 features ({len(selected_cats)} categoricals)")

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

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d = X_full.loc[mask_90d]
y_90d = y_full.loc[mask_90d]

print(f"Full Train: {len(fit_full):,} rows | 90d Specialist: {len(X_90d):,} rows | Holdout: {len(hold_df):,} rows\n")

# Datasets for LightGBM
ds_lgb_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
ds_lgb_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
ds_lgb_hold = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_lgb_full, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

# DMatrices for XGBoost
dm_xgb_full = xgb.DMatrix(X_full, label=y_full, enable_categorical=True)
dm_xgb_90d = xgb.DMatrix(X_90d, label=y_90d, enable_categorical=True)
dm_xgb_hold = xgb.DMatrix(X_hold, label=y_hold, enable_categorical=True)
dm_xgb_test = xgb.DMatrix(X_test, enable_categorical=True)

lgb_params = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)

xgb_params = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 30,
    "nthread": 8,
    "seed": SEED,
}

# 1. Model 1: LightGBM Full Train
print("1/4 Training LightGBM Full Train...")
t1 = time.time()
b_lgb_full = lgb.train(
    lgb_params, ds_lgb_full, num_boost_round=3000, valid_sets=[ds_lgb_hold], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
p_lgb_full_hold = b_lgb_full.predict(X_hold, num_iteration=b_lgb_full.best_iteration)
p_lgb_full_test = b_lgb_full.predict(X_test, num_iteration=b_lgb_full.best_iteration)
sc_lgb_full = average_precision_score(y_hold, p_lgb_full_hold)
print(f"   LightGBM Full: PR-AUC = {sc_lgb_full:.4f} (iter={b_lgb_full.best_iteration}, {time.time()-t1:.1f}s)")

# 2. Model 2: LightGBM 90-Day Specialist
print("2/4 Training LightGBM 90d Specialist...")
t1 = time.time()
b_lgb_90d = lgb.train(
    lgb_params, ds_lgb_90d, num_boost_round=3000, valid_sets=[ds_lgb_hold], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
p_lgb_90d_hold = b_lgb_90d.predict(X_hold, num_iteration=b_lgb_90d.best_iteration)
p_lgb_90d_test = b_lgb_90d.predict(X_test, num_iteration=b_lgb_90d.best_iteration)
sc_lgb_90d = average_precision_score(y_hold, p_lgb_90d_hold)
print(f"   LightGBM 90d: PR-AUC = {sc_lgb_90d:.4f} (iter={b_lgb_90d.best_iteration}, {time.time()-t1:.1f}s)")

# 3. Model 3: XGBoost 90-Day Specialist
print("3/4 Training XGBoost 90d Specialist...")
t1 = time.time()
b_xgb_90d = xgb.train(
    xgb_params, dm_xgb_90d, num_boost_round=1500, evals=[(dm_xgb_hold, 'val')],
    early_stopping_rounds=80, verbose_eval=False,
)
p_xgb_90d_hold = b_xgb_90d.predict(dm_xgb_hold)
p_xgb_90d_test = b_xgb_90d.predict(dm_xgb_test)
sc_xgb_90d = average_precision_score(y_hold, p_xgb_90d_hold)
print(f"   XGBoost 90d: PR-AUC = {sc_xgb_90d:.4f} (iter={b_xgb_90d.best_iteration}, {time.time()-t1:.1f}s)")

# 4. Model 4: XGBoost Full Train
print("4/4 Training XGBoost Full Train...")
t1 = time.time()
b_xgb_full = xgb.train(
    xgb_params, dm_xgb_full, num_boost_round=1500, evals=[(dm_xgb_hold, 'val')],
    early_stopping_rounds=80, verbose_eval=False,
)
p_xgb_full_hold = b_xgb_full.predict(dm_xgb_hold)
p_xgb_full_test = b_xgb_full.predict(dm_xgb_test)
sc_xgb_full = average_precision_score(y_hold, p_xgb_full_hold)
print(f"   XGBoost Full: PR-AUC = {sc_xgb_full:.4f} (iter={b_xgb_full.best_iteration}, {time.time()-t1:.1f}s)")

# -------------------------------------------------------------------------
# Ensembling Analysis
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("ENSEMBLE EXPLORATION")
print("="*70)

r_lgb_full = rankdata(p_lgb_full_hold) / len(p_lgb_full_hold)
r_lgb_90d  = rankdata(p_lgb_90d_hold)  / len(p_lgb_90d_hold)
r_xgb_90d  = rankdata(p_xgb_90d_hold)  / len(p_xgb_90d_hold)
r_xgb_full = rankdata(p_xgb_full_hold) / len(p_xgb_full_hold)

# Check individual scores
models = {
    "LGB Full": (r_lgb_full, sc_lgb_full),
    "LGB 90d": (r_lgb_90d, sc_lgb_90d),
    "XGB Full": (r_xgb_full, sc_xgb_full),
    "XGB 90d": (r_xgb_90d, sc_xgb_90d),
}
for name, (_, s) in models.items():
    print(f"  {name:12s}: PR-AUC = {s:.4f}")

# Pairwise correlations among ranks
pred_matrix = np.column_stack([r_lgb_full, r_lgb_90d, r_xgb_full, r_xgb_90d])
cols = ["LGB_Full", "LGB_90d", "XGB_Full", "XGB_90d"]
corr_df = pd.DataFrame(np.corrcoef(pred_matrix.T), index=cols, columns=cols)
print("\nRank Correlations:\n", corr_df.round(3))

# Incumbent blend (LGB Full 60% + LGB 90d 40%)
b1 = 0.60 * r_lgb_full + 0.40 * r_lgb_90d
sc_b1 = average_precision_score(y_hold, b1)
print(f"\nIncumbent Blend (LGB Full 60% + LGB 90d 40%): PR-AUC = {sc_b1:.4f}")

# Grid search over simplex weights with step 0.05
best_score = sc_b1
best_weights = (0.6, 0.4, 0.0, 0.0)

steps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for w1 in steps:
    for w2 in steps:
        for w3 in steps:
            w4 = round(1.0 - (w1 + w2 + w3), 2)
            if w4 < 0 or w4 > 1.0:
                continue
            combo = w1 * r_lgb_full + w2 * r_lgb_90d + w3 * r_xgb_full + w4 * r_xgb_90d
            sc = average_precision_score(y_hold, combo)
            if sc > best_score:
                best_score = sc
                best_weights = (w1, w2, w3, w4)

print(f"\n>>> OPTIMAL MULTI-MODEL MULTI-HORIZON ENSEMBLE <<<")
print(f"PR-AUC: {best_score:.4f} (Delta vs Incumbent 0.5282: {best_score - sc_b1:+.4f})")
print(f"Weights: LGB_Full={best_weights[0]:.2f}, LGB_90d={best_weights[1]:.2f}, XGB_Full={best_weights[2]:.2f}, XGB_90d={best_weights[3]:.2f}")

# Save the predictions to candidate file
r_lgb_full_test = rankdata(p_lgb_full_test) / len(p_lgb_full_test)
r_lgb_90d_test  = rankdata(p_lgb_90d_test)  / len(p_lgb_90d_test)
r_xgb_full_test = rankdata(p_xgb_full_test) / len(p_xgb_full_test)
r_xgb_90d_test  = rankdata(p_xgb_90d_test)  / len(p_xgb_90d_test)

final_test_preds = (
    best_weights[0] * r_lgb_full_test +
    best_weights[1] * r_lgb_90d_test +
    best_weights[2] * r_xgb_full_test +
    best_weights[3] * r_xgb_90d_test
)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_preds})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_multimodel_multihorizon_{best_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"Generated candidate submission file: {out_path}")

GATE = 0.5400
if best_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_score:.4f}) <<<")
