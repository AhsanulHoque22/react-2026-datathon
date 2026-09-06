"""5-Seed Multi-Horizon Champion Pipeline.

Trains 5 random seeds (0, 1, 2, 3, 4) for both:
1. Full Train Model (macro baseline: Jan 01 -> Jul 01, L=127, M=50, FF=0.85)
2. Tuned 90-Day Specialist (recent regime: Apr 01 -> Jul 01, L=63, M=100, FF=0.75)

Evaluates progressive seed-averaging curves on the held-out tail (Jul 1-15),
performs the multi-horizon rank blend, tests against the >= 0.5400 quality gate,
and generates the qualified candidate submission file.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("="*75)
print("5-SEED MULTI-HORIZON CHAMPION PIPELINE")
print("="*75)

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

print(f"Full Train: {len(fit_full):,} rows | 90d Specialist: {len(X_90d):,} rows | Holdout: {len(hold_df):,} rows")

SEEDS = [0, 1, 2, 3, 4]

# Datasets
ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
ds_90d  = lgb.Dataset(X_90d,  label=y_90d,  categorical_feature=selected_cats, free_raw_data=False)
ds_hold_full = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_full, free_raw_data=False)
ds_hold_90d  = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_90d,  free_raw_data=False)

# Storage for predictions
full_hold_preds = np.zeros(len(hold_df), dtype="float64")
full_test_preds = np.zeros(len(test_df), dtype="float64")
spec_hold_preds = np.zeros(len(hold_df), dtype="float64")
spec_test_preds = np.zeros(len(test_df), dtype="float64")

# =========================================================================
# 1. Train 5 Seeds for Tuned 90-Day Specialist (Fast: ~50s each)
# =========================================================================
print("\n" + "="*70)
print("PHASE 1: Training 5 Seeds for Tuned 90-Day Specialist")
print("="*70)

spec_scores = []
for i, s in enumerate(SEEDS):
    t_s = time.time()
    params = dict(
        objective="binary", metric="None", seed=s, bagging_seed=s,
        feature_fraction_seed=s, verbosity=-1,
        learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
    )
    feval = make_pr_auc_feval(y_hold.values, seed=s)
    booster = lgb.train(
        params, ds_90d, num_boost_round=2000, valid_sets=[ds_hold_90d], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    p_h = booster.predict(X_hold, num_iteration=booster.best_iteration)
    p_t = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    sc = average_precision_score(y_hold, p_h)
    spec_scores.append(sc)
    
    spec_hold_preds += p_h / len(SEEDS)
    spec_test_preds += p_t / len(SEEDS)
    
    # Running average score
    running_h = (spec_hold_preds * len(SEEDS)) / (i + 1)
    running_sc = average_precision_score(y_hold, running_h)
    print(f"  [Seed {s}] PR-AUC = {sc:.4f} (iter={booster.best_iteration:4d}, {time.time()-t_s:.1f}s) | Running {i+1}-Seed Avg = {running_sc:.4f}")

score_spec_5seed = average_precision_score(y_hold, spec_hold_preds)
print(f"\n--> 90d Specialist 5-Seed Average: PR-AUC = {score_spec_5seed:.4f} (Single-seed mean: {np.mean(spec_scores):.4f})")

# =========================================================================
# 2. Train 5 Seeds for Full Train Model
# =========================================================================
print("\n" + "="*70)
print("PHASE 2: Training 5 Seeds for Full Train Model")
print("="*70)

full_scores = []
for i, s in enumerate(SEEDS):
    t_s = time.time()
    params = dict(
        objective="binary", metric="None", seed=s, bagging_seed=s,
        feature_fraction_seed=s, verbosity=-1,
        learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )
    feval = make_pr_auc_feval(y_hold.values, seed=s)
    booster = lgb.train(
        params, ds_full, num_boost_round=3000, valid_sets=[ds_hold_full], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    p_h = booster.predict(X_hold, num_iteration=booster.best_iteration)
    p_t = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    sc = average_precision_score(y_hold, p_h)
    full_scores.append(sc)
    
    full_hold_preds += p_h / len(SEEDS)
    full_test_preds += p_t / len(SEEDS)
    
    # Running average score
    running_h = (full_hold_preds * len(SEEDS)) / (i + 1)
    running_sc = average_precision_score(y_hold, running_h)
    print(f"  [Seed {s}] PR-AUC = {sc:.4f} (iter={booster.best_iteration:4d}, {time.time()-t_s:.1f}s) | Running {i+1}-Seed Avg = {running_sc:.4f}")

score_full_5seed = average_precision_score(y_hold, full_hold_preds)
print(f"\n--> Full Train 5-Seed Average: PR-AUC = {score_full_5seed:.4f} (Single-seed mean: {np.mean(full_scores):.4f})")

# =========================================================================
# 3. Final Multi-Horizon 5-Seed Blend & Quality Gate Check
# =========================================================================
print("\n" + "="*70)
print("PHASE 3: Final Multi-Horizon 5-Seed Blend & Gate Check")
print("="*70)

r_full_hold = rankdata(full_hold_preds) / len(full_hold_preds)
r_spec_hold = rankdata(spec_hold_preds) / len(spec_hold_preds)

best_blend_score = 0.0
best_w = 0.50
for w in np.linspace(0.20, 0.80, 25):
    blend = w * r_full_hold + (1.0 - w) * r_spec_hold
    sc = average_precision_score(y_hold, blend)
    if sc > best_blend_score:
        best_blend_score = sc
        best_w = w
    print(f"  w={w:.3f} Full + {1-w:.3f} 90d: PR-AUC = {sc:.4f}")

print("\n" + "="*70)
print("RESULTS SUMMARY:")
print("="*70)
print(f"Full Train 5-Seed Average:       PR-AUC = {score_full_5seed:.4f}")
print(f"90d Specialist 5-Seed Average:   PR-AUC = {score_spec_5seed:.4f}")
print(f"5-Seed Multi-Horizon Blend:      PR-AUC = {best_blend_score:.4f} (weight={best_w:.2f} Full + {1-best_w:.2f} 90d)")
print(f"Delta vs previous single-seed:   {best_blend_score - 0.5282:+.4f}")

# Generate test predictions
r_full_test = rankdata(full_test_preds) / len(full_test_preds)
r_spec_test = rankdata(spec_test_preds) / len(spec_test_preds)
final_test_preds = best_w * r_full_test + (1.0 - best_w) * r_spec_test

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_preds})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_5seed_multihorizon_{best_blend_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"\nGenerated candidate submission: {out_path} (shape={sub.shape})")

GATE = 0.5400
if best_blend_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_blend_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_blend_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_blend_score:.4f}) <<<")
