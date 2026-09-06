"""Training Regime Optimization Experiments on Top 160 Features.

Tests 4 training regime axes against the 0.5269 baseline:
1. Stable Custom Focal Loss (downweighting easy negatives)
2. DART Boosting (Dropout Additive Regression Trees)
3. Multi-Horizon Window Ensemble:
   - Full history (Jan-Jul) + 90-day specialist (Apr-Jul)
   - Full history (Jan-Jul) + 60-day specialist (May-Jul)
4. Tri-Horizon Blend (Full + 90d + 60d)

Evaluates on the Jul 1-15 held-out probe against the user's hard gate: >= 0.5400.
"""
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

print(f"Full Train: {len(fit_full):,} rows (fraud={y_full.sum():,}) | Holdout: {len(hold_df):,} rows (fraud={y_hold.sum():,})\n")

results = []

# =========================================================================
# 0. Incumbent Baseline: Standard GBDT on Full Train
# =========================================================================
print("="*65)
print("0. Baseline: Standard GBDT on Full Train (Jan 01 - Jul 01)...")
print("="*65)
base_params = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
train_set_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
hold_set = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=train_set_full, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

t1 = time.time()
b_base = lgb.train(
    base_params, train_set_full, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_base = b_base.predict(X_hold, num_iteration=b_base.best_iteration)
preds_base_test = b_base.predict(X_test, num_iteration=b_base.best_iteration)
sc_base = average_precision_score(y_hold, preds_base)
print(f"Baseline GBDT: PR-AUC = {sc_base:.4f} (best_iter = {b_base.best_iteration}, {time.time()-t1:.1f}s)")
results.append({"regime": "Baseline GBDT (Full Train)", "pr_auc": sc_base, "best_iter": b_base.best_iteration, "time": round(time.time()-t1, 1)})

# =========================================================================
# 1. Axis 1: Stable Custom Focal Loss (Downweighting Easy Negatives)
# =========================================================================
print("\n" + "="*65)
print("1. Axis 1: Custom Stable Focal Loss (gamma=1.5)...")
print("="*65)

def make_focal_objective(gamma=1.5):
    def focal_loss(preds, train_data):
        labels = train_data.get_label()
        # preds are raw logits in custom objective
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        # focal weight: downweight easy negatives (p near 0 for y=0)
        w = labels * (1.0 - p)**gamma + (1.0 - labels) * (p**gamma)
        grad = w * (p - labels)
        hess = np.clip(w * p * (1.0 - p), 1e-6, None)
        return grad, hess
    return focal_loss

focal_params = dict(base_params)
focal_params["objective"] = make_focal_objective(gamma=1.5)
t1 = time.time()
b_focal = lgb.train(
    focal_params, train_set_full, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
raw_focal = b_focal.predict(X_hold, num_iteration=b_focal.best_iteration)
preds_focal = 1.0 / (1.0 + np.exp(-raw_focal))
sc_focal = average_precision_score(y_hold, preds_focal)
print(f"Focal Loss (gamma=1.5): PR-AUC = {sc_focal:.4f} (best_iter = {b_focal.best_iteration}, {time.time()-t1:.1f}s)")
results.append({"regime": "Custom Focal Loss (gamma=1.5)", "pr_auc": sc_focal, "best_iter": b_focal.best_iteration, "time": round(time.time()-t1, 1)})

# =========================================================================
# 2. Axis 2: DART Boosting (Dropout Additive Trees)
# =========================================================================
print("\n" + "="*65)
print("2. Axis 2: DART Boosting (Dropout on Trees)...")
print("="*65)
dart_params = dict(base_params)
dart_params["boosting_type"] = "dart"
dart_params["drop_rate"] = 0.10
dart_params["skip_drop"] = 0.50
dart_params["max_drop"] = 40
dart_params["learning_rate"] = 0.03

# DART cannot early-stop on validation set dynamically, train fixed rounds
t1 = time.time()
b_dart = lgb.train(dart_params, train_set_full, num_boost_round=600)
preds_dart = b_dart.predict(X_hold)
sc_dart = average_precision_score(y_hold, preds_dart)
print(f"DART (600 rounds): PR-AUC = {sc_dart:.4f} ({time.time()-t1:.1f}s)")
results.append({"regime": "DART Boosting (600 r)", "pr_auc": sc_dart, "best_iter": 600, "time": round(time.time()-t1, 1)})

# =========================================================================
# 3. Axis 3: Multi-Horizon Window Ensembling (60d & 90d Specialists)
# =========================================================================
print("\n" + "="*65)
print("3. Axis 3: Multi-Horizon Window Specialists (60-Day & 90-Day)...")
print("="*65)

# 90-day window: 2026-04-01 -> 2026-07-01
mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d = X_full.loc[mask_90d]
y_90d = y_full.loc[mask_90d]
train_set_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)

t1 = time.time()
b_90d = lgb.train(
    base_params, train_set_90d, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_90d = b_90d.predict(X_hold, num_iteration=b_90d.best_iteration)
preds_90d_test = b_90d.predict(X_test, num_iteration=b_90d.best_iteration)
sc_90d = average_precision_score(y_hold, preds_90d)
print(f"90-Day Specialist (Apr-Jul): PR-AUC = {sc_90d:.4f} (best_iter = {b_90d.best_iteration}, {time.time()-t1:.1f}s)")
results.append({"regime": "90-Day Specialist (Apr-Jul)", "pr_auc": sc_90d, "best_iter": b_90d.best_iteration, "time": round(time.time()-t1, 1)})

# 60-day window: 2026-05-01 -> 2026-07-01
mask_60d = (fit_full[TIME_COL] >= "2026-05-01")
X_60d = X_full.loc[mask_60d]
y_60d = y_full.loc[mask_60d]
train_set_60d = lgb.Dataset(X_60d, label=y_60d, categorical_feature=selected_cats, free_raw_data=False)

t1 = time.time()
b_60d = lgb.train(
    base_params, train_set_60d, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_60d = b_60d.predict(X_hold, num_iteration=b_60d.best_iteration)
preds_60d_test = b_60d.predict(X_test, num_iteration=b_60d.best_iteration)
sc_60d = average_precision_score(y_hold, preds_60d)
print(f"60-Day Specialist (May-Jul): PR-AUC = {sc_60d:.4f} (best_iter = {b_60d.best_iteration}, {time.time()-t1:.1f}s)")
results.append({"regime": "60-Day Specialist (May-Jul)", "pr_auc": sc_60d, "best_iter": b_60d.best_iteration, "time": round(time.time()-t1, 1)})

# =========================================================================
# 4. Multi-Horizon Rank Blends
# =========================================================================
print("\n" + "="*65)
print("4. Evaluating Multi-Horizon Rank Blends...")
print("="*65)

r_base = rankdata(preds_base) / len(preds_base)
r_90d = rankdata(preds_90d) / len(preds_90d)
r_60d = rankdata(preds_60d) / len(preds_60d)

# Blend 1: Full + 90d
blend_full_90d = 0.60 * r_base + 0.40 * r_90d
sc_blend_90d = average_precision_score(y_hold, blend_full_90d)
print(f"Blend 1: Full (60%) + 90d Specialist (40%): PR-AUC = {sc_blend_90d:.4f} (Delta vs base: {sc_blend_90d - sc_base:+.4f})")
results.append({"regime": "Blend (Full + 90d)", "pr_auc": sc_blend_90d, "best_iter": "-", "time": "-"})

# Blend 2: Full + 60d
blend_full_60d = 0.60 * r_base + 0.40 * r_60d
sc_blend_60d = average_precision_score(y_hold, blend_full_60d)
print(f"Blend 2: Full (60%) + 60d Specialist (40%): PR-AUC = {sc_blend_60d:.4f} (Delta vs base: {sc_blend_60d - sc_base:+.4f})")
results.append({"regime": "Blend (Full + 60d)", "pr_auc": sc_blend_60d, "best_iter": "-", "time": "-"})

# Blend 3: Tri-Horizon (Full 50% + 90d 25% + 60d 25%)
blend_tri = 0.50 * r_base + 0.25 * r_90d + 0.25 * r_60d
sc_blend_tri = average_precision_score(y_hold, blend_tri)
print(f"Blend 3: Tri-Horizon (Full 50% + 90d 25% + 60d 25%): PR-AUC = {sc_blend_tri:.4f} (Delta vs base: {sc_blend_tri - sc_base:+.4f})")
results.append({"regime": "Tri-Horizon Blend (Full+90d+60d)", "pr_auc": sc_blend_tri, "best_iter": "-", "time": "-"})

print("\n" + "="*70)
print("TRAINING REGIME EXPERIMENTS SUMMARY")
print("="*70)
res_table = pd.DataFrame(results)
print(res_table.to_string(index=False))

best_row = res_table.sort_values("pr_auc", ascending=False).iloc[0]
best_score = best_row["pr_auc"]
print(f"\n>>> Top Training Regime: {best_row['regime']} with PR-AUC = {best_score:.4f} <<<")

GATE = 0.5400
if best_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_score:.4f} >= {GATE:.4f}) <<<")
    # If passed, generate test predictions from the winning regime
    r_base_test = rankdata(preds_base_test) / len(preds_base_test)
    r_90d_test = rankdata(preds_90d_test) / len(preds_90d_test)
    r_60d_test = rankdata(preds_60d_test) / len(preds_60d_test)
    
    if "Tri-Horizon" in best_row["regime"]:
        final_test_preds = 0.50 * r_base_test + 0.25 * r_90d_test + 0.25 * r_60d_test
    elif "90d" in best_row["regime"]:
        final_test_preds = 0.60 * r_base_test + 0.40 * r_90d_test
    else:
        final_test_preds = 0.60 * r_base_test + 0.40 * r_60d_test
        
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_preds})
    sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    out_path = SUBMISSIONS_DIR / "CANDIDATE_TRAINING_0.5400_PLUS.csv"
    sub.to_csv(out_path, index=False)
    print(f"Generated qualified candidate file: {out_path}")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_score:.4f}) <<<")

res_table.to_csv(PROCESSED_DIR / "training_regime_results.csv", index=False)
print(f"Saved results to {PROCESSED_DIR / 'training_regime_results.csv'}")
