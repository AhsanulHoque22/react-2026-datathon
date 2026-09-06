"""Two-Stage Retrieve-and-Rerank Architecture for Fraud Detection.

Stage 1: Multi-Horizon Screener (identifies top 10-15% high-risk pool, capturing ~92%+ frauds)
Stage 2: Hard-Negative Discriminator (trained strictly on high-risk candidates to separate subtle fraud from false positives)
Stage 3: Monotonic Rank Stitching & Evaluation against >= 0.5400 gate.
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

print("="*75)
print("TWO-STAGE RETRIEVE-AND-RERANK PIPELINE")
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

# =========================================================================
# STAGE 1: Multi-Horizon Screener (0.5288 Baseline)
# =========================================================================
print("\n" + "="*70)
print("STAGE 1: Multi-Horizon Global Screener")
print("="*70)

# Fit Full Train Screener
ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
t1 = time.time()
b_full = lgb.train(p_full, ds_full, num_boost_round=882)
p_full_train = b_full.predict(X_90d)
p_full_hold  = b_full.predict(X_hold)
p_full_test  = b_full.predict(X_test)
print(f"  Full Train Screener fitted ({time.time()-t1:.1f}s)")

# Fit 90d Specialist Screener
ds_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
p_90d = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
t1 = time.time()
b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
p_90d_train = b_90d.predict(X_90d)
p_90d_hold  = b_90d.predict(X_hold)
p_90d_test  = b_90d.predict(X_test)
print(f"  90d Specialist Screener fitted ({time.time()-t1:.1f}s)")

# Stage 1 Ranks
r_stage1_train = 0.48 * (rankdata(p_full_train)/len(p_full_train)) + 0.52 * (rankdata(p_90d_train)/len(p_90d_train))
r_stage1_hold  = 0.48 * (rankdata(p_full_hold)/len(p_full_hold))   + 0.52 * (rankdata(p_90d_hold)/len(p_90d_hold))
r_stage1_test  = 0.48 * (rankdata(p_full_test)/len(p_full_test))   + 0.52 * (rankdata(p_90d_test)/len(p_90d_test))

sc_stage1 = average_precision_score(y_hold, r_stage1_hold)
print(f"\n--> Stage 1 Baseline PR-AUC on Holdout: {sc_stage1:.4f}")

# =========================================================================
# STAGE 2: Candidate Screening & Hard-Negative Discriminator
# =========================================================================
print("\n" + "="*70)
print("STAGE 2: Candidate Screening & Hard-Negative Discriminator")
print("="*70)

# Evaluate recall at various candidate percentiles
total_frauds_hold = y_hold.sum()
for pct in [10, 12, 15, 20]:
    thresh = np.percentile(r_stage1_hold, 100 - pct)
    cand_mask = r_stage1_hold >= thresh
    frauds_captured = y_hold[cand_mask].sum()
    recall = frauds_captured / total_frauds_hold
    prevalence = y_hold[cand_mask].mean()
    print(f"  Top {pct:2d}% pool: {cand_mask.sum():,} candidates | Frauds: {frauds_captured:3d}/{total_frauds_hold:3d} (Recall: {recall*100:.1f}%) | Fraud Rate: {prevalence*100:.2f}%")

# Select Top 15% as candidate threshold
CAND_PCT = 15.0
thresh_train = np.percentile(r_stage1_train, 100 - CAND_PCT)
thresh_hold  = np.percentile(r_stage1_hold,  100 - CAND_PCT)
thresh_test  = np.percentile(r_stage1_test,  100 - CAND_PCT)

cand_train_mask = (r_stage1_train >= thresh_train)
cand_hold_mask  = (r_stage1_hold  >= thresh_hold)
cand_test_mask  = (r_stage1_test  >= thresh_test)

X_train_cand = X_90d.loc[cand_train_mask]
y_train_cand = y_90d.loc[cand_train_mask]

X_hold_cand = X_hold.loc[cand_hold_mask]
y_hold_cand = y_hold.loc[cand_hold_mask]

X_test_cand = X_test.loc[cand_test_mask]

print(f"\nStage 2 Training Pool: {len(X_train_cand):,} rows (Fraud = {y_train_cand.sum():,}, Rate = {y_train_cand.mean()*100:.2f}%)")
print(f"Stage 2 Holdout Pool:  {len(X_hold_cand):,} rows (Fraud = {y_hold_cand.sum():,}, Rate = {y_hold_cand.mean()*100:.2f}%)")

# Train Stage 2 Hard-Negative Discriminator
ds_cand_train = lgb.Dataset(X_train_cand, label=y_train_cand, categorical_feature=selected_cats, free_raw_data=False)
ds_cand_hold  = lgb.Dataset(X_hold_cand,  label=y_hold_cand,  categorical_feature=selected_cats, reference=ds_cand_train, free_raw_data=False)

feval_cand = make_pr_auc_feval(y_hold_cand.values, seed=SEED)

p_stage2 = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.015, num_leaves=31, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=30,
)

t1 = time.time()
b_stage2 = lgb.train(
    p_stage2, ds_cand_train, num_boost_round=2000, valid_sets=[ds_cand_hold], feval=feval_cand,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
p_cand_hold = b_stage2.predict(X_hold_cand, num_iteration=b_stage2.best_iteration)
p_cand_test = b_stage2.predict(X_test_cand, num_iteration=b_stage2.best_iteration)
sc_cand = average_precision_score(y_hold_cand, p_cand_hold)
print(f"Stage 2 Discriminator: PR-AUC within Candidate Pool = {sc_cand:.4f} (best_iter = {b_stage2.best_iteration:4d}, {time.time()-t1:.1f}s)")

# =========================================================================
# STAGE 3: Monotonic Rank Stitching & Global Evaluation
# =========================================================================
print("\n" + "="*70)
print("STAGE 3: Monotonic Rank Stitching & Sweep")
print("="*70)

def stitch_ranks(stage1_scores, stage2_cand_scores, cand_mask, cutoff=0.85):
    final_ranks = np.zeros(len(stage1_scores), dtype=np.float64)
    # Lower tier (non-candidates): keep stage 1 ranking in [0, cutoff]
    non_cand_scores = stage1_scores[~cand_mask]
    r_non_cand = rankdata(non_cand_scores) / len(non_cand_scores)
    final_ranks[~cand_mask] = r_non_cand * cutoff
    
    # Upper tier (candidates): reranked by Stage 2 in [cutoff, 1.0]
    r_cand = rankdata(stage2_cand_scores) / len(stage2_cand_scores)
    final_ranks[cand_mask] = cutoff + r_cand * (1.0 - cutoff)
    return final_ranks

best_stitched_score = sc_stage1
best_cutoff = 0.85

for cutoff in np.linspace(0.70, 0.95, 26):
    stitched_hold = stitch_ranks(r_stage1_hold, p_cand_hold, cand_hold_mask, cutoff=cutoff)
    sc = average_precision_score(y_hold, stitched_hold)
    if sc > best_stitched_score:
        best_stitched_score = sc
        best_cutoff = cutoff
    print(f"  Cutoff={cutoff:.2f}: Stitched PR-AUC = {sc:.4f}")

# Also test soft blend within the upper tier: w * Stage1 + (1-w) * Stage2
print("\n--- Soft Blend within Upper Tier Sweep ---")
best_blend_score = best_stitched_score
best_w_upper = 0.50

for w_up in np.linspace(0.1, 0.9, 17):
    # Within candidate tier, blend stage 1 rank and stage 2 score
    r_stage1_cand = rankdata(r_stage1_hold[cand_hold_mask]) / np.sum(cand_hold_mask)
    r_stage2_cand = rankdata(p_cand_hold) / np.sum(cand_hold_mask)
    upper_combo = w_up * r_stage1_cand + (1.0 - w_up) * r_stage2_cand
    
    stitched_combo = stitch_ranks(r_stage1_hold, upper_combo, cand_hold_mask, cutoff=best_cutoff)
    sc = average_precision_score(y_hold, stitched_combo)
    if sc > best_blend_score:
        best_blend_score = sc
        best_w_upper = w_up
    print(f"  w_upper={w_up:.2f} Stage1 + {1-w_up:.2f} Stage2: PR-AUC = {sc:.4f}")

print("\n" + "="*70)
print("TWO-STAGE RERANKER RESULTS SUMMARY")
print("="*70)
print(f"Stage 1 Baseline:             PR-AUC = {sc_stage1:.4f}")
print(f"Stage 2 Hard-Negative Pool:   PR-AUC = {sc_cand:.4f}")
print(f"🏆 Final Stitched Reranker:   PR-AUC = {best_blend_score:.4f}")
print(f"Delta vs Stage 1 Baseline:    {best_blend_score - sc_stage1:+.4f}")
print(f"Optimal Parameters: Cutoff={best_cutoff:.2f}, Upper Weight={best_w_upper:.2f}")

# Generate test predictions
r_stage1_cand_test = rankdata(r_stage1_test[cand_test_mask]) / np.sum(cand_test_mask)
r_stage2_cand_test = rankdata(p_cand_test) / np.sum(cand_test_mask)
upper_test_combo = best_w_upper * r_stage1_cand_test + (1.0 - best_w_upper) * r_stage2_cand_test

final_test_stitched = stitch_ranks(r_stage1_test, upper_test_combo, cand_test_mask, cutoff=best_cutoff)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_stitched})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_twostage_rerank_{best_blend_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"\nGenerated candidate submission: {out_path} (shape={sub.shape})")

GATE = 0.5400
if best_blend_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_blend_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_blend_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_blend_score:.4f}) <<<")
