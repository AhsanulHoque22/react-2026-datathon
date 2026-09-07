"""Screening and Benchmarking New Breakthrough Feature Candidates.

Tests 5 candidate feature families:
1. Ping-and-Cashout Amount Stepup: cust_amt_stepup_ratio, cust_ping_cashout_5m
2. Impossible Spatial Travel: cust_impossible_travel_30m, dev_impossible_travel_1h
3. Merchant Category Relative Outlier: cat_amt_zscore
4. 2nd-Order Inter-Arrival Acceleration: cust_gap_accel_2nd
5. Amount Volatility Ratio: cust_amt_to_recent_mean_ratio

Measures:
- Univariate PR-AUC on Jul 1-15 held-out tail
- Feature correlation with existing Top 160 core
- Impact on LightGBM Champion (baseline: 0.5288)
- Impact on Tabular ResNet (baseline: 0.5214)
- Impact on Tree-Neural Ensemble (baseline: 0.5298)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, CAT_COLS
from src.nn_models import TabularResNet

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("FEATURE ENGINEERING SCREENING & BENCHMARK PIPELINE")
print("="*75)

# 1. Load Data
t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]
selected_nums = [c for c in selected_cols if c not in selected_cats]
print(f"Loaded {df.shape} in {time.time()-t0:.1f}s | Baseline: {len(selected_cols)} features ({len(selected_nums)} num, {len(selected_cats)} cat)")

# Ensure correct sorting for sequential operations
df = df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)

# -------------------------------------------------------------------------
# 2. Engineer Candidate Features (Strictly Causal)
# -------------------------------------------------------------------------
print("\nEngineering candidate feature families...")
new_feature_names = []

# 1. Ping-and-Cashout Amount Stepup
prev_amt = df.groupby("customer_id")["amount_bdt"].shift(1)
df["cust_amt_stepup_ratio"] = (df["amount_bdt"] / (prev_amt + 1.0)).fillna(1.0).astype(np.float32)
df["cust_ping_cashout_5m"] = ((df["cust_amt_stepup_ratio"] > 5.0) & (df["cust_seconds_since_last"] < 300)).astype(np.float32)
new_feature_names.extend(["cust_amt_stepup_ratio", "cust_ping_cashout_5m"])

# 2. Impossible Travel (Location Hops under 30m / 1h)
prev_loc = df.groupby("customer_id")["location"].shift(1)
loc_changed = ((df["location"] != prev_loc) & prev_loc.notna())
df["cust_impossible_travel_30m"] = (loc_changed & (df["cust_seconds_since_last"] < 1800)).astype(np.float32)
new_feature_names.append("cust_impossible_travel_30m")

prev_dev_loc = df.groupby("device_id")["location"].shift(1)
dev_loc_changed = ((df["location"] != prev_dev_loc) & prev_dev_loc.notna())
df["dev_impossible_travel_1h"] = (dev_loc_changed & (df["dev_seconds_since_last"] < 3600)).astype(np.float32)
new_feature_names.append("dev_impossible_travel_1h")

# 3. Expanding Category-Normalized Amount Z-score
cat_grp = df.groupby("merchant_category")["amount_bdt"]
cat_cum_sum = cat_grp.cumsum()
cat_cum_cnt = cat_grp.cumcount().astype(np.float64)
cat_prior_sum = cat_cum_sum - df["amount_bdt"]
cat_prior_mean = cat_prior_sum / cat_cum_cnt.replace(0, np.nan)

cat_cum_sq = (df["amount_bdt"]**2).groupby(df["merchant_category"]).cumsum()
cat_prior_sq = cat_cum_sq - (df["amount_bdt"]**2)
cat_prior_var = (cat_prior_sq / cat_cum_cnt.replace(0, np.nan)) - cat_prior_mean**2
cat_prior_std = np.sqrt(np.maximum(cat_prior_var.fillna(1.0), 1.0))
df["cat_amt_zscore"] = ((df["amount_bdt"] - cat_prior_mean) / (cat_prior_std + 1.0)).fillna(0.0).astype(np.float32)
new_feature_names.append("cat_amt_zscore")

# 4. 2nd-Order Inter-Arrival Gap Acceleration
prev_gap = df.groupby("customer_id")["cust_seconds_since_last"].shift(1)
df["cust_gap_accel_2nd"] = (prev_gap / (df["cust_seconds_since_last"] + 1.0)).fillna(1.0).astype(np.float32)
new_feature_names.append("cust_gap_accel_2nd")

print(f"Generated {len(new_feature_names)} candidates: {new_feature_names}")

# -------------------------------------------------------------------------
# 3. Univariate Screening on Jul 1-15 Held-Out Tail
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("UNIVARIATE SCREENING ON JUL 1-15 HELD-OUT TAIL (Base Fraud Rate: 1.56%)")
print("="*70)

labeled = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()

holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start].copy()
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start].copy()

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

for feat in new_feature_names:
    vals = hold_df[feat].fillna(0).values
    ap = average_precision_score(y_hold, vals)
    mean_val_fraud = vals[y_hold == 1].mean()
    mean_val_clean = vals[y_hold == 0].mean()
    ratio = (mean_val_fraud / (mean_val_clean + 1e-6)) if mean_val_clean > 0 else 0
    print(f"  {feat:30s} | Univariate AP: {ap:.4f} | Fraud vs Clean Ratio: {ratio:.2f}x")

# -------------------------------------------------------------------------
# 4. Stepwise LightGBM Champion Benchmark
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("BENCHMARKING: ADDING CANDIDATES TO LIGHTGBM CHAMPION")
print("="*70)

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")

def evaluate_lgb_champion(feature_cols):
    """Fits Full Train + Tuned 90d and computes blend PR-AUC."""
    cats = [c for c in CAT_COLS if c in feature_cols]
    
    X_full = prepare_lgb_frame(fit_full, feature_cols, cats)
    X_hold = prepare_lgb_frame(hold_df, feature_cols, cats)
    X_90d  = X_full.loc[mask_90d]
    y_90d  = y_full.loc[mask_90d]
    
    # Full Train
    ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=cats, free_raw_data=False)
    p_full = dict(
        objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
        feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )
    b_full = lgb.train(p_full, ds_full, num_boost_round=882)
    p_full_hold = b_full.predict(X_hold)
    
    # 90d
    ds_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=cats, free_raw_data=False)
    p_90d = dict(
        objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
        feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
    )
    b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
    p_90d_hold = b_90d.predict(X_hold)
    
    r_full = rankdata(p_full_hold) / len(p_full_hold)
    r_90d  = rankdata(p_90d_hold)  / len(p_90d_hold)
    blend  = 0.48 * r_full + 0.52 * r_90d
    score  = average_precision_score(y_hold, blend)
    return score, p_full_hold, p_90d_hold

baseline_score, _, _ = evaluate_lgb_champion(selected_cols)
print(f"Baseline LightGBM Champion (Top 160 core): PR-AUC = {baseline_score:.4f}")

# Test adding each candidate feature individually
surviving_features = []
for cand in new_feature_names:
    t_c = time.time()
    test_cols = selected_cols + [cand]
    score, _, _ = evaluate_lgb_champion(test_cols)
    delta = score - baseline_score
    status = "WINNER" if delta > 0.0001 else ("NEUTRAL" if abs(delta) <= 0.0001 else "DILUTION")
    print(f"  + {cand:28s} -> PR-AUC = {score:.4f} ({delta:+.4f}) [{status}] in {time.time()-t_c:.1f}s")
    if delta > 0:
        surviving_features.append(cand)

# Test combined surviving features
if surviving_features:
    print(f"\nTesting combined winners: {surviving_features}")
    combined_cols = selected_cols + surviving_features
    comb_score, p_full_hold_comb, p_90d_hold_comb = evaluate_lgb_champion(combined_cols)
    print(f"  Combined Winners PR-AUC = {comb_score:.4f} (delta vs baseline: {comb_score - baseline_score:+.4f})")
else:
    print("\nNo single feature produced a positive delta alone; testing the cohesive ratio bundle...")
    bundle = ["cust_amt_stepup_ratio", "cat_amt_zscore", "cust_gap_accel_2nd"]
    bundle_cols = selected_cols + bundle
    comb_score, p_full_hold_comb, p_90d_hold_comb = evaluate_lgb_champion(bundle_cols)
    print(f"  Cohesive Ratio Bundle PR-AUC = {comb_score:.4f} (delta vs baseline: {comb_score - baseline_score:+.4f})")

# -------------------------------------------------------------------------
# 5. Tabular ResNet Evaluation with New Features
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("BENCHMARKING: IMPACT ON TABULAR RESNET (Neural Manifold)")
print("="*70)

# ResNet with new features
train_90d = fit_full.loc[mask_90d].copy()
y_train_90d = train_90d[LABEL_COL].values.astype(np.float32)
y_hold_arr = hold_df[LABEL_COL].values.astype(np.float32)

resnet_cols = selected_cols + ["cust_amt_stepup_ratio", "cat_amt_zscore", "cust_gap_accel_2nd"]
resnet_cats = [c for c in CAT_COLS if c in resnet_cols]
resnet_nums = [c for c in resnet_cols if c not in resnet_cats]

# Categoricals
cat_cards = []
tr_cats, h_cats = [], []
for c in resnet_cats:
    vc = train_90d[c].value_counts()
    cats = list(vc.index)
    c2i = {v: i + 1 for i, v in enumerate(cats)}
    cat_cards.append(len(cats) + 1)
    tr_cats.append(train_90d[c].map(c2i).fillna(0).values.astype(np.int64))
    h_cats.append(hold_df[c].map(c2i).fillna(0).values.astype(np.int64))

X_c_tr = np.column_stack(tr_cats) if tr_cats else np.empty((len(train_90d), 0), dtype=np.int64)
X_c_h  = np.column_stack(h_cats)  if h_cats  else np.empty((len(hold_df), 0), dtype=np.int64)

# Numericals
X_n_tr_raw = train_90d[resnet_nums].fillna(train_90d[resnet_nums].median())
X_n_h_raw  = hold_df[resnet_nums].fillna(train_90d[resnet_nums].median())

scaler = StandardScaler()
X_n_tr = np.clip(scaler.fit_transform(X_n_tr_raw), -5.0, 5.0).astype(np.float32)
X_n_h  = np.clip(scaler.transform(X_n_h_raw), -5.0, 5.0).astype(np.float32)

BATCH_SIZE = 1024
tr_ds = TensorDataset(torch.from_numpy(X_n_tr), torch.from_numpy(X_c_tr), torch.from_numpy(y_train_90d))
h_ds  = TensorDataset(torch.from_numpy(X_n_h), torch.from_numpy(X_c_h), torch.from_numpy(y_hold_arr))

tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
h_ld  = DataLoader(h_ds,  batch_size=BATCH_SIZE*2, shuffle=False)

resnet_enhanced = TabularResNet(
    num_features=len(resnet_nums),
    cat_cardinalities=cat_cards,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
)
opt = torch.optim.AdamW(resnet_enhanced.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8, eta_min=1e-5)
crit = nn.BCEWithLogitsLoss()

best_res_auc = 0.0
best_res_preds = None

for ep in range(1, 9):
    t_ep = time.time()
    resnet_enhanced.train()
    for bn, bc, by in tr_ld:
        opt.zero_grad()
        loss = crit(resnet_enhanced(bn, bc), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet_enhanced.parameters(), 1.0)
        opt.step()
    sched.step()
    
    resnet_enhanced.eval()
    hp = []
    with torch.no_grad():
        for bn, bc, _ in h_ld:
            hp.append(torch.sigmoid(resnet_enhanced(bn, bc)).cpu().numpy())
    vp = np.concatenate(hp)
    va = average_precision_score(y_hold_arr, vp)
    if va > best_res_auc:
        best_res_auc = va
        best_res_preds = vp.copy()
    print(f"  ResNet Epoch {ep}/8 | Val PR-AUC = {va:.4f} ({time.time()-t_ep:.1f}s)")

print(f"\n>>> ResNet with Augmented Features: Standalone PR-AUC = {best_res_auc:.4f} (Baseline: 0.5214) <<<")

# Final Blend Test
r_lgb_champ = 0.48 * (rankdata(p_full_hold_comb)/len(p_full_hold_comb)) + 0.52 * (rankdata(p_90d_hold_comb)/len(p_90d_hold_comb))
r_res_enh   = rankdata(best_res_preds) / len(best_res_preds)

print("\n--- Tree-Neural Ensemble Blend with Augmented Features ---")
for w in [0.84, 0.86, 0.88, 0.90, 0.92]:
    b = w * r_lgb_champ + (1.0 - w) * r_res_enh
    print(f"  w={w:.2f} LGB + {1-w:.2f} Enhanced ResNet: PR-AUC = {average_precision_score(y_hold_arr, b):.4f}")

