"""Test location surge dynamics and customer-location familiarity features.

Engineers 9 targeted non-leaking features:
1. loc_surge_accel: (loc_cnt_1h * 24) / (loc_cnt_24h + 1)
2. loc_amt_surge_accel: (loc_amtsum_1h * 24) / (loc_amtsum_24h + 10)
3. loc_avg_amt_1h: loc_amtsum_1h / (loc_cnt_1h + EPS)
4. loc_avg_amt_24h: loc_amtsum_24h / (loc_cnt_24h + EPS)
5. loc_amt_collapse_ratio: loc_avg_amt_1h / (loc_avg_amt_24h + EPS) (captures ticket-size halving)
6. cust_loc_prior_visits: strictly prior count of (customer, location) transactions
7. cust_merch_prior_visits: strictly prior count of (customer, merchant) transactions
8. dev_loc_prior_uses: strictly prior count of (device, location) uses
9. cust_loc_visit_share: cust_loc_prior_visits / (cust_history_count + 1)

Evaluates on the Jul 1-15 held-out probe against the 0.5288 benchmark.
"""
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

print("Loading features.pkl...")
t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]
print(f"Loaded {df.shape} in {time.time()-t0:.1f}s")

# Compute the 9 candidate features strictly chronologically
print("Engineering 9 surge & familiarity features...")
t1 = time.time()
EPS = 1e-6

# 1. Location surges
df["loc_surge_accel"] = (df["loc_cnt_1h"] * 24.0) / (df["loc_cnt_24h"] + 1.0)
df["loc_amt_surge_accel"] = (df["loc_amtsum_1h"] * 24.0) / (df["loc_amtsum_24h"] + 10.0)
df["loc_avg_amt_1h"] = df["loc_amtsum_1h"] / (df["loc_cnt_1h"] + EPS)
df["loc_avg_amt_24h"] = df["loc_amtsum_24h"] / (df["loc_cnt_24h"] + EPS)
df["loc_amt_collapse_ratio"] = (df["loc_avg_amt_1h"] + EPS) / (df["loc_avg_amt_24h"] + EPS)

# 2. Expanding pair counts (strictly prior using cumcount)
cust_loc = df["customer_id"].astype(str) + "_" + df["location"].astype(str)
df["cust_loc_prior_visits"] = cust_loc.groupby(cust_loc).cumcount().astype("float32")

cust_merch = df["customer_id"].astype(str) + "_" + df["merchant_id"].astype(str)
df["cust_merch_prior_visits"] = cust_merch.groupby(cust_merch).cumcount().astype("float32")

dev_loc = df["device_id"].astype(str) + "_" + df["location"].astype(str)
df["dev_loc_prior_uses"] = dev_loc.groupby(dev_loc).cumcount().astype("float32")

if "cust_history_count" in df.columns:
    df["cust_loc_visit_share"] = df["cust_loc_prior_visits"] / (df["cust_history_count"] + 1.0)
else:
    df["cust_loc_visit_share"] = 0.0

new_cols = [
    "loc_surge_accel", "loc_amt_surge_accel", "loc_avg_amt_1h", "loc_avg_amt_24h",
    "loc_amt_collapse_ratio", "cust_loc_prior_visits", "cust_merch_prior_visits",
    "dev_loc_prior_uses", "cust_loc_visit_share"
]
print(f"Engineered {len(new_cols)} features in {time.time()-t1:.1f}s")

# Prepare train / holdout splits
cols_with_new = selected_cols + new_cols
print(f"Total features to evaluate: {len(cols_with_new)} (160 baseline + 9 new)")

labeled = df[~df["is_test"]].copy()
holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

X_full = prepare_lgb_frame(fit_full, cols_with_new, selected_cats)
X_hold = prepare_lgb_frame(hold_df, cols_with_new, selected_cats)

# Step 1: Train Tuned 90d Specialist with new features
print("\n1. Training Tuned 90d Specialist with new features...")
mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d = X_full.loc[mask_90d]
y_90d = y_full.loc[mask_90d]

ds_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
ds_hold_90d = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_90d, free_raw_data=False)
feval = make_pr_auc_feval(y_hold.values, seed=SEED)

p_90d = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
t_fit = time.time()
b_90d = lgb.train(
    p_90d, ds_90d, num_boost_round=3000, valid_sets=[ds_hold_90d], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_90d = b_90d.predict(X_hold, num_iteration=b_90d.best_iteration)
sc_90d = average_precision_score(y_hold, preds_90d)
print(f"90d Specialist with new features: PR-AUC = {sc_90d:.4f} (best_iter = {b_90d.best_iteration:4d}, {time.time()-t_fit:.1f}s)")
print(f"Delta vs baseline 90d (0.5272): {sc_90d - 0.527205:+.4f}")

# Check feature importance of new features in 90d model
gain = b_90d.feature_importance(importance_type="gain")
fi = pd.DataFrame({"feature": cols_with_new, "gain": gain}).sort_values("gain", ascending=False)
print("\nTop 20 Features in 90d Specialist:")
print(fi.head(20).to_string(index=False))

print("\nNew Features Importance in 90d Specialist:")
print(fi[fi["feature"].isin(new_cols)].to_string(index=False))

# Step 2: Train Full Train with new features
print("\n2. Training Full Train with new features...")
ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
ds_hold_full = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=ds_full, free_raw_data=False)

p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
t_fit = time.time()
b_full = lgb.train(
    p_full, ds_full, num_boost_round=3000, valid_sets=[ds_hold_full], feval=feval,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_full = b_full.predict(X_hold, num_iteration=b_full.best_iteration)
sc_full = average_precision_score(y_hold, preds_full)
print(f"Full Train with new features: PR-AUC = {sc_full:.4f} (best_iter = {b_full.best_iteration:4d}, {time.time()-t_fit:.1f}s)")
print(f"Delta vs baseline Full (0.5269): {sc_full - 0.526887:+.4f}")

# Step 3: Evaluate Blend
r_full = rankdata(preds_full) / len(preds_full)
r_90d = rankdata(preds_90d) / len(preds_90d)
blend = 0.48 * r_full + 0.52 * r_90d
sc_blend = average_precision_score(y_hold, blend)
print("\n" + "="*70)
print(f"NEW BLEND PR-AUC = {sc_blend:.4f}")
print(f"Delta vs previous champion (0.5288): {sc_blend - 0.5288:+.4f}")
print("="*70)
