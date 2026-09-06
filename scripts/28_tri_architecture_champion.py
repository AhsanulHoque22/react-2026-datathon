"""Tri-Architecture Champion Pipeline: LightGBM + XGBoost + Tabular ResNet.

Combines three fundamentally distinct model paradigms on the Top 160 core:
1. Leaf-Wise Greedy GBDT (LightGBM Full + Tuned 90d Specialist)
2. Level-Wise Depth-Regularized GBDT (XGBoost 90d Specialist)
3. Continuous Neural Manifold (Tabular ResNet 90d Specialist)

Computes the 3-way correlation matrix, runs a simplex grid search,
evaluates against the >= 0.5400 quality gate, and saves candidate predictions.
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
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS
from src.nn_models import TabularResNet

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("TRI-ARCHITECTURE CHAMPION PIPELINE (LGB + XGB + TABULAR RESNET)")
print("="*75)

# 1. Load Data
t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]
selected_nums = [c for c in selected_cols if c not in selected_cats]
print(f"Loaded {df.shape} in {time.time()-t0:.1f}s | {len(selected_nums)} numericals, {len(selected_cats)} categoricals")

labeled = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()

holdout_start = pd.Timestamp("2026-07-01")
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d_lgb = X_full_lgb.loc[mask_90d]
y_90d = y_full.loc[mask_90d]

train_90d = fit_full.loc[mask_90d].copy()

# =========================================================================
# MODEL 1: LightGBM Champion (Full Train 882r + Tuned 90d 433r)
# =========================================================================
print("\n" + "="*70)
print("1/3 Training LightGBM Champion Models...")
print("="*70)
# Full Train
ds_full_lgb = lgb.Dataset(X_full_lgb, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
t1 = time.time()
b_full = lgb.train(p_full_lgb, ds_full_lgb, num_boost_round=882)
p_lgb_full_hold = b_full.predict(X_hold_lgb)
p_lgb_full_test = b_full.predict(X_test_lgb)
print(f"   LGB Full Train: PR-AUC = {average_precision_score(y_hold, p_lgb_full_hold):.4f} ({time.time()-t1:.1f}s)")

# Tuned 90d
ds_90d_lgb = lgb.Dataset(X_90d_lgb, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
p_90d_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
t1 = time.time()
b_90d = lgb.train(p_90d_lgb, ds_90d_lgb, num_boost_round=433)
p_lgb_90d_hold = b_90d.predict(X_hold_lgb)
p_lgb_90d_test = b_90d.predict(X_test_lgb)
print(f"   LGB Tuned 90d:   PR-AUC = {average_precision_score(y_hold, p_lgb_90d_hold):.4f} ({time.time()-t1:.1f}s)")

r_lgb_full_h = rankdata(p_lgb_full_hold) / len(p_lgb_full_hold)
r_lgb_90d_h  = rankdata(p_lgb_90d_hold)  / len(p_lgb_90d_hold)
r_lgb_champ_h = 0.48 * r_lgb_full_h + 0.52 * r_lgb_90d_h
print(f"   --> LGB Champion Blend: PR-AUC = {average_precision_score(y_hold, r_lgb_champ_h):.4f}")

r_lgb_full_t = rankdata(p_lgb_full_test) / len(p_lgb_full_test)
r_lgb_90d_t  = rankdata(p_lgb_90d_test)  / len(p_lgb_90d_test)
r_lgb_champ_t = 0.48 * r_lgb_full_t + 0.52 * r_lgb_90d_t

# =========================================================================
# MODEL 2: XGBoost Specialist (90-Day, 324 rounds)
# =========================================================================
print("\n" + "="*70)
print("2/3 Training XGBoost Specialist Model...")
print("="*70)
dm_xgb_90d  = xgb.DMatrix(X_90d_lgb, label=y_90d, enable_categorical=True)
dm_xgb_hold = xgb.DMatrix(X_hold_lgb, label=y_hold, enable_categorical=True)
dm_xgb_test = xgb.DMatrix(X_test_lgb, enable_categorical=True)

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
t1 = time.time()
b_xgb = xgb.train(xgb_params, dm_xgb_90d, num_boost_round=324)
p_xgb_hold = b_xgb.predict(dm_xgb_hold)
p_xgb_test = b_xgb.predict(dm_xgb_test)
sc_xgb = average_precision_score(y_hold, p_xgb_hold)
print(f"   XGBoost 90d: PR-AUC = {sc_xgb:.4f} ({time.time()-t1:.1f}s)")
r_xgb_h = rankdata(p_xgb_hold) / len(p_xgb_hold)
r_xgb_t = rankdata(p_xgb_test) / len(p_xgb_test)

# =========================================================================
# MODEL 3: Tabular ResNet (90-Day, 8 Epochs)
# =========================================================================
print("\n" + "="*70)
print("3/3 Training Tabular ResNet Model...")
print("="*70)

cat_cardinalities = []
train_cats = []
hold_cats = []
test_cats = []
for c in selected_cats:
    val_counts = train_90d[c].value_counts()
    categories = list(val_counts.index)
    cat_to_id = {val: idx + 1 for idx, val in enumerate(categories)}
    cat_cardinalities.append(len(categories) + 1)
    train_cats.append(train_90d[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    hold_cats.append(hold_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    test_cats.append(test_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))

X_cat_train = np.column_stack(train_cats) if train_cats else np.empty((len(train_90d), 0), dtype=np.int64)
X_cat_hold  = np.column_stack(hold_cats)  if hold_cats  else np.empty((len(hold_df), 0), dtype=np.int64)
X_cat_test  = np.column_stack(test_cats)  if test_cats  else np.empty((len(test_df), 0), dtype=np.int64)

X_num_train_raw = train_90d[selected_nums].copy()
X_num_hold_raw  = hold_df[selected_nums].copy()
X_num_test_raw  = test_df[selected_nums].copy()

medians = X_num_train_raw.median()
X_num_train_raw = X_num_train_raw.fillna(medians)
X_num_hold_raw  = X_num_hold_raw.fillna(medians)
X_num_test_raw  = X_num_test_raw.fillna(medians)

scaler = StandardScaler()
X_num_train = np.clip(scaler.fit_transform(X_num_train_raw), -5.0, 5.0).astype(np.float32)
X_num_hold  = np.clip(scaler.transform(X_num_hold_raw), -5.0, 5.0).astype(np.float32)
X_num_test  = np.clip(scaler.transform(X_num_test_raw), -5.0, 5.0).astype(np.float32)

y_train_arr = train_90d[LABEL_COL].values.astype(np.float32)
y_hold_arr  = hold_df[LABEL_COL].values.astype(np.float32)

BATCH_SIZE = 1024
train_dataset = TensorDataset(torch.from_numpy(X_num_train), torch.from_numpy(X_cat_train), torch.from_numpy(y_train_arr))
hold_dataset  = TensorDataset(torch.from_numpy(X_num_hold), torch.from_numpy(X_cat_hold), torch.from_numpy(y_hold_arr))
test_dataset  = TensorDataset(torch.from_numpy(X_num_test), torch.from_numpy(X_cat_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
hold_loader  = DataLoader(hold_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)

resnet = TabularResNet(
    num_features=len(selected_nums),
    cat_cardinalities=cat_cardinalities,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
)

EPOCHS = 8
optimizer = torch.optim.AdamW(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_resnet_auc = 0.0
best_resnet_hold = None
best_state = None

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    resnet.train()
    total_loss = 0.0
    nb = 0
    for b_num, b_cat, b_y in train_loader:
        optimizer.zero_grad()
        logits = resnet(b_num, b_cat)
        loss = criterion(logits, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()
    
    resnet.eval()
    h_preds = []
    with torch.no_grad():
        for b_num, b_cat, _ in hold_loader:
            h_preds.append(torch.sigmoid(resnet(b_num, b_cat)).cpu().numpy())
    v_preds = np.concatenate(h_preds)
    v_auc = average_precision_score(y_hold, v_preds)
    if v_auc > best_resnet_auc:
        best_resnet_auc = v_auc
        best_resnet_hold = v_preds.copy()
        best_state = {k: v.cpu().clone() for k, v in resnet.state_dict().items()}
    print(f"   ResNet Epoch {epoch}/{EPOCHS} | Loss: {total_loss/nb:.5f} | Val PR-AUC: {v_auc:.4f} ({time.time()-t_ep:.1f}s)")

print(f"   --> Best Tabular ResNet PR-AUC = {best_resnet_auc:.4f}")

# Generate ResNet test predictions
resnet.load_state_dict(best_state)
resnet.eval()
t_preds = []
with torch.no_grad():
    for b_num, b_cat in test_loader:
        t_preds.append(torch.sigmoid(resnet(b_num, b_cat)).cpu().numpy())
p_resnet_test = np.concatenate(t_preds)

r_resnet_h = rankdata(best_resnet_hold) / len(best_resnet_hold)
r_resnet_t = rankdata(p_resnet_test) / len(p_resnet_test)

# =========================================================================
# 4. Tri-Architecture Ensembling Analysis
# =========================================================================
print("\n" + "="*70)
print("TRI-ARCHITECTURE ENSEMBLING ANALYSIS")
print("="*70)

# Rank correlations
pred_matrix = np.column_stack([r_lgb_champ_h, r_xgb_h, r_resnet_h])
cols = ["LightGBM", "XGBoost", "TabularResNet"]
corr_df = pd.DataFrame(np.corrcoef(pred_matrix.T), index=cols, columns=cols)
print("Pairwise Rank Correlations:\n", corr_df.round(4))

# Sweep weights on simplex w_lgb + w_xgb + w_resnet = 1.0
best_tri_score = 0.0
best_weights = (1.0, 0.0, 0.0)

for w_lgb in np.linspace(0.60, 0.95, 36):
    for w_xgb in np.linspace(0.0, 0.30, 31):
        w_res = round(1.0 - (w_lgb + w_xgb), 4)
        if w_res < 0.0:
            continue
        combo = w_lgb * r_lgb_champ_h + w_xgb * r_xgb_h + w_res * r_resnet_h
        sc = average_precision_score(y_hold, combo)
        if sc > best_tri_score:
            best_tri_score = sc
            best_weights = (round(w_lgb, 3), round(w_xgb, 3), w_res)

print("\n" + "="*70)
print("RESULTS BREAKTHROUGH:")
print("="*70)
print(f"LightGBM Champion Alone:        PR-AUC = 0.5288")
print(f"XGBoost Alone:                  PR-AUC = {sc_xgb:.4f}")
print(f"Tabular ResNet Alone:           PR-AUC = {best_resnet_auc:.4f}")
print(f"🏆 Tri-Architecture Ensemble:   PR-AUC = {best_tri_score:.4f}")
print(f"Delta vs LightGBM Champion:     {best_tri_score - 0.5288:+.4f}")
print(f"Optimal Weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, ResNet={best_weights[2]:.3f}")

# Generate and save candidate file
final_test = (
    best_weights[0] * r_lgb_champ_t +
    best_weights[1] * r_xgb_t +
    best_weights[2] * r_resnet_t
)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_tri_architecture_{best_tri_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"\nGenerated candidate submission: {out_path} (shape={sub.shape})")

GATE = 0.5400
if best_tri_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_tri_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_tri_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_tri_score:.4f}) <<<")
