"""Generate the 0.5298 All-Time Champion Candidate & Export Teammate ResNet Predictions.

Combines:
- 88% LightGBM Champion (48% Full Train 882r + 52% Tuned 90d 433r)
- 12% Tabular ResNet Specialist (Seed 42, 12 epochs, r=0.4331 diversity)

Outputs:
1. submissions/CANDIDATE_tree_resnet_0.5298.csv (Full submission file)
2. submissions/resnet_predictions_holdout_jul1_15.csv (Teammate validation predictions)
3. submissions/resnet_predictions_test.csv (Teammate test predictions)
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
print("GENERATING 0.5298 ALL-TIME CHAMPION & TEAMMATE RESNET EXPORTS")
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
fit_full = labeled.loc[labeled[TIME_COL] < holdout_start].copy()
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start].copy()

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
train_90d = fit_full.loc[mask_90d].copy()

y_full = fit_full[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)
y_90d = train_90d[LABEL_COL].astype(int)

# -------------------------------------------------------------------------
# Step 1: LightGBM Champion (Full Train 882r + Tuned 90d 433r)
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("1. Training LightGBM Champion Models...")
print("="*70)

X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)
X_90d_lgb  = X_full_lgb.loc[mask_90d]

# Full Train LGB
ds_full_lgb = lgb.Dataset(X_full_lgb, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
t1 = time.time()
b_full = lgb.train(p_full_lgb, ds_full_lgb, num_boost_round=882)
lgb_full_hold = b_full.predict(X_hold_lgb)
lgb_full_test = b_full.predict(X_test_lgb)
print(f"   LGB Full Train: PR-AUC = {average_precision_score(y_hold, lgb_full_hold):.4f} ({time.time()-t1:.1f}s)")

# Tuned 90d LGB
ds_90d_lgb = lgb.Dataset(X_90d_lgb, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
p_90d_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
t1 = time.time()
b_90d = lgb.train(p_90d_lgb, ds_90d_lgb, num_boost_round=433)
lgb_90d_hold = b_90d.predict(X_hold_lgb)
lgb_90d_test = b_90d.predict(X_test_lgb)
print(f"   LGB Tuned 90d:   PR-AUC = {average_precision_score(y_hold, lgb_90d_hold):.4f} ({time.time()-t1:.1f}s)")

r_lgb_full_h = rankdata(lgb_full_hold) / len(lgb_full_hold)
r_lgb_90d_h  = rankdata(lgb_90d_hold)  / len(lgb_90d_hold)
r_lgb_champ_h = 0.48 * r_lgb_full_h + 0.52 * r_lgb_90d_h
lgb_champ_score = average_precision_score(y_hold, r_lgb_champ_h)
print(f"   --> LightGBM Champion Benchmark: PR-AUC = {lgb_champ_score:.4f}")

r_lgb_full_t = rankdata(lgb_full_test) / len(lgb_full_test)
r_lgb_90d_t  = rankdata(lgb_90d_test)  / len(lgb_90d_test)
r_lgb_champ_t = 0.48 * r_lgb_full_t + 0.52 * r_lgb_90d_t

# -------------------------------------------------------------------------
# Step 2: Tabular ResNet (Seed 42, 12 epochs)
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("2. Training Tabular ResNet (Seed 42, 12 epochs)...")
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

model = TabularResNet(
    num_features=len(selected_nums),
    cat_cardinalities=cat_cardinalities,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
)

EPOCHS = 12
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

best_val_prauc = 0.0
best_model_state = None
best_hold_preds = None

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    total_loss = 0.0
    nb = 0
    for b_num, b_cat, b_y in train_loader:
        optimizer.zero_grad()
        logits = model(b_num, b_cat)
        loss = criterion(logits, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()
    
    # Eval
    model.eval()
    hold_preds_list = []
    with torch.no_grad():
        for b_num, b_cat, _ in hold_loader:
            hold_preds_list.append(torch.sigmoid(model(b_num, b_cat)).cpu().numpy())
    val_preds = np.concatenate(hold_preds_list)
    val_prauc = average_precision_score(y_hold_arr, val_preds)
    
    is_best = val_prauc > best_val_prauc
    if is_best:
        best_val_prauc = val_prauc
        best_hold_preds = val_preds.copy()
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    star = " *** BEST ***" if is_best else ""
    print(f"   Epoch {epoch:2d}/{EPOCHS:2d} | Loss: {total_loss/nb:.5f} | Val PR-AUC: {val_prauc:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} ({time.time()-t_ep:.1f}s){star}")

print(f"   >>> Tabular ResNet Best Standalone PR-AUC = {best_val_prauc:.4f} <<<")

# Generate ResNet test predictions using the best checkpoint
model.load_state_dict(best_model_state)
model.eval()
test_preds_list = []
with torch.no_grad():
    for b_num, b_cat in test_loader:
        test_preds_list.append(torch.sigmoid(model(b_num, b_cat)).cpu().numpy())
resnet_test_preds = np.concatenate(test_preds_list)

r_resnet_h = rankdata(best_hold_preds) / len(best_hold_preds)
r_resnet_t = rankdata(resnet_test_preds) / len(resnet_test_preds)

corr = np.corrcoef(r_lgb_champ_h, r_resnet_h)[0, 1]
print(f"   Rank Correlation r(Tabular ResNet, LightGBM Champion): {corr:.4f}")

# -------------------------------------------------------------------------
# Step 3: Optimal Rank Ensembling & Verification
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("3. Evaluating 0.88 / 0.12 Blend...")
print("="*70)

final_hold_blend = 0.88 * r_lgb_champ_h + 0.12 * r_resnet_h
final_score = average_precision_score(y_hold_arr, final_hold_blend)
print(f"   🏆 Final Blend (88% LGBM + 12% ResNet) PR-AUC = {final_score:.4f}")
print(f"   Lift over LightGBM Champion: {final_score - lgb_champ_score:+.4f}")

# Sweep around 0.88 to display the entire plateau
print("\n--- Weight Plateau Verification ---")
for w in [0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
    sc = average_precision_score(y_hold_arr, w * r_lgb_champ_h + (1.0 - w) * r_resnet_h)
    print(f"   w={w:.2f} LGB + {1-w:.2f} ResNet: PR-AUC = {sc:.4f}")

# -------------------------------------------------------------------------
# Step 4: Export Deliverables
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("4. Saving Deliverables...")
print("="*70)

# 1. Candidate Submission File
final_test_blend = 0.88 * r_lgb_champ_t + 0.12 * r_resnet_t
sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test_blend})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

cand_file = SUBMISSIONS_DIR / f"CANDIDATE_tree_resnet_{final_score:.4f}.csv"
sub.to_csv(cand_file, index=False)
print(f"   ✅ Saved Champion Candidate: {cand_file} (shape={sub.shape})")

# 2. Teammate Holdout ResNet File
df_holdout_export = pd.DataFrame({
    "transaction_id": hold_df["transaction_id"].values,
    "timestamp": hold_df[TIME_COL].values,
    "actual_fraud": y_hold_arr.astype(int),
    "resnet_prob": best_hold_preds,
    "resnet_rank": r_resnet_h,
})
holdout_file = SUBMISSIONS_DIR / "resnet_predictions_holdout_jul1_15.csv"
df_holdout_export.to_csv(holdout_file, index=False)
print(f"   ✅ Saved Teammate Holdout Predictions: {holdout_file} (shape={df_holdout_export.shape})")

# 3. Teammate Test ResNet File
df_test_export = pd.DataFrame({
    "transaction_id": test_df["transaction_id"].values,
    "resnet_prob": resnet_test_preds,
    "resnet_rank": r_resnet_t,
})
df_test_export = df_test_export.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
test_file = SUBMISSIONS_DIR / "resnet_predictions_test.csv"
df_test_export.to_csv(test_file, index=False)
print(f"   ✅ Saved Teammate Test Predictions: {test_file} (shape={df_test_export.shape})")

print("\n" + "="*70)
print("ALL DELIVERABLES GENERATED SUCCESSFULLY")
print("="*70)
