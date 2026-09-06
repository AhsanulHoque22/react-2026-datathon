"""Train and evaluate Tabular ResNet on 90-day window and blend with LightGBM champion."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import CAT_COLS
from src.nn_models import TabularResNet

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("TABULAR RESNET EXPERIMENT (90-DAY HORIZON)")
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

# 90-day specialist slice
mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
train_90d = fit_full.loc[mask_90d].copy()

y_train = train_90d[LABEL_COL].values.astype(np.float32)
y_hold = hold_df[LABEL_COL].values.astype(np.float32)

print(f"90d Train: {len(train_90d):,} rows (fraud={int(y_train.sum()):,}) | Holdout: {len(hold_df):,} rows (fraud={int(y_hold.sum()):,})")

# 2. Preprocess Categoricals (Integer Encoding for Embedding)
cat_cardinalities = []
train_cats = []
hold_cats = []
test_cats = []

for c in selected_cats:
    val_counts = train_90d[c].value_counts()
    categories = list(val_counts.index)
    cat_to_id = {val: idx + 1 for idx, val in enumerate(categories)}  # 1-indexed, 0 for unknown
    cat_cardinalities.append(len(categories) + 1)
    
    train_cats.append(train_90d[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    hold_cats.append(hold_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    test_cats.append(test_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))

X_cat_train = np.column_stack(train_cats) if train_cats else np.empty((len(train_90d), 0), dtype=np.int64)
X_cat_hold  = np.column_stack(hold_cats)  if hold_cats  else np.empty((len(hold_df), 0), dtype=np.int64)
X_cat_test  = np.column_stack(test_cats)  if test_cats  else np.empty((len(test_df), 0), dtype=np.int64)

# 3. Preprocess Numericals (Median Impute + Scaler + Clip)
print("Preprocessing numerical features...")
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

print(f"Feature Matrices: Num={X_num_train.shape}, Cat={X_cat_train.shape}")

# 4. PyTorch Datasets & DataLoaders
BATCH_SIZE = 1024
train_dataset = TensorDataset(torch.from_numpy(X_num_train), torch.from_numpy(X_cat_train), torch.from_numpy(y_train))
hold_dataset  = TensorDataset(torch.from_numpy(X_num_hold), torch.from_numpy(X_cat_hold), torch.from_numpy(y_hold))
test_dataset  = TensorDataset(torch.from_numpy(X_num_test), torch.from_numpy(X_cat_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
hold_loader  = DataLoader(hold_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)

# 5. Initialize Model, Optimizer, Loss
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

print(f"\nModel Architecture: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Training for {EPOCHS} epochs with batch_size={BATCH_SIZE}...")

best_val_prauc = 0.0
best_model_state = None
best_hold_preds = None

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for b_num, b_cat, b_y in train_loader:
        optimizer.zero_grad()
        logits = model(b_num, b_cat)
        loss = criterion(logits, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    scheduler.step()
    avg_loss = total_loss / num_batches
    
    # Evaluation on Holdout Tail
    model.eval()
    hold_preds_list = []
    with torch.no_grad():
        for b_num, b_cat, _ in hold_loader:
            logits = model(b_num, b_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            hold_preds_list.append(probs)
            
    val_preds = np.concatenate(hold_preds_list)
    val_prauc = average_precision_score(y_hold, val_preds)
    dt = time.time() - t_ep
    
    is_best = val_prauc > best_val_prauc
    if is_best:
        best_val_prauc = val_prauc
        best_hold_preds = val_preds.copy()
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    star = " *** BEST ***" if is_best else ""
    print(f"Epoch {epoch:2d}/{EPOCHS:2d} | Train Loss: {avg_loss:.5f} | Val PR-AUC: {val_prauc:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s{star}")

print(f"\n>>> Best Tabular ResNet Validation PR-AUC = {best_val_prauc:.4f} <<<")

# Restore best weights and generate test predictions
model.load_state_dict(best_model_state)
model.eval()
test_preds_list = []
with torch.no_grad():
    for b_num, b_cat in test_loader:
        logits = model(b_num, b_cat)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds_list.append(probs)
resnet_test_preds = np.concatenate(test_preds_list)

# -------------------------------------------------------------------------
# Ensembling Analysis with LightGBM Champion
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("ENSEMBLING: TABULAR RESNET + LIGHTGBM CHAMPION")
print("="*70)

# Load LightGBM 0.5288 holdout predictions from scripts/24
import lightgbm as lgb
from src.model import prepare_lgb_frame

X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)

ds_full_lgb = lgb.Dataset(X_full_lgb, label=fit_full[LABEL_COL].astype(int), categorical_feature=selected_cats, free_raw_data=False)
p_full_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
print("Fitting LightGBM Full Train (882 rounds)...")
b_full = lgb.train(p_full_lgb, ds_full_lgb, num_boost_round=882)
lgb_full_hold = b_full.predict(X_hold_lgb)
lgb_full_test = b_full.predict(X_test_lgb)

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d_lgb = X_full_lgb.loc[mask_90d]
y_90d_lgb = fit_full.loc[mask_90d, LABEL_COL].astype(int)
ds_90d_lgb = lgb.Dataset(X_90d_lgb, label=y_90d_lgb, categorical_feature=selected_cats, free_raw_data=False)
p_90d_lgb = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
print("Fitting LightGBM Tuned 90d (433 rounds)...")
b_90d = lgb.train(p_90d_lgb, ds_90d_lgb, num_boost_round=433)
lgb_90d_hold = b_90d.predict(X_hold_lgb)
lgb_90d_test = b_90d.predict(X_test_lgb)

r_lgb_full_hold = rankdata(lgb_full_hold) / len(lgb_full_hold)
r_lgb_90d_hold  = rankdata(lgb_90d_hold) / len(lgb_90d_hold)
r_lgb_champ_hold = 0.48 * r_lgb_full_hold + 0.52 * r_lgb_90d_hold
lgb_champ_score = average_precision_score(y_hold, r_lgb_champ_hold)
print(f"LightGBM Champion (0.5288 benchmark): PR-AUC = {lgb_champ_score:.4f}")

r_resnet_hold = rankdata(best_hold_preds) / len(best_hold_preds)

# Rank correlation between ResNet and LightGBM
corr = np.corrcoef(r_lgb_champ_hold, r_resnet_hold)[0, 1]
print(f"Rank Correlation r(Tabular ResNet, LightGBM Champion): {corr:.4f}")

# Sweep ensemble weights
print("\n--- Tree-Neural Rank Blend Sweep ---")
best_ensemble_score = lgb_champ_score
best_w = 1.0

for w in np.linspace(0.50, 1.0, 26):
    blend = w * r_lgb_champ_hold + (1.0 - w) * r_resnet_hold
    sc = average_precision_score(y_hold, blend)
    if sc > best_ensemble_score:
        best_ensemble_score = sc
        best_w = w
    print(f"  w={w:.2f} LGB + {1-w:.2f} ResNet: PR-AUC = {sc:.4f}")

print("\n" + "="*70)
print(f"FINAL TREE-NEURAL ENSEMBLE PR-AUC = {best_ensemble_score:.4f}")
print(f"Delta vs LightGBM Champion (0.5288): {best_ensemble_score - lgb_champ_score:+.4f}")
print(f"Optimal Weights: {best_w:.2f} LightGBM + {1-best_w:.2f} Tabular ResNet")
print("="*70)

GATE = 0.5400
if best_ensemble_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_ensemble_score:.4f} >= {GATE:.4f}) <<<")
    # Generate test predictions
    r_lgb_full_test = rankdata(lgb_full_test) / len(lgb_full_test)
    r_lgb_90d_test  = rankdata(lgb_90d_test) / len(lgb_90d_test)
    r_lgb_champ_test = 0.48 * r_lgb_full_test + 0.52 * r_lgb_90d_test
    r_resnet_test = rankdata(resnet_test_preds) / len(resnet_test_preds)
    
    final_test = best_w * r_lgb_champ_test + (1.0 - best_w) * r_resnet_test
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test})
    sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    out_path = SUBMISSIONS_DIR / f"CANDIDATE_resnet_tree_ensemble_{best_ensemble_score:.4f}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved qualified candidate: {out_path}")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_ensemble_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_ensemble_score:.4f}) <<<")
