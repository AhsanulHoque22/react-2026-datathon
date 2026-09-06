"""Train Tabular ResNet on Full Train (672k rows, Jan-Jul) & Evaluate Multi-Horizon Neural Ensemble.

Hypothesis:
Deep neural networks thrive on larger sample sizes. Tripling the training rows from
234k (90d) to 672k (Full Train) allows Tabular ResNet to learn robust long-term representations,
reducing variance and improving generalizability on the held-out tail.
"""
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
from src.model import prepare_lgb_frame, CAT_COLS
from src.nn_models import TabularResNet

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("FULL TRAIN TABULAR RESNET & DUAL-HORIZON NEURAL ENSEMBLE")
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

print(f"Full Train: {len(fit_full):,} rows | Holdout: {len(hold_df):,} rows")

# 2. Preprocess Categoricals on Full Train
cat_cardinalities = []
train_cats = []
hold_cats = []
test_cats = []
for c in selected_cats:
    val_counts = fit_full[c].value_counts()
    categories = list(val_counts.index)
    cat_to_id = {val: idx + 1 for idx, val in enumerate(categories)}
    cat_cardinalities.append(len(categories) + 1)
    train_cats.append(fit_full[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    hold_cats.append(hold_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))
    test_cats.append(test_df[c].map(cat_to_id).fillna(0).values.astype(np.int64))

X_cat_full = np.column_stack(train_cats) if train_cats else np.empty((len(fit_full), 0), dtype=np.int64)
X_cat_hold = np.column_stack(hold_cats)  if hold_cats  else np.empty((len(hold_df), 0), dtype=np.int64)
X_cat_test = np.column_stack(test_cats)  if test_cats  else np.empty((len(test_df), 0), dtype=np.int64)

# 3. Preprocess Numericals
print("Scaling numerical features on Full Train (672k rows)...")
X_num_full_raw = fit_full[selected_nums].copy()
X_num_hold_raw = hold_df[selected_nums].copy()
X_num_test_raw = test_df[selected_nums].copy()

medians = X_num_full_raw.median()
X_num_full_raw = X_num_full_raw.fillna(medians)
X_num_hold_raw = X_num_hold_raw.fillna(medians)
X_num_test_raw = X_num_test_raw.fillna(medians)

scaler = StandardScaler()
X_num_full = np.clip(scaler.fit_transform(X_num_full_raw), -5.0, 5.0).astype(np.float32)
X_num_hold = np.clip(scaler.transform(X_num_hold_raw), -5.0, 5.0).astype(np.float32)
X_num_test = np.clip(scaler.transform(X_num_test_raw), -5.0, 5.0).astype(np.float32)

y_full_arr = fit_full[LABEL_COL].values.astype(np.float32)
y_hold_arr = hold_df[LABEL_COL].values.astype(np.float32)

BATCH_SIZE = 1024
full_dataset = TensorDataset(torch.from_numpy(X_num_full), torch.from_numpy(X_cat_full), torch.from_numpy(y_full_arr))
hold_dataset = TensorDataset(torch.from_numpy(X_num_hold), torch.from_numpy(X_cat_hold), torch.from_numpy(y_hold_arr))
test_dataset = TensorDataset(torch.from_numpy(X_num_test), torch.from_numpy(X_cat_test))

full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
hold_loader = DataLoader(hold_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

# 4. Model Architecture & Training
model_full = TabularResNet(
    num_features=len(selected_nums),
    cat_cardinalities=cat_cardinalities,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
)

EPOCHS = 7
optimizer = torch.optim.AdamW(model_full.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()

print(f"\nTraining Full Train Tabular ResNet ({sum(p.numel() for p in model_full.parameters()):,} parameters) for {EPOCHS} epochs...")

best_val_auc = 0.0
best_hold_pred_full = None
best_state = None

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    model_full.train()
    total_loss = 0.0
    nb = 0
    for b_num, b_cat, b_y in full_loader:
        optimizer.zero_grad()
        logits = model_full(b_num, b_cat)
        loss = criterion(logits, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_full.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()
    
    # Eval
    model_full.eval()
    h_preds = []
    with torch.no_grad():
        for b_num, b_cat, _ in hold_loader:
            h_preds.append(torch.sigmoid(model_full(b_num, b_cat)).cpu().numpy())
    v_preds = np.concatenate(h_preds)
    v_auc = average_precision_score(y_hold_arr, v_preds)
    
    is_best = v_auc > best_val_auc
    if is_best:
        best_val_auc = v_auc
        best_hold_pred_full = v_preds.copy()
        best_state = {k: v.cpu().clone() for k, v in model_full.state_dict().items()}
        
    star = " *** BEST ***" if is_best else ""
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/nb:.5f} | Val PR-AUC: {v_auc:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} ({time.time()-t_ep:.1f}s){star}")

print(f"\n>>> Full Train Tabular ResNet Best PR-AUC = {best_val_auc:.4f} <<<")

# Generate Test Predictions for Full ResNet
model_full.load_state_dict(best_state)
model_full.eval()
t_preds = []
with torch.no_grad():
    for b_num, b_cat in test_loader:
        t_preds.append(torch.sigmoid(model_full(b_num, b_cat)).cpu().numpy())
test_pred_full = np.concatenate(t_preds)

r_res_full_h = rankdata(best_hold_pred_full) / len(best_hold_pred_full)
r_res_full_t = rankdata(test_pred_full) / len(test_pred_full)

# -------------------------------------------------------------------------
# Dual-Horizon ResNet & Tree-Neural Ensembling
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("ENSEMBLING: DUAL-HORIZON RESNET + LIGHTGBM CHAMPION")
print("="*70)

# Load LightGBM Champion
import lightgbm as lgb
X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)

ds_full_lgb = lgb.Dataset(X_full_lgb, label=fit_full[LABEL_COL].astype(int), categorical_feature=selected_cats, free_raw_data=False)
p_full_lgb = dict(objective="binary", metric="None", seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                  learning_rate=0.02, num_leaves=127, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)
b_full = lgb.train(p_full_lgb, ds_full_lgb, num_boost_round=882)
lgb_full_hold = b_full.predict(X_hold_lgb)
lgb_full_test = b_full.predict(X_test_lgb)

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d_lgb = X_full_lgb.loc[mask_90d]
y_90d_lgb = fit_full.loc[mask_90d, LABEL_COL].astype(int)
ds_90d_lgb = lgb.Dataset(X_90d_lgb, label=y_90d_lgb, categorical_feature=selected_cats, free_raw_data=False)
p_90d_lgb = dict(objective="binary", metric="None", seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                 learning_rate=0.02, num_leaves=63, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100)
b_90d = lgb.train(p_90d_lgb, ds_90d_lgb, num_boost_round=433)
lgb_90d_hold = b_90d.predict(X_hold_lgb)
lgb_90d_test = b_90d.predict(X_test_lgb)

r_lgb_full_h = rankdata(lgb_full_hold) / len(lgb_full_hold)
r_lgb_90d_h  = rankdata(lgb_90d_hold)  / len(lgb_90d_hold)
r_lgb_champ_h = 0.48 * r_lgb_full_h + 0.52 * r_lgb_90d_h
lgb_champ_score = average_precision_score(y_hold_arr, r_lgb_champ_h)
print(f"LightGBM Champion PR-AUC: {lgb_champ_score:.4f}")

r_lgb_full_t = rankdata(lgb_full_test) / len(lgb_full_test)
r_lgb_90d_t  = rankdata(lgb_90d_test)  / len(lgb_90d_test)
r_lgb_champ_t = 0.48 * r_lgb_full_t + 0.52 * r_lgb_90d_t

# Rank correlation between Full Train ResNet and LightGBM Champion
corr = np.corrcoef(r_lgb_champ_h, r_res_full_h)[0, 1]
print(f"Rank Correlation r(Full Train ResNet, LightGBM Champion): {corr:.4f}")

# Sweep ensemble weights between LightGBM Champion and Full Train ResNet
print("\n--- LightGBM Champion + Full Train ResNet Sweep ---")
best_score = lgb_champ_score
best_w = 1.0
for w in np.linspace(0.70, 0.98, 29):
    blend = w * r_lgb_champ_h + (1.0 - w) * r_res_full_h
    sc = average_precision_score(y_hold_arr, blend)
    if sc > best_score:
        best_score = sc
        best_w = w
    print(f"  w_lgb={w:.2f} + w_res_full={1-w:.2f}: PR-AUC = {sc:.4f}")

print(f"\nOptimal Blend: PR-AUC = {best_score:.4f} (LGB={best_w:.2f}, ResNet_Full={1-best_w:.2f})")
print(f"Delta vs LightGBM Champion: {best_score - lgb_champ_score:+.4f}")

GATE = 0.5400
if best_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_score:.4f} >= {GATE:.4f}) <<<")
    final_test = best_w * r_lgb_champ_t + (1.0 - best_w) * r_res_full_t
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test})
    sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    out_path = SUBMISSIONS_DIR / f"CANDIDATE_full_resnet_ensemble_{best_score:.4f}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved qualified candidate: {out_path}")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_score:.4f}) <<<")
