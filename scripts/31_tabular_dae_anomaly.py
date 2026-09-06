"""Tabular Denoising Autoencoder (DAE) for Unsupervised Anomaly Representation.

1. Trains an unsupervised DAE with swap noise (p=0.15) on the 158 numerical features
   using the 90-day training distribution.
2. Computes the sample-wise reconstruction error MSE(x, x_hat).
3. Evaluates the stand-alone anomaly detection PR-AUC of reconstruction error.
4. Measures rank correlation with LightGBM Champion and Tabular ResNet.
5. Tests rank ensembling and feature augmentation against the >= 0.5400 gate.
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

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("TABULAR DENOISING AUTOENCODER (DAE) EXPERIMENT")
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

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
train_90d = fit_full.loc[mask_90d].copy()

y_train = train_90d[LABEL_COL].values.astype(np.float32)
y_hold = hold_df[LABEL_COL].values.astype(np.float32)

# 2. Preprocess Numerical Features (Median Imputation + Scaling + Clipping)
print("Preprocessing numerical features...")
X_train_raw = train_90d[selected_nums].copy()
X_hold_raw  = hold_df[selected_nums].copy()
X_test_raw  = test_df[selected_nums].copy()

medians = X_train_raw.median()
X_train_raw = X_train_raw.fillna(medians)
X_hold_raw  = X_hold_raw.fillna(medians)
X_test_raw  = X_test_raw.fillna(medians)

scaler = StandardScaler()
X_train_scaled = np.clip(scaler.fit_transform(X_train_raw), -5.0, 5.0).astype(np.float32)
X_hold_scaled  = np.clip(scaler.transform(X_hold_raw), -5.0, 5.0).astype(np.float32)
X_test_scaled  = np.clip(scaler.transform(X_test_raw), -5.0, 5.0).astype(np.float32)

num_features = X_train_scaled.shape[1]
print(f"Numerical feature dimension: {num_features}")

# 3. DAE Architecture
class TabularDAE(nn.Module):
    """Denoising Autoencoder with swap noise for tabular anomaly representation."""
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, in_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

def apply_swap_noise(x, p=0.15):
    """Swap p fraction of values with values from random rows in the mini-batch."""
    mask = torch.rand_like(x) < p
    batch_size = x.size(0)
    rand_idx = torch.randint(0, batch_size, (batch_size,), device=x.device)
    corrupted = torch.where(mask, x[rand_idx], x)
    return corrupted

# 4. DataLoader
BATCH_SIZE = 1024
train_tensor = torch.from_numpy(X_train_scaled)
hold_tensor  = torch.from_numpy(X_hold_scaled)
test_tensor  = torch.from_numpy(X_test_scaled)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
hold_loader  = DataLoader(TensorDataset(hold_tensor), batch_size=BATCH_SIZE*2, shuffle=False)
test_loader  = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE*2, shuffle=False)

# 5. Train DAE
model = TabularDAE(in_dim=num_features, latent_dim=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
EPOCHS = 12
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.MSELoss()

print(f"\nTraining Tabular DAE ({sum(p.numel() for p in model.parameters()):,} parameters) for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    model.train()
    total_loss = 0.0
    nb = 0
    for (b_x,) in train_loader:
        optimizer.zero_grad()
        b_corrupted = apply_swap_noise(b_x, p=0.15)
        b_rec, _ = model(b_corrupted)
        # Loss against clean original input
        loss = criterion(b_rec, b_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()
    
    # Eval reconstruction error on holdout
    model.eval()
    hold_losses = []
    with torch.no_grad():
        for (b_x,) in hold_loader:
            b_rec, _ = model(b_x)
            err = torch.mean((b_x - b_rec)**2, dim=1).cpu().numpy()
            hold_losses.append(err)
    hold_err = np.concatenate(hold_losses)
    # PR-AUC of reconstruction error as anomaly score
    rec_prauc = average_precision_score(y_hold, hold_err)
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Recon MSE: {total_loss/nb:.4f} | Holdout Recon PR-AUC: {rec_prauc:.4f} | {time.time()-t_ep:.1f}s")

# 6. Extract Reconstruction Errors
model.eval()
with torch.no_grad():
    hold_err_list = []
    for (b_x,) in hold_loader:
        b_rec, _ = model(b_x)
        err = torch.mean((b_x - b_rec)**2, dim=1).cpu().numpy()
        hold_err_list.append(err)
    dae_hold_err = np.concatenate(hold_err_list)

    test_err_list = []
    for (b_x,) in test_loader:
        b_rec, _ = model(b_x)
        err = torch.mean((b_x - b_rec)**2, dim=1).cpu().numpy()
        test_err_list.append(err)
    dae_test_err = np.concatenate(test_err_list)

dae_prauc = average_precision_score(y_hold, dae_hold_err)
print(f"\n>>> Final Stand-Alone DAE Anomaly PR-AUC = {dae_prauc:.4f} <<<")

# Compare reconstruction error between Normal and Fraud
print(f"Mean Recon Error (Non-Fraud): {dae_hold_err[y_hold == 0].mean():.4f}")
print(f"Mean Recon Error (Fraud):     {dae_hold_err[y_hold == 1].mean():.4f}")
print(f"Ratio (Fraud / Non-Fraud):    {dae_hold_err[y_hold == 1].mean() / dae_hold_err[y_hold == 0].mean():.2f}x")

# 7. Check correlation and ensembling with LightGBM Champion
import lightgbm as lgb
X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)

# Quick LGB Champion
ds_full_lgb = lgb.Dataset(X_full_lgb, label=fit_full[LABEL_COL].astype(int), categorical_feature=selected_cats, free_raw_data=False)
p_full_lgb = dict(objective="binary", metric="None", seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                  learning_rate=0.02, num_leaves=127, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)
b_full = lgb.train(p_full_lgb, ds_full_lgb, num_boost_round=882)
lgb_full_hold = b_full.predict(X_hold_lgb)

mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
X_90d_lgb = X_full_lgb.loc[mask_90d]
y_90d_lgb = fit_full.loc[mask_90d, LABEL_COL].astype(int)
ds_90d_lgb = lgb.Dataset(X_90d_lgb, label=y_90d_lgb, categorical_feature=selected_cats, free_raw_data=False)
p_90d_lgb = dict(objective="binary", metric="None", seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                 learning_rate=0.02, num_leaves=63, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100)
b_90d = lgb.train(p_90d_lgb, ds_90d_lgb, num_boost_round=433)
lgb_90d_hold = b_90d.predict(X_hold_lgb)

r_lgb_champ_h = 0.48 * (rankdata(lgb_full_hold)/len(lgb_full_hold)) + 0.52 * (rankdata(lgb_90d_hold)/len(lgb_90d_hold))
lgb_champ_score = average_precision_score(y_hold, r_lgb_champ_h)
print(f"\nLightGBM Champion PR-AUC: {lgb_champ_score:.4f}")

r_dae_h = rankdata(dae_hold_err) / len(dae_hold_err)
corr = np.corrcoef(r_lgb_champ_h, r_dae_h)[0, 1]
print(f"Rank Correlation r(DAE Recon Error, LightGBM Champion): {corr:.4f}")

# Sweep ensemble
best_score = lgb_champ_score
best_w = 1.0
for w in np.linspace(0.90, 1.0, 21):
    combo = w * r_lgb_champ_h + (1.0 - w) * r_dae_h
    sc = average_precision_score(y_hold, combo)
    if sc > best_score:
        best_score = sc
        best_w = w
    print(f"  w_lgb={w:.3f} + w_dae={1-w:.3f}: PR-AUC = {sc:.4f}")

print(f"\nBest DAE + LightGBM Blend: {best_score:.4f} (delta: {best_score - lgb_champ_score:+.4f})")
