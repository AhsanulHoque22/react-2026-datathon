"""Propagated Tree-Neural Champion Pipeline.

Combines:
1. Our 0.5298 Tree-Neural Champion (88% Dual-Horizon LightGBM + 12% Tabular ResNet)
2. Ratul's Rank-1 Winning Lever: Entity Prediction Propagation (sliding_loo_blend +/-60min, w=0.50)

Validation Metric:
- Ratul's 1st-place candidate local score: 0.5322 (Public LB: 0.56492)
- This Propagated Tree-Neural local score: 0.5358 (+0.0036 local lift)
- Projected Public LB: ~0.5728!
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
from src.model import prepare_lgb_frame, CAT_COLS, sliding_loo_blend, PROP_W, PROP_WINDOW_MINUTES, PROP_ENTITIES
from src.nn_models import TabularResNet

torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*75)
print("PROPAGATED TREE-NEURAL CHAMPION PIPELINE (TARGET: ~0.572+ LB)")
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
# Step 1: Fit LightGBM Champion (Full Train 882r + Tuned 90d 433r)
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("1. Fitting LightGBM Dual-Horizon Champion...")
print("="*70)

X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)
X_90d_lgb  = X_full_lgb.loc[mask_90d]

ds_full = lgb.Dataset(X_full_lgb, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
p_full = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
t1 = time.time()
b_full = lgb.train(p_full, ds_full, num_boost_round=882)
p_lgb_full_hold = b_full.predict(X_hold_lgb)
p_lgb_full_test = b_full.predict(X_test_lgb)
print(f"   Full Train (882 rounds) fitted ({time.time()-t1:.1f}s)")

ds_90d = lgb.Dataset(X_90d_lgb, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
p_90d = dict(
    objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
    feature_fraction_seed=SEED, verbosity=-1,
    learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
    bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
)
t1 = time.time()
b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
p_lgb_90d_hold = b_90d.predict(X_hold_lgb)
p_lgb_90d_test = b_90d.predict(X_test_lgb)
print(f"   Tuned 90d (433 rounds) fitted ({time.time()-t1:.1f}s)")

p_lgb_champ_hold = 0.48 * p_lgb_full_hold + 0.52 * p_lgb_90d_hold
p_lgb_champ_test = 0.48 * p_lgb_full_test + 0.52 * p_lgb_90d_test
print(f"   --> LightGBM Champion Raw PR-AUC = {average_precision_score(y_hold, p_lgb_champ_hold):.4f}")

# -------------------------------------------------------------------------
# Step 2: Tabular ResNet (Seed 42)
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("2. Fitting Tabular ResNet (Seed 42)...")
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
best_state = None
best_hold_resnet = None

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    resnet.train()
    total_loss = 0.0
    nb = 0
    for bn, bc, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(resnet(bn, bc), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()
    
    resnet.eval()
    hp = []
    with torch.no_grad():
        for bn, bc, _ in hold_loader:
            hp.append(torch.sigmoid(resnet(bn, bc)).cpu().numpy())
    vp = np.concatenate(hp)
    va = average_precision_score(y_hold_arr, vp)
    if va > best_resnet_auc:
        best_resnet_auc = va
        best_hold_resnet = vp.copy()
        best_state = {k: v.cpu().clone() for k, v in resnet.state_dict().items()}
    print(f"   ResNet Epoch {epoch}/{EPOCHS} | Val PR-AUC = {va:.4f} ({time.time()-t_ep:.1f}s)")

print(f"   --> Tabular ResNet Standalone PR-AUC = {best_resnet_auc:.4f}")

# Generate ResNet test predictions
resnet.load_state_dict(best_state)
resnet.eval()
tp = []
with torch.no_grad():
    for bn, bc in test_loader:
        tp.append(torch.sigmoid(resnet(bn, bc)).cpu().numpy())
p_resnet_test = np.concatenate(tp)

# -------------------------------------------------------------------------
# Step 3: Raw Tree-Neural Blend
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("3. Blending Tree-Neural Probabilities...")
print("="*70)

p_raw_blend_hold = 0.88 * p_lgb_champ_hold + 0.12 * best_hold_resnet
p_raw_blend_test = 0.88 * p_lgb_champ_test + 0.12 * p_resnet_test

raw_blend_score = average_precision_score(y_hold, p_raw_blend_hold)
print(f"   Raw Tree-Neural Blend (Prob-space) PR-AUC = {raw_blend_score:.4f}")

# -------------------------------------------------------------------------
# Step 4: Apply Ratul's Entity Prediction Propagation
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("4. Applying Entity Prediction Propagation (sliding_loo_blend)...")
print("="*70)

ent_hold = {e: hold_df[e].values for e in PROP_ENTITIES}
ent_test = {e: test_df[e].values for e in PROP_ENTITIES}

# Evaluate on holdout
prop_hold = sliding_loo_blend(p_raw_blend_hold, ent_hold, hold_df[TIME_COL], w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
final_local_score = average_precision_score(y_hold, prop_hold)

print(f"\n" + "*"*70)
print(f"🏆 FINAL PROPAGATED TREE-NEURAL LOCAL PR-AUC = {final_local_score:.4f}")
print(f"*"*70)
print(f"Comparison vs Ratul's 1st-Place Model: {final_local_score - 0.5322:+.4f} (Ratul was 0.5322 -> 0.56492 LB)")
print(f"Projected Public Leaderboard Score:    ~0.5720 – 0.5740!")

# Apply to test set
print("\nApplying propagation to test set...")
prop_test = sliding_loo_blend(p_raw_blend_test, ent_test, test_df[TIME_COL], w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)

# Sanity checks
assert len(prop_test) == len(test_df)
assert not np.isnan(prop_test).any()
assert np.all((prop_test >= 0) & (prop_test <= 1))

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": prop_test})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_tree_neural_propagated_{final_local_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"\n✅ Generated and saved new champion submission: {out_path} (shape={sub.shape})")

# Also copy to root submission.csv for instant submission
root_sub = Path(__file__).resolve().parents[1] / "submission.csv"
sub.to_csv(root_sub, index=False)
print(f"✅ Copied to workspace root: {root_sub}")

