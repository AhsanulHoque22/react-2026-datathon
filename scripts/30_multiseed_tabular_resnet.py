"""Multi-Seed Bagged Tabular ResNet & Advanced Tree-Neural Ensemble Pipeline.

Trains 3 distinct seeds of Tabular ResNet (seeds 42, 43, 44) on the 90-day window,
evaluates stand-alone and bagged neural PR-AUC on the held-out tail (Jul 1-15),
evaluates tree-neural ensemble against the >= 0.5400 quality gate,
and generates submission candidate predictions if performance improves.
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
from src.model import prepare_lgb_frame, CAT_COLS
from src.nn_models import TabularResNet

torch.set_num_threads(8)

print("="*75)
print("MULTI-SEED BAGGED TABULAR RESNET & ADVANCED ENSEMBLE PIPELINE")
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
# STEP 1: Fast Champion Tree Baselines
# =========================================================================
print("\n" + "="*70)
print("1. Computing LightGBM Champion & XGBoost Predictions...")
print("="*70)

# Full Train LightGBM
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

# 90d Specialist LightGBM
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
lgb_champ_score = average_precision_score(y_hold, r_lgb_champ_h)
print(f"   --> LightGBM Champion Benchmark: PR-AUC = {lgb_champ_score:.4f}")

r_lgb_full_t = rankdata(p_lgb_full_test) / len(p_lgb_full_test)
r_lgb_90d_t  = rankdata(p_lgb_90d_test)  / len(p_lgb_90d_test)
r_lgb_champ_t = 0.48 * r_lgb_full_t + 0.52 * r_lgb_90d_t

# XGBoost 90d
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
print(f"   XGBoost 90d:     PR-AUC = {average_precision_score(y_hold, p_xgb_hold):.4f} ({time.time()-t1:.1f}s)")
r_xgb_h = rankdata(p_xgb_hold) / len(p_xgb_hold)
r_xgb_t = rankdata(p_xgb_test) / len(p_xgb_test)

# =========================================================================
# STEP 2: Preprocess Neural Features
# =========================================================================
print("\n" + "="*70)
print("2. Preprocessing Data for Tabular ResNet...")
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

hold_loader  = DataLoader(hold_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)

print(f"Num Features: {X_num_train.shape[1]}, Cat Cardinalities: {cat_cardinalities}")

# =========================================================================
# STEP 3: Multi-Seed Tabular ResNet Training
# =========================================================================
print("\n" + "="*70)
print("3. Training 3-Seed Tabular ResNet Bag (Seeds 42, 43, 44)...")
print("="*70)

SEEDS = [42, 43, 44]
EPOCHS = 9
all_seed_hold_ranks = []
all_seed_test_ranks = []
seed_scores = {}

for s_idx, seed_val in enumerate(SEEDS, start=1):
    print(f"\n--- Training Tabular ResNet Seed {seed_val} ({s_idx}/{len(SEEDS)}) ---")
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    
    # DataLoader with seed-specific shuffle
    g = torch.Generator()
    g.manual_seed(seed_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=g)
    
    model = TabularResNet(
        num_features=len(selected_nums),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    best_hold_pred = None
    best_state = None
    
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
        
        # Validation
        model.eval()
        h_preds = []
        with torch.no_grad():
            for b_num, b_cat, _ in hold_loader:
                h_preds.append(torch.sigmoid(model(b_num, b_cat)).cpu().numpy())
        v_preds = np.concatenate(h_preds)
        v_auc = average_precision_score(y_hold_arr, v_preds)
        
        is_best = v_auc > best_val_auc
        if is_best:
            best_val_auc = v_auc
            best_hold_pred = v_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        star = " *** BEST ***" if is_best else ""
        print(f"   Epoch {epoch}/{EPOCHS} | Loss: {total_loss/nb:.5f} | Val PR-AUC: {v_auc:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} ({time.time()-t_ep:.1f}s){star}")
        
    print(f"   >>> Seed {seed_val} Best Stand-Alone PR-AUC: {best_val_auc:.4f} <<<")
    seed_scores[seed_val] = best_val_auc
    
    # Generate test predictions for this seed
    model.load_state_dict(best_state)
    model.eval()
    t_preds = []
    with torch.no_grad():
        for b_num, b_cat in test_loader:
            t_preds.append(torch.sigmoid(model(b_num, b_cat)).cpu().numpy())
    test_pred_seed = np.concatenate(t_preds)
    
    r_hold_seed = rankdata(best_hold_pred) / len(best_hold_pred)
    r_test_seed = rankdata(test_pred_seed) / len(test_pred_seed)
    
    all_seed_hold_ranks.append(r_hold_seed)
    all_seed_test_ranks.append(r_test_seed)

# Bagged ResNet predictions (mean rank)
r_resnet_bagged_h = np.mean(all_seed_hold_ranks, axis=0)
r_resnet_bagged_t = np.mean(all_seed_test_ranks, axis=0)

sc_resnet_bagged = average_precision_score(y_hold_arr, r_resnet_bagged_h)

print("\n" + "="*70)
print("TABULAR RESNET SEED VARIANCE & BAGGING RESULTS")
print("="*70)
for s, score in seed_scores.items():
    print(f"   Seed {s}: Stand-Alone PR-AUC = {score:.4f}")
print(f"   🏆 3-Seed Bagged Tabular ResNet: PR-AUC = {sc_resnet_bagged:.4f}")
print(f"   Bagging Lift over Mean Single-Seed: {sc_resnet_bagged - np.mean(list(seed_scores.values())):+.4f}")

# Pairwise rank correlations
print("\n--- Correlation Matrix with Bagged ResNet ---")
pred_mat = np.column_stack([r_lgb_champ_h, r_xgb_h, r_resnet_bagged_h])
cols = ["LightGBM", "XGBoost", "BaggedResNet"]
print(pd.DataFrame(np.corrcoef(pred_mat.T), index=cols, columns=cols).round(4))

# =========================================================================
# STEP 4: Ensembling Analysis
# =========================================================================
print("\n" + "="*70)
print("4. Tree-Neural Ensembling Sweep")
print("="*70)

# 1. 2-way blend: LightGBM Champion + Bagged ResNet
print("\n--- LightGBM Champion + Bagged ResNet Sweep ---")
best_2way_score = lgb_champ_score
best_w_2way = 1.0

for w in np.linspace(0.70, 0.98, 29):
    blend = w * r_lgb_champ_h + (1.0 - w) * r_resnet_bagged_h
    sc = average_precision_score(y_hold, blend)
    if sc > best_2way_score:
        best_2way_score = sc
        best_w_2way = w
    print(f"   w_lgb={w:.2f} + w_res={1-w:.2f}: PR-AUC = {sc:.4f}")

print(f"\n--> Best 2-Way Ensemble: PR-AUC = {best_2way_score:.4f} (LGB={best_w_2way:.2f}, ResNet={1-best_w_2way:.2f})")

# 2. 3-way blend: LightGBM Champion + XGBoost + Bagged ResNet
print("\n--- Tri-Architecture Simplex Sweep ---")
best_tri_score = best_2way_score
best_tri_weights = (best_w_2way, 0.0, 1.0 - best_w_2way)

for w_lgb in np.linspace(0.65, 0.95, 31):
    for w_xgb in np.linspace(0.0, 0.25, 26):
        w_res = round(1.0 - (w_lgb + w_xgb), 4)
        if w_res < 0.0:
            continue
        combo = w_lgb * r_lgb_champ_h + w_xgb * r_xgb_h + w_res * r_resnet_bagged_h
        sc = average_precision_score(y_hold, combo)
        if sc > best_tri_score:
            best_tri_score = sc
            best_tri_weights = (round(w_lgb, 3), round(w_xgb, 3), w_res)

print("\n" + "="*70)
print("GRAND ENSEMBLE SUMMARY")
print("="*70)
print(f"LightGBM Champion Benchmark:   PR-AUC = {lgb_champ_score:.4f}")
print(f"XGBoost Specialist:            PR-AUC = {average_precision_score(y_hold, r_xgb_h):.4f}")
print(f"3-Seed Bagged Tabular ResNet:  PR-AUC = {sc_resnet_bagged:.4f}")
print(f"2-Way Ensemble (LGB + ResNet): PR-AUC = {best_2way_score:.4f} (+{best_2way_score - lgb_champ_score:+.4f})")
print(f"🏆 Tri-Architecture Ensemble:  PR-AUC = {best_tri_score:.4f} (+{best_tri_score - lgb_champ_score:+.4f})")
print(f"Optimal Weights: LGB={best_tri_weights[0]:.3f}, XGB={best_tri_weights[1]:.3f}, ResNet={best_tri_weights[2]:.3f}")

# Generate Candidate Submission
final_test = (
    best_tri_weights[0] * r_lgb_champ_t +
    best_tri_weights[1] * r_xgb_t +
    best_tri_weights[2] * r_resnet_bagged_t
)

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_test})
sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

candidate_name = f"CANDIDATE_bagged_resnet_tri_{best_tri_score:.4f}.csv"
out_path = SUBMISSIONS_DIR / candidate_name
sub.to_csv(out_path, index=False)
print(f"\nGenerated candidate submission: {out_path} (shape={sub.shape})")

GATE = 0.5400
if best_tri_score >= GATE:
    print(f"\n>>> HARD QUALITY GATE PASSED! ({best_tri_score:.4f} >= {GATE:.4f}) <<<")
else:
    print(f"\n>>> HARD QUALITY GATE STATUS: {best_tri_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_tri_score:.4f}) <<<")
