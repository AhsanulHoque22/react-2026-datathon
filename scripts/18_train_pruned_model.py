"""Train and evaluate the pruned Top 160 feature set with 5-seed averaging and CatBoost blend."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import prepare_lgb_frame, make_pr_auc_feval, CAT_COLS

print("Loading features.pkl and optimal_pruned_features.csv...")
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
selected_cats = [c for c in CAT_COLS if c in selected_cols]

print(f"Using {len(selected_cols)} pruned features ({len(selected_cats)} categoricals)")

labeled = df[~df["is_test"]].copy()
test_df = df[df["is_test"]].copy()

holdout_start = pd.Timestamp("2026-07-01")
fit_df = labeled.loc[labeled[TIME_COL] < holdout_start]
hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]

y_fit = fit_df[LABEL_COL].astype(int)
y_hold = hold_df[LABEL_COL].astype(int)

X_fit = prepare_lgb_frame(fit_df, selected_cols, selected_cats)
X_hold = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
X_test = prepare_lgb_frame(test_df, selected_cols, selected_cats)

# -----------------------------------------------------------------
# 1. 5-Seed Averaged LightGBM on Pruned Top 160 Features
# -----------------------------------------------------------------
print("\n" + "="*65)
print("1. Training 5-Seed Averaged LightGBM on Top 160...")
print("="*65)

SEEDS = [0, 1, 2, 3, 4]
lgb_hold_preds = np.zeros(len(hold_df), dtype="float64")
lgb_test_preds = np.zeros(len(test_df), dtype="float64")
seed_scores = []
best_iters = []

for s in SEEDS:
    params = dict(
        objective="binary", metric="None", seed=s, bagging_seed=s,
        feature_fraction_seed=s, verbosity=-1,
        learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )
    train_set = lgb.Dataset(X_fit, label=y_fit, categorical_feature=selected_cats, free_raw_data=False)
    hold_set = lgb.Dataset(X_hold, label=y_hold, categorical_feature=selected_cats, reference=train_set, free_raw_data=False)
    feval = make_pr_auc_feval(y_hold.values, seed=s)
    
    t0 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=3000, valid_sets=[hold_set], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    p_hold = booster.predict(X_hold, num_iteration=booster.best_iteration)
    p_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    sc = average_precision_score(y_hold, p_hold)
    seed_scores.append(sc)
    best_iters.append(booster.best_iteration)
    lgb_hold_preds += p_hold / len(SEEDS)
    lgb_test_preds += p_test / len(SEEDS)
    print(f"  Seed {s}: PR-AUC = {sc:.4f} (best_iter = {booster.best_iteration}, {time.time()-t0:.1f}s)")

score_lgb_avg = average_precision_score(y_hold, lgb_hold_preds)
print(f"\n--> 5-Seed Average LightGBM Score: PR-AUC = {score_lgb_avg:.4f} (Single-seed mean: {np.mean(seed_scores):.4f})")

# -----------------------------------------------------------------
# 2. CatBoost on Pruned Top 160 Features
# -----------------------------------------------------------------
print("\n" + "="*65)
print("2. Training CatBoost on Top 160...")
print("="*65)

num_cols = [c for c in selected_cols if c not in selected_cats]
onehot_fit = pd.get_dummies(fit_df[selected_cats].astype(str), prefix=selected_cats)
onehot_hold = pd.get_dummies(hold_df[selected_cats].astype(str), prefix=selected_cats)
onehot_test = pd.get_dummies(test_df[selected_cats].astype(str), prefix=selected_cats)

onehot_fit, onehot_hold = onehot_fit.align(onehot_hold, join="left", axis=1, fill_value=0)
onehot_fit, onehot_test = onehot_fit.align(onehot_test, join="left", axis=1, fill_value=0)

X_fit_cat = pd.concat([fit_df[num_cols].reset_index(drop=True), onehot_fit.reset_index(drop=True)], axis=1)
X_hold_cat = pd.concat([hold_df[num_cols].reset_index(drop=True), onehot_hold.reset_index(drop=True)], axis=1)
X_test_cat = pd.concat([test_df[num_cols].reset_index(drop=True), onehot_test.reset_index(drop=True)], axis=1)

cat_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.03, depth=7,
    loss_function="Logloss", eval_metric="PRAUC",
    random_seed=SEED, early_stopping_rounds=100, verbose=False, thread_count=-1,
)
t0 = time.time()
cat_model.fit(Pool(X_fit_cat, y_fit), eval_set=Pool(X_hold_cat, y_hold), use_best_model=True)
cat_hold_preds = cat_model.predict_proba(X_hold_cat)[:, 1]
cat_test_preds = cat_model.predict_proba(X_test_cat)[:, 1]
score_cat = average_precision_score(y_hold, cat_hold_preds)
print(f"--> CatBoost Score: PR-AUC = {score_cat:.4f} (best_iter = {cat_model.get_best_iteration()}, {time.time()-t0:.1f}s)")

# -----------------------------------------------------------------
# 3. Rank Ensemble & Quality Gate Check
# -----------------------------------------------------------------
print("\n" + "="*65)
print("3. Evaluating Rank Ensemble & Quality Gate (>= 0.5400)...")
print("="*65)

rank_lgb_hold = rankdata(lgb_hold_preds) / len(lgb_hold_preds)
rank_cat_hold = rankdata(cat_hold_preds) / len(cat_hold_preds)

best_score = max(score_lgb_avg, score_cat)
best_w = 1.0
for w in np.linspace(0.0, 1.0, 21):
    blend_hold = w * rank_lgb_hold + (1 - w) * rank_cat_hold
    sc = average_precision_score(y_hold, blend_hold)
    if sc > best_score:
        best_score = sc
        best_w = w

print(f"5-Seed LightGBM: {score_lgb_avg:.4f}")
print(f"CatBoost:        {score_cat:.4f}")
print(f"Best Ensemble:   PR-AUC = {best_score:.4f} (Weight: {best_w:.2f} LGB + {1-best_w:.2f} CAT)")

GATE = 0.5400
if best_score >= GATE:
    print(f"\n>>> QUALITY GATE PASSED! ({best_score:.4f} >= {GATE:.4f}) <<<")
    # Generate final ensemble predictions
    rank_lgb_test = rankdata(lgb_test_preds) / len(lgb_test_preds)
    rank_cat_test = rankdata(cat_test_preds) / len(cat_test_preds)
    final_preds = best_w * rank_lgb_test + (1 - best_w) * rank_cat_test
    
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": final_preds})
    sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    
    out_path = SUBMISSIONS_DIR / "CANDIDATE_PRUNED_0.5400_PLUS.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote qualified submission to {out_path} (shape={sub.shape})")
else:
    print(f"\n>>> QUALITY GATE STATUS: {best_score:.4f} vs target {GATE:.4f} (gap: {GATE - best_score:.4f}) <<<")
    # Also save the top pruned submission candidate for inspection
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": lgb_test_preds})
    sub = sub.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    out_path = SUBMISSIONS_DIR / f"CANDIDATE_pruned_top160_lgb_{score_lgb_avg:.4f}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved candidate submission file to {out_path}")
