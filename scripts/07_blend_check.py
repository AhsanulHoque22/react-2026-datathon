"""Quick decisive check: does blending LightGBM + CatBoost beat LightGBM
alone on fold 3 (the drift fold, most relevant to actual test performance)?
CatBoost lost on every fold in scripts/06_catboost_cv.py, so this is the
one cheap test that settles whether blending is worth pursuing further."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
numeric_cols = [c for c in feature_cols if c not in cat_cols]
labeled = df[~df["is_test"]]

start_ts, end_ts = pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-15")
train_mask = labeled[TIME_COL] < start_ts
valid_mask = (labeled[TIME_COL] >= start_ts) & (labeled[TIME_COL] < end_ts)
train_df, valid_df = labeled.loc[train_mask], labeled.loc[valid_mask]
y_train = train_df[LABEL_COL].astype(int)
y_valid = valid_df[LABEL_COL].astype(int)

# LightGBM
X_train_lgb = prepare_lgb_frame(train_df, feature_cols, cat_cols)
X_valid_lgb = prepare_lgb_frame(valid_df, feature_cols, cat_cols)
n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
spw = n_neg / n_pos
params = dict(objective="binary", metric="None", scale_pos_weight=spw, seed=SEED,
              bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
              learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)
train_set = lgb.Dataset(X_train_lgb, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
valid_set = lgb.Dataset(X_valid_lgb, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
feval = make_pr_auc_feval(y_valid.values, seed=SEED)
t1 = time.time()
lgb_booster = lgb.train(params, train_set, num_boost_round=3000, valid_sets=[valid_set], feval=feval,
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)])
lgb_preds = lgb_booster.predict(X_valid_lgb, num_iteration=lgb_booster.best_iteration)
print(f"LightGBM alone: PR-AUC={average_precision_score(y_valid, lgb_preds):.4f}  ({time.time()-t1:.1f}s)")

# CatBoost (one-hot, no native cat_features -- see scripts/06 for why)
onehot = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
X_full = pd.concat([df[numeric_cols], onehot], axis=1)
X_train_cat, X_valid_cat = X_full.loc[train_mask.reindex(df.index, fill_value=False)], X_full.loc[valid_mask.reindex(df.index, fill_value=False)]
t1 = time.time()
cat_model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8, loss_function="Logloss",
                                eval_metric="PRAUC", scale_pos_weight=spw, random_seed=SEED,
                                early_stopping_rounds=100, verbose=False, thread_count=-1)
cat_model.fit(Pool(X_train_cat, y_train), eval_set=Pool(X_valid_cat, y_valid), use_best_model=True)
cat_preds = cat_model.predict_proba(X_valid_cat)[:, 1]
print(f"CatBoost alone: PR-AUC={average_precision_score(y_valid, cat_preds):.4f}  ({time.time()-t1:.1f}s)")

print("\n=== Blend weights (w * LightGBM + (1-w) * CatBoost) ===")
for w in [0.9, 0.8, 0.7, 0.6, 0.5]:
    blend = w * lgb_preds + (1 - w) * cat_preds
    score = average_precision_score(y_valid, blend)
    print(f"w={w:.1f}: PR-AUC={score:.4f}")
