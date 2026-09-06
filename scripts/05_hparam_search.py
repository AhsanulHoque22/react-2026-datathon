"""Targeted hyperparameter search on fold 3 (the drift fold) -- same
methodology as the recency-weighting experiment: cheap, targeted at the
hardest/most test-relevant fold rather than a full 4-fold grid search,
which would cost 4x the time for the same signal."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
labeled = df[~df["is_test"]]

start_ts, end_ts = pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-15")
train_df = labeled.loc[labeled[TIME_COL] < start_ts]
valid_df = labeled.loc[(labeled[TIME_COL] >= start_ts) & (labeled[TIME_COL] < end_ts)]

X_train = prepare_lgb_frame(train_df, feature_cols, cat_cols)
y_train = train_df[LABEL_COL].astype(int)
X_valid = prepare_lgb_frame(valid_df, feature_cols, cat_cols)
y_valid = valid_df[LABEL_COL].astype(int)

n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
scale_pos_weight = n_neg / n_pos

configs = [
    ("current (lr.05 leaves63 mdl50)", dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85)),
    ("more regularized (lr.03 leaves31 mdl100)", dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8)),
    ("less regularized (lr.05 leaves127 mdl30)", dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9)),
    ("slow+regularized (lr.02 leaves63 mdl100)", dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8)),
    ("heavy L1L2 (lr.05 leaves63 mdl50 + reg)", dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85, lambda_l1=1.0, lambda_l2=1.0)),
]

for tag, extra in configs:
    params = dict(
        objective="binary", metric="None", scale_pos_weight=scale_pos_weight,
        seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
        bagging_freq=1,
    )
    params.update(extra)
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
    feval = make_pr_auc_feval(y_valid.values, seed=SEED)
    t1 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=3000, valid_sets=[valid_set], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
    pr_auc = average_precision_score(y_valid, preds)
    print(f"{tag:45s} PR-AUC={pr_auc:.4f}  best_iter={booster.best_iteration}  ({time.time()-t1:.1f}s)", flush=True)
