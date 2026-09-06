"""Targeted recency-weighting experiment on fold 3 (the fold that showed decay),
per PLAN.md's contingency: only build this because the drift plot actually
showed decay, not preemptively."""
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

train_end = train_df[TIME_COL].max()
days_before_end = (train_end - train_df[TIME_COL]).dt.total_seconds() / 86400.0


def run(weight=None, tag=""):
    params = dict(
        objective="binary", metric="None", scale_pos_weight=scale_pos_weight,
        seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )
    train_set = lgb.Dataset(X_train, label=y_train, weight=weight, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
    feval = make_pr_auc_feval(y_valid.values, seed=SEED)
    t1 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=3000, valid_sets=[valid_set], feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
    pr_auc = average_precision_score(y_valid, preds)
    print(f"{tag:30s} PR-AUC={pr_auc:.4f}  best_iter={booster.best_iteration}  ({time.time()-t1:.1f}s)")
    return pr_auc


print("=== Fold 3 recency-weighting experiment ===")
run(weight=None, tag="no weighting (baseline)")
for half_life in [14, 30, 60]:
    w = np.exp(-np.log(2) / half_life * days_before_end.values)
    run(weight=w, tag=f"exp decay half_life={half_life}d")

# recent-window-only: train on last 90 days before fold start
recent_cutoff = start_ts - pd.Timedelta(days=90)
recent_mask = train_df[TIME_COL] >= recent_cutoff
X_train_recent = X_train.loc[recent_mask.values]
y_train_recent = y_train.loc[recent_mask.values]
print(f"\nrecent-window (90d) train rows: {len(X_train_recent)} (vs full {len(X_train)})")
n_pos_r, n_neg_r = y_train_recent.sum(), len(y_train_recent) - y_train_recent.sum()
spw_r = n_neg_r / n_pos_r
params_r = dict(
    objective="binary", metric="None", scale_pos_weight=spw_r, seed=SEED,
    bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1, learning_rate=0.05,
    num_leaves=63, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
)
train_set_r = lgb.Dataset(X_train_recent, label=y_train_recent, categorical_feature=cat_cols, free_raw_data=False)
valid_set_r = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set_r, free_raw_data=False)
feval_r = make_pr_auc_feval(y_valid.values, seed=SEED)
t1 = time.time()
booster_r = lgb.train(
    params_r, train_set_r, num_boost_round=3000, valid_sets=[valid_set_r], feval=feval_r,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
)
preds_r = booster_r.predict(X_valid, num_iteration=booster_r.best_iteration)
pr_auc_r = average_precision_score(y_valid, preds_r)
print(f"{'recent-window-only (90d)':30s} PR-AUC={pr_auc_r:.4f}  best_iter={booster_r.best_iteration}  ({time.time()-t1:.1f}s)")
