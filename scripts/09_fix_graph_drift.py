"""Adversarial validation found the graph component-size features are the
overwhelming driver of train/test separability (AUC=1.0 driven almost
entirely by 2 features) and the top PSI offenders -- likely because
Union-Find component sizes only grow over time, so test-period values sit
at a scale never seen in training. Compare on the hard fold: current
(unnormalized) vs dropped vs normalized (component size / total nodes seen
so far, a scale-invariant fraction)."""
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
y_train = train_df[LABEL_COL].astype(int)
y_valid = valid_df[LABEL_COL].astype(int)

graph_cols = [c for c in feature_cols if "component_size_prior" in c]
print(f"Graph columns: {graph_cols}")

# Normalized version: divide each component size by the running total number
# of distinct nodes unioned so far in that graph (a monotonic denominator
# too, but the RATIO is scale-invariant, unlike the raw count).
def add_normalized_graph_features(frame):
    frame = frame.copy()
    for col in graph_cols:
        # running max seen so far as a cheap scale-invariant denominator
        running_max = frame[col].expanding().max().shift(1).bfill()
        frame[col + "_norm"] = frame[col] / running_max.replace(0, np.nan)
    return frame

df_norm = add_normalized_graph_features(df)


def run(cols, tag):
    X_train = prepare_lgb_frame(train_df, cols, cat_cols)
    X_valid = prepare_lgb_frame(valid_df, cols, cat_cols)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    spw = n_neg / n_pos
    params = dict(objective="binary", metric="None", scale_pos_weight=spw, seed=SEED,
                  bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
    feval = make_pr_auc_feval(y_valid.values, seed=SEED)
    t1 = time.time()
    booster = lgb.train(params, train_set, num_boost_round=3000, valid_sets=[valid_set], feval=feval,
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)])
    preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
    score = average_precision_score(y_valid, preds)
    print(f"{tag:35s} PR-AUC={score:.4f}  best_iter={booster.best_iteration}  ({time.time()-t1:.1f}s)", flush=True)
    return score


cols_current = feature_cols
cols_dropped = [c for c in feature_cols if c not in graph_cols]
cols_normalized = [c for c in feature_cols if c not in graph_cols] + [c + "_norm" for c in graph_cols]

print("=== Fold 3 (hard fold) comparison ===")
run(cols_current, "current (raw component size)")
run(cols_dropped, "dropped (no graph features)")

# re-slice train/valid from the normalized frame
train_df_n = df_norm[~df_norm["is_test"]].loc[df_norm[TIME_COL] < start_ts]
valid_df_n = df_norm[~df_norm["is_test"]].loc[(df_norm[TIME_COL] >= start_ts) & (df_norm[TIME_COL] < end_ts)]
train_df, valid_df = train_df_n, valid_df_n
run(cols_normalized, "normalized (fraction of running max)")
