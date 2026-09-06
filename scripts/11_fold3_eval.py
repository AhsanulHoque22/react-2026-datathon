"""Fast evaluator on the fold that actually predicts the leaderboard.

Fold 3 (Jul 2-15) scored 0.5029 while the leaderboard came in at 0.51055 --
the 4-fold MEAN (0.7117) is dominated by three easy-regime folds that do not
resemble the test window at all. So feature/model decisions get made here,
on fold 3, with fold 2 reported alongside only as a guard against wrecking
the easy regime.

~2-4 min per run vs ~20 min for full CV, so it's the right iteration loop.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, make_pr_auc_feval

FOLDS = {"fold2 (Jun18-Jul02)": ("2026-06-18", "2026-07-02"),
         "fold3 (Jul02-Jul15)": ("2026-07-02", "2026-07-15")}


def evaluate(df, feature_cols, cat_cols, tag=""):
    labeled = df[~df["is_test"]]
    out = {}
    for name, (start, end) in FOLDS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        tr = labeled.loc[labeled[TIME_COL] < start_ts]
        va = labeled.loc[(labeled[TIME_COL] >= start_ts) & (labeled[TIME_COL] < end_ts)]
        X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)
        X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)
        spw = (len(y_tr) - y_tr.sum()) / y_tr.sum()
        params = dict(objective="binary", metric="None", scale_pos_weight=spw, seed=SEED,
                      bagging_seed=SEED, feature_fraction_seed=SEED, verbosity=-1,
                      learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
                      bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
        t1 = time.time()
        bst = lgb.train(params, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                        feval=make_pr_auc_feval(y_va.values, seed=SEED),
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)])
        p = bst.predict(X_va, num_iteration=bst.best_iteration)
        out[name] = average_precision_score(y_va, p)
        print(f"  {tag:28s} {name}: PR-AUC={out[name]:.4f}  iter={bst.best_iteration}  ({time.time()-t1:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
    feature_cols, cat_cols = get_feature_columns(df)
    print(f"{len(feature_cols)} features")
    print("Reference: previous feature set scored fold3=0.5029, fold2=0.7810\n")
    evaluate(df, feature_cols, cat_cols, tag="current")
