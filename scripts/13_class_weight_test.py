"""Never-tested axis: does scale_pos_weight actually help AP?

Every previous hyperparameter sweep varied leaves/learning-rate/regularization
but always kept scale_pos_weight at neg/pos (~56x). AP is a RANK metric --
aggressive positive-class weighting changes the fitted probabilities but can
distort the ranking, and several practitioners report it hurting AP even
while helping ROC-AUC. Cheap to check, and it is one of the few untested
axes left after seven within-noise feature/model experiments.
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

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
labeled = df[~df["is_test"]]

start_ts, end_ts = pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-15")
tr = labeled.loc[labeled[TIME_COL] < start_ts]
va = labeled.loc[(labeled[TIME_COL] >= start_ts) & (labeled[TIME_COL] < end_ts)]
X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)
X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)
full_spw = (len(y_tr) - y_tr.sum()) / y_tr.sum()
print(f"full scale_pos_weight would be {full_spw:.1f}\n")


def run(spw, tag, seed=SEED):
    params = dict(objective="binary", metric="None", seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed, verbosity=-1, learning_rate=0.05,
                  num_leaves=63, feature_fraction=0.85, bagging_fraction=0.85,
                  bagging_freq=1, min_data_in_leaf=50)
    if spw is not None:
        params["scale_pos_weight"] = spw
    ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
    t1 = time.time()
    bst = lgb.train(params, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                    feval=make_pr_auc_feval(y_va.values, seed=seed),
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)])
    p = bst.predict(X_va, num_iteration=bst.best_iteration)
    ap = average_precision_score(y_va, p)
    print(f"{tag:34s} PR-AUC={ap:.4f}  iter={bst.best_iteration}  ({time.time()-t1:.0f}s)", flush=True)
    return ap


print("=== scale_pos_weight sweep on fold 3 ===")
run(full_spw, f"current (spw={full_spw:.0f})")
run(1.0, "spw=1 (no weighting)")
run(np.sqrt(full_spw), f"spw=sqrt({full_spw:.0f})={np.sqrt(full_spw):.1f}")
run(10.0, "spw=10")

# Seed-noise floor: same config, 3 seeds. Tells us what effect size is even
# detectable on this fold -- seven straight experiments have landed inside
# what may simply be seed noise.
print("\n=== seed-noise floor (spw=1, 3 seeds) ===")
scores = [run(1.0, f"spw=1 seed={s}", seed=s) for s in (0, 1, 2)]
print(f"\nseed spread: mean={np.mean(scores):.4f} std={np.std(scores):.4f} "
      f"min={np.min(scores):.4f} max={np.max(scores):.4f}")
