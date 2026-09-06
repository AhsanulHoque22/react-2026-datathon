"""CatBoost as the opportunistic second model (PLAN.md: "CatBoost only
opportunistically if hours allow with a solid LightGBM already frozen").

Compliance note (deliberate, not an oversight): CatBoost's default handling
of `cat_features` uses ordered target statistics (CTR) -- an internal
per-category *target* encoding. That is exactly the mechanism PLAN.md's
Banned rule forbids ("any feature whose aggregation touches the `fraud`
column at a grain finer than the whole training set"), regardless of how
low the cardinality is. So this script never passes `cat_features` to
CatBoost -- the five low-cardinality categoricals are one-hot encoded by us
(a pure count-based transform, already banned-rule-compliant) instead,
keeping CatBoost's inputs identical in spirit to LightGBM's.

Same walk-forward folds as scripts/02_cv_eval.py, so per-fold PR-AUC is
directly comparable to the existing LightGBM CV results.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, CV_FOLDS, LABEL_COL, TIME_COL, SEED
from src.model import get_feature_columns, CAT_COLS

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)
numeric_cols = [c for c in feature_cols if c not in cat_cols]

# One-hot (not CatBoost's target-based CTR) for the low-cardinality categoricals.
onehot = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)
X_full = pd.concat([df[numeric_cols], onehot], axis=1)
onehot_cols = list(onehot.columns)
print(f"{len(numeric_cols)} numeric + {len(onehot_cols)} one-hot columns = {X_full.shape[1]} total")

labeled_mask = ~df["is_test"]

results = []
for i, (start, end) in enumerate(CV_FOLDS):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    train_mask = labeled_mask & (df[TIME_COL] < start_ts)
    valid_mask = labeled_mask & (df[TIME_COL] >= start_ts) & (df[TIME_COL] < end_ts)

    X_train, y_train = X_full.loc[train_mask], df.loc[train_mask, LABEL_COL].astype(int)
    X_valid, y_valid = X_full.loc[valid_mask], df.loc[valid_mask, LABEL_COL].astype(int)

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / n_pos

    t1 = time.time()
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=SEED,
        early_stopping_rounds=100,
        verbose=False,
        thread_count=-1,
    )
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    preds = model.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, preds)
    results.append(dict(fold=i, start=start, end=end, pr_auc=pr_auc,
                         best_iteration=model.get_best_iteration(), seconds=time.time() - t1))
    r = results[-1]
    print(f"Fold {i} [{start}->{end}] PR-AUC={r['pr_auc']:.4f} best_iter={r['best_iteration']} ({r['seconds']:.1f}s)", flush=True)

res_df = pd.DataFrame(results)
print("\n=== CatBoost fold-to-fold spread ===")
print(f"mean={res_df['pr_auc'].mean():.4f} min={res_df['pr_auc'].min():.4f} "
      f"max={res_df['pr_auc'].max():.4f} std={res_df['pr_auc'].std():.4f}")

print("\n=== Comparison vs LightGBM (from cv_results.csv) ===")
lgb_res = pd.read_csv(PROCESSED_DIR / "cv_results.csv")
for i in range(len(CV_FOLDS)):
    lgb_score = lgb_res.loc[lgb_res["fold"] == i, "pr_auc"].values[0]
    cat_score = res_df.loc[res_df["fold"] == i, "pr_auc"].values[0]
    print(f"Fold {i}: LightGBM={lgb_score:.4f}  CatBoost={cat_score:.4f}  diff={cat_score-lgb_score:+.4f}")

res_df.to_csv(PROCESSED_DIR / "cv_results_catboost.csv", index=False)
print(f"\nSaved {PROCESSED_DIR / 'cv_results_catboost.csv'}")
