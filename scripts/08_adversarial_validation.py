"""Adversarial validation + per-feature PSI: which features differ most
between train and test? PLAN.md always said "adversarial validation kept,
as a train/test time-drift sanity check" but it was never actually run as
a distinct diagnostic (only CV folds were compared). This targets a
different lever than anything tried so far: which SPECIFIC features are
unstable, not how much to weight recent rows or how to regularize.

Uses only non-label feature columns (is_test as the target here is not the
`fraud` label -- this is a standard, compliant diagnostic, not a banned
target-encoding scheme).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, SEED
from src.model import get_feature_columns, prepare_lgb_frame

df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
feature_cols, cat_cols = get_feature_columns(df)

X = prepare_lgb_frame(df, feature_cols, cat_cols)
y = df["is_test"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

params = dict(objective="binary", metric="auc", seed=SEED, verbosity=-1,
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=50)
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

t1 = time.time()
booster = lgb.train(params, train_set, num_boost_round=500, valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
auc = roc_auc_score(y_valid, preds)
print(f"Adversarial validation AUC (train vs test classifier): {auc:.4f}  ({time.time()-t1:.1f}s)")
print("(0.5 = indistinguishable, i.e. no drift; 1.0 = perfectly separable, i.e. severe drift)\n")

imp = pd.Series(booster.feature_importance(importance_type="gain"), index=feature_cols).sort_values(ascending=False)
print("Top 20 features distinguishing train from test (most unstable):")
print(imp.head(20))

# PSI per numeric feature: train vs test distribution
def psi(expected, actual, bins=10):
    expected, actual = expected.dropna(), actual.dropna()
    if len(expected) < 10 or len(actual) < 10:
        return np.nan
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return np.nan
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

numeric_cols = [c for c in feature_cols if c not in cat_cols]
train_df, test_df = df[~df["is_test"]], df[df["is_test"]]
psi_scores = {c: psi(train_df[c], test_df[c]) for c in numeric_cols}
psi_series = pd.Series(psi_scores).sort_values(ascending=False)
print("\nTop 20 features by PSI (train vs test; >0.2 = significant drift):")
print(psi_series.head(20))

psi_series.to_csv(PROCESSED_DIR / "feature_psi.csv")
imp.to_csv(PROCESSED_DIR / "adversarial_validation_importance.csv")
print(f"\nSaved feature_psi.csv and adversarial_validation_importance.csv")
