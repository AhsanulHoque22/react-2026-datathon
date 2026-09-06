"""Shared feature-column selection + LightGBM training helpers."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.config import ID_COLS, LABEL_COL, TIME_COL, SEED, FEVAL_SUBSAMPLE_SIZE

CAT_COLS = ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
# log_amount_bdt is a helper column for the log-space z-scores, not a model
# feature: it is a monotonic transform of amount_bdt, so it yields identical
# tree splits and identical AP while consuming a feature_fraction slot.
EXCLUDE_COLS = set(ID_COLS) | {LABEL_COL, TIME_COL, "is_test", "log_amount_bdt"}


def get_feature_columns(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    return feature_cols, cat_cols


def prepare_lgb_frame(df: pd.DataFrame, feature_cols, cat_cols) -> pd.DataFrame:
    """Return a copy of df[feature_cols] with categoricals cast to pandas 'category'
    dtype (LightGBM native categorical handling) and raw ID / label columns absent
    (banned per PLAN.md — asserted, not just assumed)."""
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    # float32 halves memory (this box has 7GB RAM and the frame grew to 124
    # columns). LightGBM histogram-bins features anyway, so the reduced
    # precision does not change the splits it can find.
    for c in X.columns:
        if X[c].dtype == "float64":
            X[c] = X[c].astype("float32")
    for banned in ID_COLS:
        assert banned not in X.columns, f"banned raw ID column {banned} leaked into feature matrix"
    assert LABEL_COL not in X.columns
    return X


def make_pr_auc_feval(y_valid: np.ndarray, seed: int = SEED, subsample_size: int = FEVAL_SUBSAMPLE_SIZE):
    """Fixed (not resampled per round) stratified subsample of the validation
    fold, used to keep average_precision_score cheap across boosting rounds."""
    n = len(y_valid)
    if n <= subsample_size:
        idx = np.arange(n)
    else:
        rng = np.random.RandomState(seed)
        pos_idx = np.where(y_valid == 1)[0]
        neg_idx = np.where(y_valid == 0)[0]
        frac = subsample_size / n
        n_pos = max(1, int(round(len(pos_idx) * frac)))
        n_neg = subsample_size - n_pos
        pos_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_sample, neg_sample])

    def feval(preds, train_data):
        labels = train_data.get_label()
        score = average_precision_score(labels[idx], preds[idx])
        return "pr_auc", score, True

    return feval


def train_lgb(X_train, y_train, X_valid, y_valid, cat_cols, seed: int = SEED):
    # NO scale_pos_weight. PLAN.md originally prescribed neg/pos (~55x) for the
    # 1.76% imbalance, and every hyperparameter sweep held it fixed, so it went
    # untested until late. Measured on fold 3, less weighting is monotonically
    # better: spw=55 -> 0.5045, spw=10 -> 0.5069, spw=7.4 -> 0.5091,
    # spw=1 -> 0.5116 +/- 0.0020 (3 seeds). PR-AUC is a RANK metric; upweighting
    # positives distorts the ranking it is scored on. +0.0071 = 3.5x seed std.
    params = dict(
        objective="binary",
        metric="None",
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        verbosity=-1,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_data_in_leaf=50,
    )

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

    feval = make_pr_auc_feval(y_valid.values if hasattr(y_valid, "values") else y_valid, seed=seed)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        feval=feval,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)],
    )
    return booster


def precision_recall_at_k(y_true, y_score, k_frac):
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[top_k_idx] = 1
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return prec, rec
