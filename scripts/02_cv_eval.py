import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import PROCESSED_DIR, CV_FOLDS, LABEL_COL, TIME_COL, SEED
from src.model import get_feature_columns, prepare_lgb_frame, train_lgb, precision_recall_at_k

t0 = time.time()
df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
print(f"Loaded {df.shape}  ({time.time()-t0:.1f}s)")

feature_cols, cat_cols = get_feature_columns(df)
print(f"{len(feature_cols)} feature columns, {len(cat_cols)} categorical")

# CV only ever touches train.csv rows (test.csv has no label). Features already
# respect strict time-ordering globally, so no per-fold feature recomputation
# is needed -- only the train/valid row selection changes per fold.
labeled = df[~df["is_test"]]

results = []
for i, (start, end) in enumerate(CV_FOLDS):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    train_mask = labeled[TIME_COL] < start_ts
    valid_mask = (labeled[TIME_COL] >= start_ts) & (labeled[TIME_COL] < end_ts)

    train_df = labeled.loc[train_mask]
    valid_df = labeled.loc[valid_mask]

    X_train = prepare_lgb_frame(train_df, feature_cols, cat_cols)
    y_train = train_df[LABEL_COL].astype(int)
    X_valid = prepare_lgb_frame(valid_df, feature_cols, cat_cols)
    y_valid = valid_df[LABEL_COL].astype(int)

    t1 = time.time()
    booster = train_lgb(X_train, y_train, X_valid, y_valid, cat_cols, seed=SEED)
    preds = booster.predict(X_valid, num_iteration=booster.best_iteration)

    pr_auc = average_precision_score(y_valid, preds)

    has_hist = (valid_df["cust_history_count"] > 0).values
    pr_auc_hist = average_precision_score(y_valid[has_hist], preds[has_hist]) if has_hist.sum() > 0 else float("nan")
    pr_auc_cold = average_precision_score(y_valid[~has_hist], preds[~has_hist]) if (~has_hist).sum() > 0 else float("nan")

    prec_at_05pct, _ = precision_recall_at_k(y_valid.values, preds, 0.005)
    _, rec_at_1pct = precision_recall_at_k(y_valid.values, preds, 0.01)

    results.append(dict(
        fold=i, start=start, end=end,
        n_train=len(train_df), n_valid=len(valid_df), n_fraud_valid=int(y_valid.sum()),
        pr_auc=pr_auc, pr_auc_has_history=pr_auc_hist, pr_auc_cold_start=pr_auc_cold,
        cold_start_rate=float((~has_hist).mean()),
        precision_at_0_5pct=prec_at_05pct, recall_at_1pct=rec_at_1pct,
        best_iteration=booster.best_iteration,
        seconds=time.time() - t1,
    ))
    r = results[-1]
    print(f"Fold {i} [{start}->{end}] n_train={r['n_train']} n_valid={r['n_valid']} "
          f"fraud={r['n_fraud_valid']} PR-AUC={r['pr_auc']:.4f} "
          f"(history={r['pr_auc_has_history']:.4f}, cold={r['pr_auc_cold_start']:.4f}, "
          f"cold_rate={r['cold_start_rate']:.3f}) best_iter={r['best_iteration']} "
          f"({r['seconds']:.1f}s)")

res_df = pd.DataFrame(results)
scores = res_df["pr_auc"].values
print("\n=== Fold-to-fold spread ===")
print(f"mean={scores.mean():.4f} min={scores.min():.4f} max={scores.max():.4f} std={scores.std():.4f}")
print("\n=== Per-fold trend (time order) ===")
print(res_df[["fold", "start", "end", "pr_auc", "pr_auc_has_history", "pr_auc_cold_start"]].to_string(index=False))

res_df.to_csv(PROCESSED_DIR / "cv_results.csv", index=False)
print(f"\nSaved {PROCESSED_DIR / 'cv_results.csv'}")

trend = scores
decaying = trend[-1] < trend[0] - 0.02  # simple heuristic: notable drop from first to last fold
print(f"\nDrift check: first fold={trend[0]:.4f} last fold={trend[-1]:.4f} "
      f"-> {'POSSIBLE DECAY, consider recency-weighting' if decaying else 'no material decay observed'}")
