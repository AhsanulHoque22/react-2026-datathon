"""Assemble src/ into one self-contained Kaggle kernel script.

The repo is private, so a kernel cannot clone it -- the code has to be
inlined. Generating that file from src/ (rather than hand-maintaining a
copy) keeps the Kaggle run and the local source from drifting apart as we
keep iterating, which matters because this same notebook is the
reproducibility artifact the rulebook requires from top-15 teams.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle_kernel"
KAGGLE_DIR.mkdir(exist_ok=True)

MODULES = ["config.py", "data.py", "features.py", "model.py"]

HEADER = '''"""REACT 2026 Datathon -- full pipeline, generated from src/ by
scripts/build_kaggle_kernel.py. Do not edit here; edit src/ and regenerate.

Leakage-safe behavioural feature engineering + LightGBM, trained on a
chronological split. Every engineered feature for a row at time t uses only
rows strictly before t (ties broken by transaction_id). No raw entity IDs
reach the model, and nothing touches the fraud label except the label itself.
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

# Attaching the competition via the API mounts it under input/competitions/,
# while the notebook UI mounts it directly under input/. Accept either.
_CANDIDATES = [
    Path("/kaggle/input/competitions/react-2026-datathon"),
    Path("/kaggle/input/react-2026-datathon"),
]
RAW_DIR = next((p for p in _CANDIDATES if (p / "train.csv").exists()), _CANDIDATES[0])
OUT_DIR = Path("/kaggle/working")
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"
print(f"Using data dir: {RAW_DIR}")
'''

MAIN = '''

# ----------------------------- pipeline ------------------------------------
def main():
    t0 = time.time()
    print("Loading raw data...", flush=True)
    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    print(f"train={train.shape} test={test.shape} ({time.time()-t0:.1f}s)", flush=True)

    df = build_combined_frame(train, test)
    assert_frame_sane(df)
    del train, test
    gc.collect()

    print("Building features...", flush=True)
    df = build_features(df)
    leakage_assertions(df)
    print(f"featurized={df.shape} ({time.time()-t0:.1f}s)", flush=True)

    feature_cols, cat_cols = get_feature_columns(df)
    frame_time_sorted = bool(df[TIME_COL].is_monotonic_increasing)
    labeled = df[~df["is_test"]]
    test_df = df[df["is_test"]]

    # ---- walk-forward CV: report per fold, and the fold that matters ----
    print("\\n=== Walk-forward CV ===", flush=True)
    fold_scores = {}
    for i, (start, end) in enumerate(CV_FOLDS):
        s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
        tr = labeled.loc[labeled[TIME_COL] < s_ts]
        va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]
        X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)
        X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)
        bst = train_lgb(X_tr, y_tr, X_va, y_va, cat_cols)
        p = bst.predict(X_va, num_iteration=bst.best_iteration)
        ap = average_precision_score(y_va, p)
        has_hist = (va["cust_history_count"] > 0).values
        ap_hist = average_precision_score(y_va[has_hist], p[has_hist]) if has_hist.sum() else float("nan")
        ap_cold = average_precision_score(y_va[~has_hist], p[~has_hist]) if (~has_hist).sum() else float("nan")
        pk, _ = precision_recall_at_k(y_va.values, p, 0.005)
        _, rk = precision_recall_at_k(y_va.values, p, 0.01)
        fold_scores[i] = ap
        print(f"fold{i} [{start}->{end}] PR-AUC={ap:.4f} (hist={ap_hist:.4f} cold={ap_cold:.4f}) "
              f"P@0.5%={pk:.3f} R@1%={rk:.3f} iter={bst.best_iteration}", flush=True)
        del X_tr, X_va, bst
        gc.collect()

    vals = np.array(list(fold_scores.values()))
    print(f"mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f} std={vals.std():.4f}")
    print(f"LAST fold (the one that tracks the leaderboard) = {vals[-1]:.4f}", flush=True)

    # ---- iteration count from a held-out tail, then seed-averaged refit ----
    holdout_start = pd.Timestamp("2026-07-01")
    fit_df = labeled.loc[labeled[TIME_COL] < holdout_start]
    hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start]
    X_fit, y_fit = prepare_lgb_frame(fit_df, feature_cols, cat_cols), fit_df[LABEL_COL].astype(int)
    X_hold, y_hold = prepare_lgb_frame(hold_df, feature_cols, cat_cols), hold_df[LABEL_COL].astype(int)
    probe = train_lgb(X_fit, y_fit, X_hold, y_hold, cat_cols)
    probe_ap = average_precision_score(y_hold, probe.predict(X_hold, num_iteration=probe.best_iteration))
    final_rounds = int(round(probe.best_iteration * 1.1))
    print(f"\\nHeld-out PR-AUC={probe_ap:.4f} best_iter={probe.best_iteration} -> rounds={final_rounds}", flush=True)
    del X_fit, X_hold, probe
    gc.collect()

    X_full = prepare_lgb_frame(labeled, feature_cols, cat_cols)
    y_full = labeled[LABEL_COL].astype(int)
    X_test = prepare_lgb_frame(test_df, feature_cols, cat_cols)
    test_ids = test_df["transaction_id"].values
    del df, labeled, test_df
    gc.collect()

    print(f"Refitting on full train, averaged over {len(SEEDS)} seeds...", flush=True)
    ds = lgb.Dataset(X_full, label=y_full, categorical_feature=cat_cols, free_raw_data=False)
    seed_preds = []
    last_bst = None
    for s in SEEDS:
        params = dict(FINAL_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s)
        bst = lgb.train(params, ds, num_boost_round=final_rounds)
        seed_preds.append(bst.predict(X_test))
        last_bst = bst
        print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    preds = np.mean(seed_preds, axis=0)

    # ---- pre-submission assertions (PLAN.md checklist) ----
    assert frame_time_sorted, "frame not time-sorted"
    assert LABEL_COL not in X_test.columns
    for c in ID_COLS:
        assert c not in X_test.columns, f"banned raw ID column {c} in feature matrix"
    assert np.all((preds >= 0) & (preds <= 1)), "predictions outside [0,1]"
    assert not np.isnan(preds).any(), "NaN predictions"

    sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    sub = pd.DataFrame({"transaction_id": test_ids, "fraud": preds})
    sub = sub.set_index("transaction_id").loc[sample["transaction_id"]].reset_index()
    assert list(sub["transaction_id"]) == list(sample["transaction_id"]), "row order mismatch"
    assert sub.shape == sample.shape
    assert sub["fraud"].between(0, 1).all()
    assert (sub["fraud"] != sub["fraud"].round().astype(int)).any(), "looks thresholded to 0/1"

    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\\nAll assertions passed. Wrote submission.csv {sub.shape} (total {time.time()-t0:.1f}s)")
    print(sub["fraud"].describe())

    imp = pd.Series(last_bst.feature_importance(importance_type="gain"),
                    index=feature_cols).sort_values(ascending=False)
    print("\\nTop 20 features by gain:")
    print(imp.head(20).to_string())
    imp.to_csv(OUT_DIR / "feature_importance.csv")


if __name__ == "__main__":
    main()
'''


def strip_module(text: str) -> str:
    """Drop imports and module docstring -- the header supplies them once."""
    text = re.sub(r'^""".*?"""\n', "", text, flags=re.S)
    lines = []
    for ln in text.splitlines():
        if re.match(r"^(import |from )", ln) and "src." not in ln:
            continue
        if ln.startswith("from src.") or ln.startswith("import src"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip("\n")


SWEEP_MAIN = '''

# ------------------------- hyperparameter sweep ----------------------------
# The earlier sweep ran with scale_pos_weight=55, which we since measured to
# distort this rank metric -- so every conclusion it reached is void. Under
# spw=1 the model converges in ~104 rounds instead of ~764, a different
# regime entirely. Each config is run over 3 seeds because the measured
# seed-noise std is 0.0020 and single runs cannot resolve that.
def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    df = build_combined_frame(train, test)
    assert_frame_sane(df)
    del train, test
    gc.collect()
    df = build_features(df)
    leakage_assertions(df)
    feature_cols, cat_cols = get_feature_columns(df)
    print(f"featurized={df.shape} ({time.time()-t0:.0f}s)", flush=True)

    labeled = df[~df["is_test"]]
    del df
    gc.collect()

    # Evaluate on the held-out tail (Jul 1-15) -- the window that tracks the
    # leaderboard -- and on fold 2 as a guard against wrecking the easy regime.
    WINDOWS = {"tail(Jul01-15)": ("2026-07-01", "2026-07-16"),
               "fold2(Jun18-Jul02)": ("2026-06-18", "2026-07-02")}
    splits = {}
    for wname, (s, e) in WINDOWS.items():
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        tr = labeled.loc[labeled[TIME_COL] < s_ts]
        va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]
        splits[wname] = (prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int),
                         prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int))
        print(f"{wname}: train={len(tr)} valid={len(va)} frauds={int(va[LABEL_COL].sum())}", flush=True)

    # Stage 2: lr/leaves are now fixed at the stage-1 winner (0.02/127).
    # Sweep the knobs stage 1 held constant -- these control how much each
    # tree can memorise, which matters more now that we train ~940 rounds
    # instead of ~140.
    CONFIGS = []
    for mdl in (20, 50, 200, 500):
        for ff in (0.5, 0.7, 0.85):
            CONFIGS.append(dict(learning_rate=0.02, num_leaves=127,
                                min_data_in_leaf=mdl, feature_fraction=ff))

    results = []
    for cfg in CONFIGS:
        row = {**cfg}
        for wname, (X_tr, y_tr, X_va, y_va) in splits.items():
            aps, iters = [], []
            for seed in (0, 1, 2):
                params = dict(objective="binary", metric="None", verbosity=-1,
                              bagging_fraction=0.85, bagging_freq=1, seed=seed,
                              bagging_seed=seed, feature_fraction_seed=seed, **cfg)
                ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
                ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols,
                                    reference=ds_tr, free_raw_data=False)
                bst = lgb.train(params, ds_tr, num_boost_round=6000, valid_sets=[ds_va],
                                feval=make_pr_auc_feval(y_va.values, seed=seed),
                                callbacks=[lgb.early_stopping(200, verbose=False),
                                           lgb.log_evaluation(period=0)])
                aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))
                iters.append(bst.best_iteration)
            row[f"{wname}_mean"] = float(np.mean(aps))
            row[f"{wname}_std"] = float(np.std(aps))
            row[f"{wname}_iter"] = int(np.mean(iters))
            print(f"mdl={cfg['min_data_in_leaf']:<4} ff={cfg['feature_fraction']:<5} {wname}: "
                  f"{np.mean(aps):.4f} +/- {np.std(aps):.4f} (iter~{int(np.mean(iters))}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        results.append(row)

    res = pd.DataFrame(results).sort_values("tail(Jul01-15)_mean", ascending=False)
    print("\\n=== SWEEP RESULTS (sorted by tail window, the leaderboard proxy) ===")
    print(res.to_string(index=False))
    res.to_csv(OUT_DIR / "sweep_results.csv", index=False)


if __name__ == "__main__":
    main()
'''


EXP2_MAIN = '''

# --------- recency asymmetry + ensemble blending experiments ---------------
def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])
    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])
    df = build_combined_frame(train, test)
    assert_frame_sane(df)
    del train, test
    gc.collect()
    df = build_features(df)
    feature_cols, cat_cols = get_feature_columns(df)
    labeled = df[~df["is_test"]]
    del df
    gc.collect()
    print(f"featurized, {len(feature_cols)} features ({time.time()-t0:.0f}s)", flush=True)

    # EXPERIMENT A -- recency weighting, tested FAIRLY this time.
    # The earlier test validated on Jul 2-15 while training on data ending
    # Jul 2, i.e. entirely BEFORE the regime change -- there was no
    # post-change data to weight toward, so it could not have helped. Here
    # the split is Jul 8: training now contains ~1 week of post-change data,
    # which mirrors the real submission (trains through Jul 15, predicts
    # Jul 16+). This is the only configuration where recency weighting has
    # a mechanism to work.
    split = pd.Timestamp("2026-07-08")
    tr = labeled.loc[labeled[TIME_COL] < split]
    va = labeled.loc[labeled[TIME_COL] >= split]
    X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)
    X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)
    print(f"\\nEXP A split Jul08: train={len(tr)} valid={len(va)} "
          f"frauds={int(y_va.sum())}", flush=True)

    days_before = (tr[TIME_COL].max() - tr[TIME_COL]).dt.total_seconds().values / 86400.0

    def run_weighted(half_life, tag):
        aps = []
        for seed in (0, 1, 2):
            w = None if half_life is None else np.exp(-np.log(2) / half_life * days_before)
            params = dict(objective="binary", metric="None", verbosity=-1,
                          learning_rate=0.05, num_leaves=63, feature_fraction=0.85,
                          bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
                          seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=w, categorical_feature=cat_cols, free_raw_data=False)
            ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
            bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],
                            feval=make_pr_auc_feval(y_va.values, seed=seed),
                            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])
            aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))
        print(f"  {tag:32s} {np.mean(aps):.4f} +/- {np.std(aps):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        return float(np.mean(aps))

    print("EXPERIMENT A: recency weighting with post-change data in train")
    run_weighted(None, "no weighting")
    for hl in (7, 14, 30, 60):
        run_weighted(hl, f"exp decay half-life={hl}d")

    # EXPERIMENT B -- ensemble blending. Research says rank-averaging beats
    # probability-averaging for rank metrics because it ignores calibration
    # differences between models. Worth checking directly rather than assuming.
    print("\\nEXPERIMENT B: ensemble blending (rank vs probability averaging)")
    CONFIGS = [dict(learning_rate=0.05, num_leaves=63, feature_fraction=0.85),
               dict(learning_rate=0.02, num_leaves=127, feature_fraction=0.7),
               dict(learning_rate=0.1, num_leaves=31, feature_fraction=0.95)]
    preds, singles = [], []
    for i, cfg in enumerate(CONFIGS):
        params = dict(objective="binary", metric="None", verbosity=-1,
                      bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
                      seed=i, bagging_seed=i, feature_fraction_seed=i, **cfg)
        ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)
        bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],
                        feval=make_pr_auc_feval(y_va.values, seed=i),
                        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])
        p = bst.predict(X_va, num_iteration=bst.best_iteration)
        preds.append(p)
        ap = average_precision_score(y_va, p)
        singles.append(ap)
        print(f"  model {i} {cfg} -> {ap:.4f}", flush=True)

    P = np.vstack(preds)
    prob_avg = P.mean(axis=0)
    ranks = np.vstack([pd.Series(p).rank(pct=True).values for p in preds])
    rank_avg = ranks.mean(axis=0)
    print(f"  best single         : {max(singles):.4f}")
    print(f"  probability average : {average_precision_score(y_va, prob_avg):.4f}")
    print(f"  RANK average        : {average_precision_score(y_va, rank_avg):.4f}", flush=True)


if __name__ == "__main__":
    main()
'''


EXP3_MAIN = '\n\n# ------------- ablation: do the new self-relative features help? -----------\ndef main():\n    t0 = time.time()\n    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])\n    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])\n    df = build_combined_frame(train, test)\n    assert_frame_sane(df)\n    del train, test\n    gc.collect()\n    df = build_features(df)\n    leakage_assertions(df)\n    feature_cols, cat_cols = get_feature_columns(df)\n    labeled = df[~df["is_test"]]\n    del df\n    gc.collect()\n    print(f"featurized, {len(feature_cols)} features ({time.time()-t0:.0f}s)", flush=True)\n\n    NEW = [c for c in feature_cols if any(k in c for k in\n           ("_gap_accel", "_velocity_ratio_", "_amt_vs_prior_max",\n            "_is_record_amt", "_hour_bucket_share", "_new_hour_bucket"))]\n    print(f"{len(NEW)} new self-relative features: {NEW}", flush=True)\n\n    # Two windows: the tail is the leaderboard proxy, fold2 guards the easy regime.\n    WINDOWS = {"tail(Jul01-15)": ("2026-07-01", "2026-07-16"),\n               "fold2(Jun18-Jul02)": ("2026-06-18", "2026-07-02")}\n\n    def evaluate(cols, tag):\n        for wname, (s_, e_) in WINDOWS.items():\n            s_ts, e_ts = pd.Timestamp(s_), pd.Timestamp(e_)\n            tr = labeled.loc[labeled[TIME_COL] < s_ts]\n            va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]\n            cc = [c for c in cat_cols if c in cols]\n            X_tr, y_tr = prepare_lgb_frame(tr, cols, cc), tr[LABEL_COL].astype(int)\n            X_va, y_va = prepare_lgb_frame(va, cols, cc), va[LABEL_COL].astype(int)\n            aps = []\n            for seed in (0, 1, 2):\n                params = dict(objective="binary", metric="None", verbosity=-1,\n                              learning_rate=0.02, num_leaves=127, feature_fraction=0.85,\n                              bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,\n                              seed=seed, bagging_seed=seed, feature_fraction_seed=seed)\n                ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cc, free_raw_data=False)\n                ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cc, reference=ds_tr, free_raw_data=False)\n                bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],\n                                feval=make_pr_auc_feval(y_va.values, seed=seed),\n                                callbacks=[lgb.early_stopping(200, verbose=False),\n                                           lgb.log_evaluation(period=0)])\n                aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))\n            print(f"  {tag:22s} {wname:20s} {np.mean(aps):.4f} +/- {np.std(aps):.4f}  [{time.time()-t0:.0f}s]", flush=True)\n            del X_tr, X_va\n            gc.collect()\n\n    print("\\nABLATION (3 seeds each; seed-noise std is ~0.0020)")\n    evaluate([c for c in feature_cols if c not in NEW], "WITHOUT new")\n    evaluate(feature_cols, "WITH new")\n\n\nif __name__ == "__main__":\n    main()\n'


FINAL_MAIN = '\n\n# =================== FINAL: aggregate everything, score locally =============\n# Combines every validated fix (no class weighting, swept lr=0.02/leaves=127,\n# 5-seed averaging, stationary encodings, self-relative features) with the new\n# Phase 3 recent-window family, and scores across MULTIPLE recent windows\n# rather than repeatedly against fold 3\'s 804 positives.\n#\n# Acceptance rule (from FOLD3_IMPROVEMENT_PLAN.md): a candidate wins only if\n# the recent-window mean improves by >=0.004, at least two windows improve,\n# no window drops more than 0.003, and it holds across 3 seeds.\ndef main():\n    t0 = time.time()\n    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])\n    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])\n    df = build_combined_frame(train, test)\n    assert_frame_sane(df)\n    del train, test\n    gc.collect()\n    df = build_features(df)\n    leakage_assertions(df)\n    feature_cols, cat_cols = get_feature_columns(df)\n    print(f"featurized={df.shape}, {len(feature_cols)} features ({time.time()-t0:.0f}s)", flush=True)\n\n    NEW = [c for c in feature_cols if any(k in c for k in (\n        "_amt_vs_recent_mean_", "_amt_vs_recent_max_", "_fresh_",\n        "_cnt_5m", "_cnt_15m", "_cnt_30m", "_cnt_3h", "_cnt_12h",\n        "_amtsum_5m", "_amtsum_15m", "_amtsum_30m", "_amtsum_3h", "_amtsum_12h",\n        "_gap_accel", "_velocity_ratio_", "_amt_vs_prior_max", "_is_record_amt",\n        "_hour_bucket_share", "_new_hour_bucket"))]\n    GRAPH = [c for c in feature_cols if "component_size_prior" in c]\n    OLD = [c for c in feature_cols if c not in NEW]\n    print(f"{len(NEW)} new / {len(OLD)} old / {len(GRAPH)} graph", flush=True)\n\n    labeled = df[~df["is_test"]]\n    test_df = df[df["is_test"]]\n    frame_sorted = bool(df[TIME_COL].is_monotonic_increasing)\n    del df\n    gc.collect()\n\n    WINDOWS = {\n        "Jun25-Jul02": ("2026-06-25", "2026-07-02"),\n        "Jul02-Jul08": ("2026-07-02", "2026-07-08"),\n        "Jul08-Jul16": ("2026-07-08", "2026-07-16"),\n        "fold2-guard": ("2026-06-18", "2026-07-02"),\n    }\n\n    ARMS = {\n        "A_prev_best(lr.05/63,old feats)": (OLD, dict(learning_rate=0.05, num_leaves=63)),\n        "B_new_feats(lr.02/127)":          (feature_cols, dict(learning_rate=0.02, num_leaves=127)),\n        "C_B_minus_graph":                 ([c for c in feature_cols if c not in GRAPH],\n                                            dict(learning_rate=0.02, num_leaves=127)),\n    }\n\n    results = {}\n    for arm, (cols, cfg) in ARMS.items():\n        cc = [c for c in cat_cols if c in cols]\n        results[arm] = {}\n        for wname, (s_, e_) in WINDOWS.items():\n            s_ts, e_ts = pd.Timestamp(s_), pd.Timestamp(e_)\n            tr = labeled.loc[labeled[TIME_COL] < s_ts]\n            va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]\n            X_tr, y_tr = prepare_lgb_frame(tr, cols, cc), tr[LABEL_COL].astype(int)\n            X_va, y_va = prepare_lgb_frame(va, cols, cc), va[LABEL_COL].astype(int)\n            aps = []\n            for seed in (0, 1, 2):\n                params = dict(objective="binary", metric="None", verbosity=-1,\n                              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,\n                              min_data_in_leaf=50, seed=seed, bagging_seed=seed,\n                              feature_fraction_seed=seed, **cfg)\n                ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cc, free_raw_data=False)\n                ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cc, reference=ds_tr, free_raw_data=False)\n                bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],\n                                feval=make_pr_auc_feval(y_va.values, seed=seed),\n                                callbacks=[lgb.early_stopping(200, verbose=False),\n                                           lgb.log_evaluation(period=0)])\n                aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))\n            results[arm][wname] = (float(np.mean(aps)), float(np.std(aps)))\n            print(f"  {arm:34s} {wname:12s} n_fraud={int(y_va.sum()):4d} "\n                  f"{np.mean(aps):.4f} +/- {np.std(aps):.4f}  [{time.time()-t0:.0f}s]", flush=True)\n            del X_tr, X_va\n            gc.collect()\n\n    print("\\n================= FINAL LOCAL SCORES =================")\n    recent = [w for w in WINDOWS if w != "fold2-guard"]\n    hdr = f"{\'arm\':36s}" + "".join(f"{w:>16s}" for w in WINDOWS) + f"{\'RECENT MEAN\':>14s}"\n    print(hdr)\n    for arm in ARMS:\n        row = f"{arm:36s}"\n        for w in WINDOWS:\n            m, sd = results[arm][w]\n            row += f"{m:>10.4f}+-{sd:.3f}"\n        rm = np.mean([results[arm][w][0] for w in recent])\n        row += f"{rm:>14.4f}"\n        print(row)\n\n    base = np.mean([results["A_prev_best(lr.05/63,old feats)"][w][0] for w in recent])\n    for arm in ["B_new_feats(lr.02/127)", "C_B_minus_graph"]:\n        rm = np.mean([results[arm][w][0] for w in recent])\n        improved = sum(results[arm][w][0] > results["A_prev_best(lr.05/63,old feats)"][w][0] for w in recent)\n        worst = min(results[arm][w][0] - results["A_prev_best(lr.05/63,old feats)"][w][0] for w in recent)\n        verdict = ("ACCEPT" if (rm - base) >= 0.004 and improved >= 2 and worst >= -0.003\n                   else "reject (does not clear the acceptance rule)")\n        print(f"\\n{arm}: recent-mean {rm:.4f} vs {base:.4f} (delta {rm-base:+.4f}), "\n              f"{improved}/{len(recent)} windows improved, worst delta {worst:+.4f} -> {verdict}")\n\n    # Train the best-by-recent-mean arm on all of train and write a candidate.\n    best_arm = max(ARMS, key=lambda a: np.mean([results[a][w][0] for w in recent]))\n    print(f"\\nBest arm: {best_arm} -- training final candidate", flush=True)\n    cols, cfg = ARMS[best_arm]\n    cc = [c for c in cat_cols if c in cols]\n    hold_start = pd.Timestamp("2026-07-01")\n    fit = labeled.loc[labeled[TIME_COL] < hold_start]\n    hold = labeled.loc[labeled[TIME_COL] >= hold_start]\n    X_f, y_f = prepare_lgb_frame(fit, cols, cc), fit[LABEL_COL].astype(int)\n    X_h, y_h = prepare_lgb_frame(hold, cols, cc), hold[LABEL_COL].astype(int)\n    base_params = dict(objective="binary", metric="None", verbosity=-1,\n                       feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,\n                       min_data_in_leaf=50, **cfg)\n    probe = lgb.train(dict(base_params, seed=0, bagging_seed=0, feature_fraction_seed=0),\n                      lgb.Dataset(X_f, label=y_f, categorical_feature=cc, free_raw_data=False),\n                      num_boost_round=4000,\n                      valid_sets=[lgb.Dataset(X_h, label=y_h, categorical_feature=cc, free_raw_data=False)],\n                      feval=make_pr_auc_feval(y_h.values, seed=0),\n                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])\n    rounds = int(round(probe.best_iteration * 1.1))\n    print(f"held-out(Jul01-15)={average_precision_score(y_h, probe.predict(X_h, num_iteration=probe.best_iteration)):.4f} "\n          f"rounds={rounds}", flush=True)\n    del X_f, X_h, probe\n    gc.collect()\n\n    X_full, y_full = prepare_lgb_frame(labeled, cols, cc), labeled[LABEL_COL].astype(int)\n    X_test = prepare_lgb_frame(test_df, cols, cc)\n    ids = test_df["transaction_id"].values\n    ds = lgb.Dataset(X_full, label=y_full, categorical_feature=cc, free_raw_data=False)\n    preds = np.mean([lgb.train(dict(base_params, seed=s, bagging_seed=s, feature_fraction_seed=s),\n                               ds, num_boost_round=rounds).predict(X_test) for s in (0, 1, 2, 3, 4)], axis=0)\n\n    assert frame_sorted and LABEL_COL not in X_test.columns\n    for c in ID_COLS:\n        assert c not in X_test.columns\n    assert np.all((preds >= 0) & (preds <= 1)) and not np.isnan(preds).any()\n    sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)\n    sub = pd.DataFrame({"transaction_id": ids, "fraud": preds})\n    sub = sub.set_index("transaction_id").loc[sample["transaction_id"]].reset_index()\n    assert list(sub["transaction_id"]) == list(sample["transaction_id"])\n    assert (sub["fraud"] != sub["fraud"].round().astype(int)).any()\n    sub.to_csv(OUT_DIR / "submission.csv", index=False)\n    print(f"\\nWrote candidate submission.csv from {best_arm} ({time.time()-t0:.0f}s)")\n\n\nif __name__ == "__main__":\n    main()\n'


HORIZON_MAIN = '\n\n# ============ how does performance decay with forecast horizon? ============\n# Our probe tunes rounds on Jul 1-15: a 1-to-15-day-ahead prediction. The real\n# test is Jul 16 - Sep 15, up to 62 days past the end of training. If accuracy\n# and the optimal number of rounds shift with horizon, the final model is\n# tuned for the wrong problem.\n#\n# Design: hold the validation window FIXED at Jul 1-15 and walk the training\n# cutoff backwards, so the only thing changing is the gap between the end of\n# training and the prediction window.\ndef main():\n    t0 = time.time()\n    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])\n    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])\n    df = build_combined_frame(train, test)\n    assert_frame_sane(df)\n    del train, test\n    gc.collect()\n    df = build_features(df)\n    feature_cols, cat_cols = get_feature_columns(df)\n    labeled = df[~df["is_test"]]\n    del df\n    gc.collect()\n    print(f"featurized, {len(feature_cols)} features ({time.time()-t0:.0f}s)", flush=True)\n\n    va = labeled.loc[(labeled[TIME_COL] >= pd.Timestamp("2026-07-01")) &\n                     (labeled[TIME_COL] < pd.Timestamp("2026-07-16"))]\n    X_va, y_va = prepare_lgb_frame(va, feature_cols, cat_cols), va[LABEL_COL].astype(int)\n    print(f"fixed validation Jul01-16: {len(va)} rows, {int(y_va.sum())} frauds\\n", flush=True)\n\n    # gap = days between end of training and the start of the validation window\n    CUTOFFS = [("2026-07-01", 0), ("2026-06-24", 7), ("2026-06-17", 14),\n               ("2026-06-01", 30), ("2026-05-17", 45), ("2026-05-02", 60)]\n\n    print(f"{\'train ends\':<12}{\'gap(d)\':>7}{\'PR-AUC\':>18}{\'best_iter\':>12}")\n    for cutoff, gap in CUTOFFS:\n        tr = labeled.loc[labeled[TIME_COL] < pd.Timestamp(cutoff)]\n        X_tr, y_tr = prepare_lgb_frame(tr, feature_cols, cat_cols), tr[LABEL_COL].astype(int)\n        aps, iters = [], []\n        for seed in (0, 1, 2):\n            params = dict(objective="binary", metric="None", verbosity=-1,\n                          learning_rate=0.02, num_leaves=127, feature_fraction=0.85,\n                          bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,\n                          seed=seed, bagging_seed=seed, feature_fraction_seed=seed)\n            ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)\n            ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=ds_tr, free_raw_data=False)\n            bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],\n                            feval=make_pr_auc_feval(y_va.values, seed=seed),\n                            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])\n            aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))\n            iters.append(bst.best_iteration)\n        print(f"{cutoff:<12}{gap:>7}{np.mean(aps):>11.4f}+-{np.std(aps):.4f}{int(np.mean(iters)):>12}"\n              f"   [{time.time()-t0:.0f}s]", flush=True)\n        del X_tr\n        gc.collect()\n\n    print("\\nIf PR-AUC falls sharply with gap, late test rows (up to 62 days out)")\n    print("score far below our Jul01-15 probe, and the probe overstates the model.")\n    print("If best_iter also falls, the final refit is using too many rounds.")\n\n\nif __name__ == "__main__":\n    main()\n'

DRIFT_MAIN = '\n\n# ============ does time-local percentile normalisation beat raw values? =====\n# Diagnosis this tests: the test set (Jul 16 - Sep 15) lies entirely after the\n# July regime change, and we hold ~1-2 weeks of post-change training data out\n# of 6.5 months. Features whose own distribution drifts therefore teach the\n# model a threshold that has stopped meaning what it meant. Replacing a raw\n# value with its percentile against the trailing 14 days of traffic makes the\n# feature stationary by construction.\n#\n# Arms: keep raw / add percentiles alongside / replace raw with percentiles.\n# The replacement arm is the real test of the hypothesis.\ndef main():\n    t0 = time.time()\n    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])\n    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])\n    df = build_combined_frame(train, test)\n    assert_frame_sane(df)\n    del train, test\n    gc.collect()\n    df = build_features(df)\n    leakage_assertions(df)\n    feature_cols, cat_cols = get_feature_columns(df)\n    GRAPH = [c for c in feature_cols if "component_size_prior" in c]\n    BASE = [c for c in feature_cols if c not in GRAPH]   # arm C from the final run\n    print(f"featurized={df.shape}, base={len(BASE)} features ({time.time()-t0:.0f}s)", flush=True)\n\n    # --- pick the drifting features empirically, on TRAIN rows only ---\n    numeric = [c for c in BASE if c not in cat_cols and\n               pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 2]\n    psi = population_psi(df[~df["is_test"]], numeric,\n                         early=("2026-01-01", "2026-03-01"),\n                         late=("2026-05-15", "2026-07-16"))\n    DRIFTY = [c for c in psi.index if psi[c] >= 0.25][:40]\n    print(f"\\nPSI over {len(numeric)} numeric features; {int((psi>=0.25).sum())} above 0.25, taking {len(DRIFTY)}")\n    print("top 15 drifters:")\n    for c in psi.index[:15]:\n        print(f"   {c:44s} PSI={psi[c]:.3f}")\n\n    df = add_time_local_percentiles(df, DRIFTY, window_days=14, suffix="_pct")\n    PCT = [c + "_pct" for c in DRIFTY]\n    print(f"\\nadded {len(PCT)} percentile columns ({time.time()-t0:.0f}s)", flush=True)\n\n    labeled = df[~df["is_test"]]\n    test_df = df[df["is_test"]]\n    frame_sorted = bool(df[TIME_COL].is_monotonic_increasing)\n    del df\n    gc.collect()\n\n    WINDOWS = {\n        "Jun25-Jul02": ("2026-06-25", "2026-07-02"),\n        "Jul02-Jul08": ("2026-07-02", "2026-07-08"),\n        "Jul08-Jul16": ("2026-07-08", "2026-07-16"),\n        "fold2-guard": ("2026-06-18", "2026-07-02"),\n    }\n    ARMS = {\n        "A_raw(current best)": BASE,\n        "B_raw+percentiles":   BASE + PCT,\n        "C_percentiles_only":  [c for c in BASE if c not in DRIFTY] + PCT,\n    }\n\n    results = {}\n    for arm, cols in ARMS.items():\n        cc = [c for c in cat_cols if c in cols]\n        results[arm] = {}\n        for wname, (s_, e_) in WINDOWS.items():\n            s_ts, e_ts = pd.Timestamp(s_), pd.Timestamp(e_)\n            tr = labeled.loc[labeled[TIME_COL] < s_ts]\n            va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]\n            X_tr, y_tr = prepare_lgb_frame(tr, cols, cc), tr[LABEL_COL].astype(int)\n            X_va, y_va = prepare_lgb_frame(va, cols, cc), va[LABEL_COL].astype(int)\n            aps = []\n            for seed in (0, 1, 2):\n                params = dict(objective="binary", metric="None", verbosity=-1,\n                              learning_rate=0.02, num_leaves=127, feature_fraction=0.85,\n                              bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,\n                              seed=seed, bagging_seed=seed, feature_fraction_seed=seed)\n                ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cc, free_raw_data=False)\n                ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cc, reference=ds_tr, free_raw_data=False)\n                bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],\n                                feval=make_pr_auc_feval(y_va.values, seed=seed),\n                                callbacks=[lgb.early_stopping(200, verbose=False),\n                                           lgb.log_evaluation(period=0)])\n                aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))\n            results[arm][wname] = (float(np.mean(aps)), float(np.std(aps)))\n            print(f"  {arm:22s} {wname:12s} n_fraud={int(y_va.sum()):4d} "\n                  f"{np.mean(aps):.4f} +/- {np.std(aps):.4f}  [{time.time()-t0:.0f}s]", flush=True)\n            del X_tr, X_va\n            gc.collect()\n\n    print("\\n============== DRIFT-NORMALISATION RESULTS ==============")\n    recent = [w for w in WINDOWS if w != "fold2-guard"]\n    print(f"{\'arm\':24s}" + "".join(f"{w:>16s}" for w in WINDOWS) + f"{\'RECENT MEAN\':>14s}")\n    for arm in ARMS:\n        row = f"{arm:24s}"\n        for w in WINDOWS:\n            m, sd = results[arm][w]\n            row += f"{m:>10.4f}+-{sd:.3f}"\n        row += f"{np.mean([results[arm][w][0] for w in recent]):>14.4f}"\n        print(row)\n\n    base = np.mean([results["A_raw(current best)"][w][0] for w in recent])\n    best_arm, best_rm = "A_raw(current best)", base\n    for arm in ["B_raw+percentiles", "C_percentiles_only"]:\n        rm = np.mean([results[arm][w][0] for w in recent])\n        improved = sum(results[arm][w][0] > results["A_raw(current best)"][w][0] for w in recent)\n        worst = min(results[arm][w][0] - results["A_raw(current best)"][w][0] for w in recent)\n        ok = (rm - base) >= 0.004 and improved >= 2 and worst >= -0.003\n        print(f"\\n{arm}: recent-mean {rm:.4f} vs {base:.4f} (delta {rm-base:+.4f}), "\n              f"{improved}/{len(recent)} windows improved, worst delta {worst:+.4f} -> "\n              f"{\'ACCEPT\' if ok else \'reject (does not clear the acceptance rule)\'}")\n        if ok and rm > best_rm:\n            best_arm, best_rm = arm, rm\n\n    if best_arm == "A_raw(current best)":\n        print("\\nNo arm clears the rule. Current best stands; no submission written.")\n        return\n\n    print(f"\\nBest arm: {best_arm} -- training candidate", flush=True)\n    cols = ARMS[best_arm]\n    cc = [c for c in cat_cols if c in cols]\n    hold_start = pd.Timestamp("2026-07-01")\n    fit = labeled.loc[labeled[TIME_COL] < hold_start]\n    hold = labeled.loc[labeled[TIME_COL] >= hold_start]\n    X_f, y_f = prepare_lgb_frame(fit, cols, cc), fit[LABEL_COL].astype(int)\n    X_h, y_h = prepare_lgb_frame(hold, cols, cc), hold[LABEL_COL].astype(int)\n    base_params = dict(objective="binary", metric="None", verbosity=-1,\n                       learning_rate=0.02, num_leaves=127, feature_fraction=0.85,\n                       bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50)\n    probe = lgb.train(dict(base_params, seed=0, bagging_seed=0, feature_fraction_seed=0),\n                      lgb.Dataset(X_f, label=y_f, categorical_feature=cc, free_raw_data=False),\n                      num_boost_round=4000,\n                      valid_sets=[lgb.Dataset(X_h, label=y_h, categorical_feature=cc, free_raw_data=False)],\n                      feval=make_pr_auc_feval(y_h.values, seed=0),\n                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)])\n    rounds = int(round(probe.best_iteration * 1.1))\n    print(f"held-out(Jul01-15)={average_precision_score(y_h, probe.predict(X_h, num_iteration=probe.best_iteration)):.4f} "\n          f"rounds={rounds}", flush=True)\n    del X_f, X_h, probe\n    gc.collect()\n\n    X_full, y_full = prepare_lgb_frame(labeled, cols, cc), labeled[LABEL_COL].astype(int)\n    X_test = prepare_lgb_frame(test_df, cols, cc)\n    ids = test_df["transaction_id"].values\n    ds = lgb.Dataset(X_full, label=y_full, categorical_feature=cc, free_raw_data=False)\n    preds = np.mean([lgb.train(dict(base_params, seed=s, bagging_seed=s, feature_fraction_seed=s),\n                               ds, num_boost_round=rounds).predict(X_test) for s in (0, 1, 2, 3, 4)], axis=0)\n    assert frame_sorted and LABEL_COL not in X_test.columns\n    for c in ID_COLS:\n        assert c not in X_test.columns\n    assert np.all((preds >= 0) & (preds <= 1)) and not np.isnan(preds).any()\n    sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)\n    sub = pd.DataFrame({"transaction_id": ids, "fraud": preds})\n    sub = sub.set_index("transaction_id").loc[sample["transaction_id"]].reset_index()\n    assert list(sub["transaction_id"]) == list(sample["transaction_id"])\n    sub.to_csv(OUT_DIR / "submission.csv", index=False)\n    print(f"\\nWrote candidate submission.csv from {best_arm} ({time.time()-t0:.0f}s)")\n\n\nif __name__ == "__main__":\n    main()\n'

OBJ_MAIN = '\n\n# ================ does a ranking objective beat logloss? ====================\n# The single biggest win of this competition (+0.026) came from noticing that\n# PR-AUC is a RANK metric and that scale_pos_weight was distorting the ranking\n# to chase calibrated probabilities nobody scores. We never followed that\n# thread to its end: we still train plain logloss.\n#\n# Arms: binary logloss / lambdarank / rank_xendcg / focal loss. The two rank\n# objectives group by calendar day -- ranking transactions within a day is the\n# closest well-posed proxy for the global ranking the metric actually scores.\n# Measurement only: no submission is written from here.\ndef _day_groups(frame):\n    """Group sizes for LightGBM ranking. The frame is time-sorted, so equal\n    days are contiguous and np.unique\'s sorted counts line up with them."""\n    d = frame[TIME_COL].dt.floor("D").to_numpy()\n    _, counts = np.unique(d, return_counts=True)\n    assert counts.sum() == len(frame)\n    return counts\n\n\ndef make_focal_obj(gamma=2.0, eps=1e-3):\n    """Focal loss via finite differences of the loss itself.\n\n    Deliberately not hand-differentiated: the analytic gradient and Hessian of\n    focal loss are easy to get subtly wrong, and a wrong Hessian degrades\n    silently into a worse model rather than an error. Central differences cost\n    two extra elementwise passes and are checked exactly against logloss at\n    gamma=0 below, where focal loss reduces to it."""\n    def loss(z, y):\n        p = 1.0 / (1.0 + np.exp(-z))\n        pt = np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0)\n        return -np.power(1.0 - pt, gamma) * np.log(pt)\n\n    def obj(z, ds):\n        y = ds.get_label()\n        lp, l0, lm = loss(z + eps, y), loss(z, y), loss(z - eps, y)\n        grad = (lp - lm) / (2.0 * eps)\n        hess = np.maximum((lp - 2.0 * l0 + lm) / (eps * eps), 1e-6)\n        return grad, hess\n    return obj\n\n\ndef _check_focal():\n    """gamma=0 makes focal loss exactly logloss, whose grad/hess are known."""\n    rng = np.random.RandomState(0)\n    z = rng.normal(0, 3, 5000)\n    y = (rng.rand(5000) < 0.3).astype(np.float64)\n\n    class _DS:\n        def get_label(self):\n            return y\n\n    g, h = make_focal_obj(gamma=0.0)(z, _DS())\n    p = 1.0 / (1.0 + np.exp(-z))\n    assert np.allclose(g, p - y, atol=1e-6), np.abs(g - (p - y)).max()\n    assert np.allclose(h, np.clip(p * (1 - p), 1e-6, None), atol=1e-5), np.abs(h - p * (1 - p)).max()\n    print("focal objective reduces to logloss at gamma=0 (grad+hess verified)")\n\n\ndef main():\n    t0 = time.time()\n    _check_focal()\n    train = pd.read_csv(TRAIN_CSV, parse_dates=[TIME_COL])\n    test = pd.read_csv(TEST_CSV, parse_dates=[TIME_COL])\n    df = build_combined_frame(train, test)\n    assert_frame_sane(df)\n    del train, test\n    gc.collect()\n    df = build_features(df)\n    feature_cols, cat_cols = get_feature_columns(df)\n    cols = [c for c in feature_cols if "component_size_prior" not in c]\n    cc = [c for c in cat_cols if c in cols]\n    labeled = df[~df["is_test"]]\n    del df\n    gc.collect()\n    print(f"featurized, {len(cols)} features ({time.time()-t0:.0f}s)", flush=True)\n\n    WINDOWS = {"tail(Jul01-15)": ("2026-07-01", "2026-07-16"),\n               "fold2(Jun18-Jul02)": ("2026-06-18", "2026-07-02")}\n    COMMON = dict(metric="None", verbosity=-1, learning_rate=0.02, num_leaves=127,\n                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,\n                  min_data_in_leaf=50)\n    ARMS = {\n        "A_binary(baseline)": dict(objective="binary"),\n        "B_lambdarank":       dict(objective="lambdarank", lambdarank_truncation_level=50,\n                                   label_gain=[0, 1]),\n        "C_rank_xendcg":      dict(objective="rank_xendcg", label_gain=[0, 1]),\n        "D_focal_g2":         dict(objective=None),\n    }\n\n    results = {}\n    for arm, cfg in ARMS.items():\n        results[arm] = {}\n        for wname, (s_, e_) in WINDOWS.items():\n            s_ts, e_ts = pd.Timestamp(s_), pd.Timestamp(e_)\n            tr = labeled.loc[labeled[TIME_COL] < s_ts]\n            va = labeled.loc[(labeled[TIME_COL] >= s_ts) & (labeled[TIME_COL] < e_ts)]\n            X_tr, y_tr = prepare_lgb_frame(tr, cols, cc), tr[LABEL_COL].astype(int)\n            X_va, y_va = prepare_lgb_frame(va, cols, cc), va[LABEL_COL].astype(int)\n            needs_group = cfg.get("objective") in ("lambdarank", "rank_xendcg")\n            g_tr, g_va = (_day_groups(tr), _day_groups(va)) if needs_group else (None, None)\n            aps = []\n            for seed in (0, 1, 2):\n                params = dict(COMMON, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, **cfg)\n                # lightgbm >= 4.0 removed the fobj argument; a custom objective\n                # is passed as a callable in params instead.\n                if arm == "D_focal_g2":\n                    params["objective"] = make_focal_obj(gamma=2.0)\n                ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cc,\n                                    group=g_tr, free_raw_data=False)\n                ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cc, group=g_va,\n                                    reference=ds_tr, free_raw_data=False)\n                bst = lgb.train(params, ds_tr, num_boost_round=4000, valid_sets=[ds_va],\n                                feval=make_pr_auc_feval(y_va.values, seed=seed),\n                                callbacks=[lgb.early_stopping(200, verbose=False),\n                                           lgb.log_evaluation(period=0)])\n                # AP is invariant to any monotone rescaling, so the ranking\n                # objectives\' raw scores are scored directly, unmapped.\n                aps.append(average_precision_score(y_va, bst.predict(X_va, num_iteration=bst.best_iteration)))\n            results[arm][wname] = (float(np.mean(aps)), float(np.std(aps)))\n            print(f"  {arm:20s} {wname:20s} {np.mean(aps):.4f} +/- {np.std(aps):.4f}  "\n                  f"[{time.time()-t0:.0f}s]", flush=True)\n            del X_tr, X_va\n            gc.collect()\n\n    print("\\n================ OBJECTIVE COMPARISON ================")\n    print(f"{\'arm\':22s}" + "".join(f"{w:>22s}" for w in WINDOWS))\n    for arm in ARMS:\n        print(f"{arm:22s}" + "".join(f"{results[arm][w][0]:>16.4f}+-{results[arm][w][1]:.3f}"\n                                    for w in WINDOWS))\n    base = results["A_binary(baseline)"]["tail(Jul01-15)"][0]\n    print(f"\\nbaseline tail {base:.4f}; seed-noise std is ~0.0020, so a real win needs >= +0.004")\n    for arm in ARMS:\n        if arm == "A_binary(baseline)":\n            continue\n        d = results[arm]["tail(Jul01-15)"][0] - base\n        print(f"  {arm:20s} tail delta {d:+.4f} -> {\'WORTH PURSUING\' if d >= 0.004 else \'no\'}")\n\n\nif __name__ == "__main__":\n    main()\n'


def main():
    parts = [HEADER]
    for m in MODULES:
        src = (ROOT / "src" / m).read_text()
        body = strip_module(src)
        # config.py defines paths relative to the repo; the header already set
        # the Kaggle ones, so drop its path block.
        if m == "config.py":
            body = re.sub(r"^ROOT = .*?SAMPLE_SUBMISSION_CSV = .*?$", "", body, flags=re.S | re.M)
        parts.append(f"\n# ===================== src/{m} =====================\n{body}\n")

    parts.append('\nSEEDS = [0, 1, 2, 3, 4]\nFINAL_PARAMS = dict(\n'
                 '    objective="binary", metric="None", verbosity=-1, learning_rate=0.02,\n'
                 '    num_leaves=127, feature_fraction=0.85, bagging_fraction=0.85,\n'
                 '    bagging_freq=1, min_data_in_leaf=50,\n)\n')
    common = "".join(parts)

    for dirname, slug, title, filename, body in [
        ("kaggle_kernel", "react-2026-pipeline", "REACT 2026 Pipeline", "react_pipeline.py", MAIN),
        ("kaggle_sweep", "react-2026-sweep", "REACT 2026 Sweep", "react_sweep.py", SWEEP_MAIN),
        ("kaggle_exp2", "react-2026-exp2", "REACT 2026 Exp2", "react_exp2.py", EXP2_MAIN),
        ("kaggle_exp3", "react-2026-exp3", "REACT 2026 Exp3", "react_exp3.py", EXP3_MAIN),
        ("kaggle_final", "react-2026-final", "REACT 2026 Final", "react_final.py", FINAL_MAIN),
        ("kaggle_horizon", "react-2026-horizon", "REACT 2026 Horizon", "react_horizon.py", HORIZON_MAIN),
        ("kaggle_drift", "react-2026-drift", "REACT 2026 Drift", "react_drift.py", DRIFT_MAIN),
        ("kaggle_obj", "react-2026-obj", "REACT 2026 Obj", "react_obj.py", OBJ_MAIN),
    ]:
        d = ROOT / dirname
        d.mkdir(exist_ok=True)
        (d / filename).write_text(common + body)
        (d / "kernel-metadata.json").write_text(f'''{{
  "id": "ahsanulhoque48cu/{slug}",
  "title": "{title}",
  "code_file": "{filename}",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["react-2026-datathon"],
  "kernel_sources": []
}}
''')
        print(f"Wrote {d/filename} ({len((common+body).splitlines())} lines)")


if __name__ == "__main__":
    main()
