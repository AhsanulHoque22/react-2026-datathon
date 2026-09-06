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

    CONFIGS = []
    for lr in (0.10, 0.05, 0.02, 0.01):
        for leaves in (31, 63, 127):
            CONFIGS.append(dict(learning_rate=lr, num_leaves=leaves))

    results = []
    for cfg in CONFIGS:
        row = {**cfg}
        for wname, (X_tr, y_tr, X_va, y_va) in splits.items():
            aps, iters = [], []
            for seed in (0, 1, 2):
                params = dict(objective="binary", metric="None", verbosity=-1,
                              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
                              min_data_in_leaf=50, seed=seed, bagging_seed=seed,
                              feature_fraction_seed=seed, **cfg)
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
            print(f"lr={cfg['learning_rate']:<5} leaves={cfg['num_leaves']:<4} {wname}: "
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
                 '    objective="binary", metric="None", verbosity=-1, learning_rate=0.05,\n'
                 '    num_leaves=63, feature_fraction=0.85, bagging_fraction=0.85,\n'
                 '    bagging_freq=1, min_data_in_leaf=50,\n)\n')
    common = "".join(parts)

    for dirname, slug, title, filename, body in [
        ("kaggle_kernel", "react-2026-pipeline", "REACT 2026 Pipeline", "react_pipeline.py", MAIN),
        ("kaggle_sweep", "react-2026-sweep", "REACT 2026 Sweep", "react_sweep.py", SWEEP_MAIN),
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
