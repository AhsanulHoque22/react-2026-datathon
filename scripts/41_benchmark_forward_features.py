#!/usr/bin/env python3
"""
scripts/41_benchmark_forward_features.py

Frontier 1: Forward-Looking Window Features Benchmark
Evaluates Ratul's newest hypothesis (commit a19b192):
Giving the tree model forward-looking window counts directly:
  cust_fwd_cnt_60m, cust_fwd_amtsum_60m, cust_sym_cnt_60m, cust_sym_amtsum_60m, cust_seconds_to_next
  dev_fwd_cnt_60m, dev_fwd_amtsum_60m, dev_sym_cnt_60m, dev_sym_amtsum_60m, dev_seconds_to_next

Tests on the standard held-out validation split (Jul 01 - 15):
1. Base LightGBM (pruned features): Raw vs Propagated
2. Base LightGBM + Forward Features: Raw vs Propagated
3. Feature importance gain of the forward features.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED
from src.features import add_forward_window_features
from src.model import prepare_lgb_frame, CAT_COLS, sliding_loo_blend, PROP_W, PROP_WINDOW_MINUTES, PROP_ENTITIES


def main():
    print("=" * 75)
    print("FRONTIER 1: FORWARD-LOOKING WINDOW FEATURES BENCHMARK")
    print("=" * 75)

    t0 = time.time()
    df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
    selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
    selected_cats = [c for c in CAT_COLS if c in selected_cols]
    print(f"Loaded {df.shape} in {time.time()-t0:.1f}s")

    # Add forward-looking window features for customer and device
    t_fwd = time.time()
    df = add_forward_window_features(df, "customer_id", "cust", windows_min=(60, 1440))
    df = add_forward_window_features(df, "device_id", "dev", windows_min=(60, 1440))
    print(f"Computed forward features in {time.time()-t_fwd:.1f}s")

    fwd_cols = [
        "cust_fwd_cnt_60m", "cust_fwd_amtsum_60m", "cust_sym_cnt_60m", "cust_sym_amtsum_60m", "cust_seconds_to_next",
        "cust_fwd_cnt_1440m", "cust_fwd_amtsum_1440m", "cust_sym_cnt_1440m", "cust_sym_amtsum_1440m",
        "dev_fwd_cnt_60m", "dev_fwd_amtsum_60m", "dev_sym_cnt_60m", "dev_sym_amtsum_60m", "dev_seconds_to_next",
        "dev_fwd_cnt_1440m", "dev_fwd_amtsum_1440m", "dev_sym_cnt_1440m", "dev_sym_amtsum_1440m",
    ]
    fwd_cols = [c for c in fwd_cols if c in df.columns]
    print(f"Forward columns ({len(fwd_cols)}): {fwd_cols}")

    labeled = df[~df["is_test"]].copy()
    del df

    holdout_start = pd.Timestamp("2026-07-01")
    fit_full = labeled.loc[labeled[TIME_COL] < holdout_start].copy()
    hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start].copy()
    y_full = fit_full[LABEL_COL].astype(int)
    y_hold = hold_df[LABEL_COL].astype(int)

    ent_hold = {e: hold_df[e].values for e in PROP_ENTITIES}
    timestamps_hold = hold_df[TIME_COL]

    # Model parameters (standard Champion settings)
    prm = dict(
        objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
        feature_fraction_seed=SEED, verbosity=-1,
        learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
    )

    # --- Arm A: Base (Pruned Top 160) ---
    print("\nTraining Arm A: Base Features (Top 160)...")
    X_full_base = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
    X_hold_base = prepare_lgb_frame(hold_df, selected_cols, selected_cats)

    ds_base = lgb.Dataset(X_full_base, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
    b_base = lgb.train(prm, ds_base, num_boost_round=882)
    p_base_raw = b_base.predict(X_hold_base)
    p_base_prop = sliding_loo_blend(p_base_raw, ent_hold, timestamps_hold, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)

    auc_base_raw = average_precision_score(y_hold, p_base_raw)
    auc_base_prop = average_precision_score(y_hold, p_base_prop)
    print(f"  Arm A Base: Raw PR-AUC = {auc_base_raw:.4f} | Propagated PR-AUC = {auc_base_prop:.4f}")

    # --- Arm B: Base + Forward Features ---
    print("\nTraining Arm B: Base + Forward Window Features...")
    cols_b = selected_cols + fwd_cols
    X_full_fwd = prepare_lgb_frame(fit_full, cols_b, selected_cats)
    X_hold_fwd = prepare_lgb_frame(hold_df, cols_b, selected_cats)

    ds_fwd = lgb.Dataset(X_full_fwd, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
    b_fwd = lgb.train(prm, ds_fwd, num_boost_round=882)
    p_fwd_raw = b_fwd.predict(X_hold_fwd)
    p_fwd_prop = sliding_loo_blend(p_fwd_raw, ent_hold, timestamps_hold, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)

    auc_fwd_raw = average_precision_score(y_hold, p_fwd_raw)
    auc_fwd_prop = average_precision_score(y_hold, p_fwd_prop)
    print(f"  Arm B Fwd:  Raw PR-AUC = {auc_fwd_raw:.4f} | Propagated PR-AUC = {auc_fwd_prop:.4f}")

    # Feature Importance of Forward Features
    imp = pd.Series(b_fwd.feature_importance("gain"), index=cols_b).sort_values(ascending=False)
    tot_gain = imp.sum()
    print("\nTop 15 Most Important Features by Gain in Arm B:")
    for rank, (feat, g) in enumerate(imp.head(15).items(), 1):
        is_fwd = " [FORWARD]" if feat in fwd_cols else ""
        print(f"  {rank:2d}. {feat:<35} : {g / tot_gain * 100:.2f}%{is_fwd}")

    fwd_gain_pct = imp[imp.index.isin(fwd_cols)].sum() / tot_gain * 100
    print(f"\nTotal Gain Capture by all Forward Features: {fwd_gain_pct:.2f}%")

    # --- Summary ---
    print("\n" + "=" * 75)
    print("VERDICT & COMPARISON")
    print("=" * 75)
    print(f"{'Configuration':<30} | {'Raw PR-AUC':>12} | {'Propagated PR-AUC':>18} | {'Net Delta vs Base':>18}")
    print("-" * 86)
    print(f"{'Arm A (Base)':<30} | {auc_base_raw:12.4f} | {auc_base_prop:18.4f} | {'baseline':>18}")
    print(f"{'Arm B (Base + Forward)':<30} | {auc_fwd_raw:12.4f} | {auc_fwd_prop:18.4f} | {auc_fwd_prop - auc_base_prop:+18.4f}")

    if auc_fwd_prop > auc_base_prop:
        print(f"\n✅ FORWARD FEATURES WIN! Lift: {auc_fwd_prop - auc_base_prop:+.4f} local PR-AUC.")
    elif auc_fwd_prop == auc_base_prop:
        print(f"\n⚖️ NEUTRAL: Forward features neither helped nor hurt.")
    else:
        print(f"\n❌ FEATURE DILUTION DETECTED: Forward features cost {auc_fwd_prop - auc_base_prop:+.4f} local PR-AUC. Keep them out of tree training.")


if __name__ == "__main__":
    main()
