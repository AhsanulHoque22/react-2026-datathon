#!/usr/bin/env python3
"""
scripts/38_sweep_resnet_propagation.py

Frontier 2: Neural Manifold Propagation & Weight Optimization
Benchmarks:
1. Standalone ResNet: Raw PR-AUC vs Propagated PR-AUC.
2. Method A (Ensemble then Propagate): Sweep ResNet weight 0.00 to 0.30.
3. Method B (Propagate then Ensemble): Independent propagation of ResNet and LGBM.
4. Optimal fusion calibration.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SEED, SUBMISSIONS_DIR
from src.model import prepare_lgb_frame, CAT_COLS, sliding_loo_blend, PROP_W, PROP_WINDOW_MINUTES, PROP_ENTITIES

def main():
    print("=" * 75)
    print("FRONTIER 2: NEURAL MANIFOLD PROPAGATION & WEIGHT OPTIMIZATION")
    print("=" * 75)

    # 1. Load Data
    t0 = time.time()
    df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
    selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
    selected_cats = [c for c in CAT_COLS if c in selected_cols]
    print(f"Loaded {df.shape} in {time.time()-t0:.1f}s")

    labeled = df[~df["is_test"]].copy()
    test_df = df[df["is_test"]].copy()

    holdout_start = pd.Timestamp("2026-07-01")
    fit_full = labeled.loc[labeled[TIME_COL] < holdout_start].copy()
    hold_df = labeled.loc[labeled[TIME_COL] >= holdout_start].copy()

    mask_90d = (fit_full[TIME_COL] >= "2026-04-01")
    train_90d = fit_full.loc[mask_90d].copy()

    y_full = fit_full[LABEL_COL].astype(int)
    y_hold = hold_df[LABEL_COL].astype(int)
    y_90d = train_90d[LABEL_COL].astype(int)

    # Cache file paths for LGBM holdout predictions
    lgb_hold_cache = PROCESSED_DIR / "lgb_champ_holdout.npy"
    lgb_test_cache = PROCESSED_DIR / "lgb_champ_test.npy"

    if lgb_hold_cache.exists() and lgb_test_cache.exists():
        print(f"\nLoading cached LightGBM champion predictions...")
        p_lgb_champ_hold = np.load(lgb_hold_cache)
        p_lgb_champ_test = np.load(lgb_test_cache)
    else:
        print("\nFitting LightGBM Dual-Horizon Champion...")
        X_full_lgb = prepare_lgb_frame(fit_full, selected_cols, selected_cats)
        X_hold_lgb = prepare_lgb_frame(hold_df, selected_cols, selected_cats)
        X_test_lgb = prepare_lgb_frame(test_df, selected_cols, selected_cats)
        X_90d_lgb  = X_full_lgb.loc[mask_90d]

        ds_full = lgb.Dataset(X_full_lgb, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
        p_full = dict(
            objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
            feature_fraction_seed=SEED, verbosity=-1,
            learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
            bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
        )
        b_full = lgb.train(p_full, ds_full, num_boost_round=882)
        p_lgb_full_hold = b_full.predict(X_hold_lgb)
        p_lgb_full_test = b_full.predict(X_test_lgb)

        ds_90d = lgb.Dataset(X_90d_lgb, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
        p_90d = dict(
            objective="binary", metric="None", seed=SEED, bagging_seed=SEED,
            feature_fraction_seed=SEED, verbosity=-1,
            learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
            bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
        )
        b_90d = lgb.train(p_90d, ds_90d, num_boost_round=433)
        p_lgb_90d_hold = b_90d.predict(X_hold_lgb)
        p_lgb_90d_test = b_90d.predict(X_test_lgb)

        p_lgb_champ_hold = 0.48 * p_lgb_full_hold + 0.52 * p_lgb_90d_hold
        p_lgb_champ_test = 0.48 * p_lgb_full_test + 0.52 * p_lgb_90d_test

        np.save(lgb_hold_cache, p_lgb_champ_hold)
        np.save(lgb_test_cache, p_lgb_champ_test)
        print("LightGBM champion predictions computed and cached.")

    raw_lgb_auc = average_precision_score(y_hold, p_lgb_champ_hold)
    print(f"--> LightGBM Champion Raw PR-AUC = {raw_lgb_auc:.4f}")

    # Load ResNet predictions
    resnet_hold_file = SUBMISSIONS_DIR / "resnet_predictions_holdout_jul1_15.csv"
    resnet_test_file = SUBMISSIONS_DIR / "resnet_predictions_test.csv"
    assert resnet_hold_file.exists(), f"Missing {resnet_hold_file}"

    df_resnet_hold = pd.read_csv(resnet_hold_file)
    p_resnet_hold = df_resnet_hold["resnet_prob"].values
    raw_resnet_auc = average_precision_score(y_hold, p_resnet_hold)
    print(f"--> Tabular ResNet Raw PR-AUC    = {raw_resnet_auc:.4f}")

    # Entities for propagation
    ent_hold = {e: hold_df[e].values for e in PROP_ENTITIES}
    timestamps_hold = hold_df[TIME_COL]

    # --- Part 1: Standalone Propagation Tests ---
    print("\n" + "-"*50)
    print("PART 1: STANDALONE MODEL PROPAGATION")
    print("-" * 50)
    prop_lgb_hold = sliding_loo_blend(p_lgb_champ_hold, ent_hold, timestamps_hold, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
    prop_lgb_auc = average_precision_score(y_hold, prop_lgb_hold)
    print(f"LightGBM:  Raw {raw_lgb_auc:.4f}  -->  Propagated {prop_lgb_auc:.4f}  ({prop_lgb_auc - raw_lgb_auc:+.4f})")

    prop_resnet_hold = sliding_loo_blend(p_resnet_hold, ent_hold, timestamps_hold, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
    prop_resnet_auc = average_precision_score(y_hold, prop_resnet_hold)
    print(f"ResNet:    Raw {raw_resnet_auc:.4f}  -->  Propagated {prop_resnet_auc:.4f}  ({prop_resnet_auc - raw_resnet_auc:+.4f})")

    # --- Part 2: Method A (Ensemble then Propagate) Sweep ---
    print("\n" + "-"*50)
    print("PART 2: METHOD A (ENSEMBLE RAW -> THEN PROPAGATE)")
    print("-" * 50)
    print(f"{'w_resnet':>10} | {'Raw Blend PR-AUC':>16} | {'Propagated PR-AUC':>18} | {'Lift vs Pure LGBM':>18}")
    print("-" * 70)

    best_method_a_auc = 0.0
    best_w_a = 0.0

    weights = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    for w_nn in weights:
        w_lgb = 1.0 - w_nn
        raw_blend = w_lgb * p_lgb_champ_hold + w_nn * p_resnet_hold
        raw_auc = average_precision_score(y_hold, raw_blend)
        prop_blend = sliding_loo_blend(raw_blend, ent_hold, timestamps_hold, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
        prop_auc = average_precision_score(y_hold, prop_blend)
        diff = prop_auc - prop_lgb_auc
        marker = " 🏆" if prop_auc > best_method_a_auc else ""
        if prop_auc > best_method_a_auc:
            best_method_a_auc = prop_auc
            best_w_a = w_nn
        print(f"{w_nn:10.2f} | {raw_auc:16.4f} | {prop_auc:18.4f}{marker} | {diff:+18.4f}")

    # --- Part 3: Method B (Propagate Each Model Independently -> Then Ensemble) ---
    print("\n" + "-"*50)
    print("PART 3: METHOD B (PROPAGATE INDEPENDENTLY -> THEN ENSEMBLE)")
    print("-" * 50)
    print(f"{'w_resnet':>10} | {'Prob-Space Blend':>18} | {'Rank-Space Blend':>18} | {'Lift vs Pure LGBM':>18}")
    print("-" * 70)

    best_method_b_auc = 0.0
    best_w_b = 0.0
    best_mode_b = "prob"

    r_prop_lgb = rankdata(prop_lgb_hold) / len(prop_lgb_hold)
    r_prop_resnet = rankdata(prop_resnet_hold) / len(prop_resnet_hold)

    for w_nn in weights:
        w_lgb = 1.0 - w_nn
        # Prob blend
        blend_prob = w_lgb * prop_lgb_hold + w_nn * prop_resnet_hold
        auc_prob = average_precision_score(y_hold, blend_prob)
        # Rank blend
        blend_rank = w_lgb * r_prop_lgb + w_nn * r_prop_resnet
        auc_rank = average_precision_score(y_hold, blend_rank)
        
        top_auc = max(auc_prob, auc_rank)
        diff = top_auc - prop_lgb_auc
        marker = " 🏆" if top_auc > best_method_b_auc else ""
        if top_auc > best_method_b_auc:
            best_method_b_auc = top_auc
            best_w_b = w_nn
            best_mode_b = "prob" if auc_prob >= auc_rank else "rank"
        print(f"{w_nn:10.2f} | {auc_prob:18.4f} | {auc_rank:18.4f}{marker} | {diff:+18.4f}")

    print("\n" + "="*75)
    print("SUMMARY & CONCLUSION")
    print("=" * 75)
    print(f"Ratul's Propagated Baseline (pure LGBM): {prop_lgb_auc:.4f}")
    print(f"Our Shipped Champion (Method A, w=0.12):  {best_method_a_auc:.4f} (at w={best_w_a:.2f})")
    print(f"Method B Champion (Propagate then Blend): {best_method_b_auc:.4f} (at w={best_w_b:.2f}, {best_mode_b}-space)")
    if best_method_b_auc > best_method_a_auc:
        print(f"--> Method B WINS by {best_method_b_auc - best_method_a_auc:+.4f}! Independent propagation unlocks new headroom.")
    else:
        print(f"--> Method A is optimal: joint manifold propagation produces cleaner estimates.")

if __name__ == "__main__":
    main()
