#!/usr/bin/env python3
"""
scripts/42_train_forward_champion.py

Full-Train Forward Champion Pipeline:
1. Augment 160 core pruned features with 18 forward window features (cust + dev).
2. 5-Seed Dual-Horizon LightGBM Ensemble:
   - Full-Train (814 rounds, num_leaves=127, lr=0.02, ff=0.85, mdl=50)
   - 90-Day Specialist (430 rounds, num_leaves=63, lr=0.02, ff=0.75, mdl=100)
   - Horizon blend: 48% Full-Train + 52% 90-Day Specialist
3. Multi-Paradigm Neural Fusion:
   - 90% Rank(Forward LGBM) + 10% Rank(Tabular ResNet)
4. Grand Blend with Ratul's v6 Stacker:
   - 50% Forward Tree-Neural + 50% v6 Stacker
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import lightgbm as lgb

from src.config import PROCESSED_DIR, TIME_COL, LABEL_COL, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.features import add_forward_window_features
from src.model import prepare_lgb_frame, CAT_COLS

SEEDS = [42, 100, 2026, 7, 777]


def main():
    print("=" * 75)
    print("FORWARD-LOOKING CHAMPION FULL-TRAIN PIPELINE")
    print("=" * 75)
    t0 = time.time()

    # 1. Load Data
    df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
    selected_cols = list(pd.read_csv(PROCESSED_DIR / "optimal_pruned_features.csv")["0"].values)
    selected_cats = [c for c in CAT_COLS if c in selected_cols]
    print(f"Loaded {df.shape} in {time.time()-t0:.1f}s")

    # 2. Add Forward Features
    t_fwd = time.time()
    df = add_forward_window_features(df, "customer_id", "cust", windows_min=(60, 1440))
    df = add_forward_window_features(df, "device_id", "dev", windows_min=(60, 1440))
    print(f"Computed forward features in {time.time()-t_fwd:.1f}s")

    fwd_cols = [c for c in df.columns if "_fwd_" in c or "_sym_" in c or c.endswith("_seconds_to_next")]
    all_cols = selected_cols + fwd_cols
    print(f"Total features: {len(all_cols)} (160 base + {len(fwd_cols)} forward)")

    # 3. Splits
    labeled = df[~df["is_test"]].copy()
    test_df = df[df["is_test"]].copy()
    del df

    # Mask for 90-day specialist (last 90 days of training data: 2026-04-16 onwards)
    train_end = labeled[TIME_COL].max()
    ninety_days_ago = train_end - pd.Timedelta(days=90)
    mask_90d = labeled[TIME_COL] >= ninety_days_ago
    train_90d = labeled.loc[mask_90d].copy()

    print(f"Full Train: {len(labeled)} rows | 90-Day Specialist: {len(train_90d)} rows (since {ninety_days_ago.date()})")
    print(f"Test Set:   {len(test_df)} rows")

    y_full = labeled[LABEL_COL].astype(int)
    y_90d = train_90d[LABEL_COL].astype(int)

    X_full = prepare_lgb_frame(labeled, all_cols, selected_cats)
    X_90d  = prepare_lgb_frame(train_90d, all_cols, selected_cats)
    X_test = prepare_lgb_frame(test_df, all_cols, selected_cats)

    # 4. Train Full-Train Models (5 Seeds)
    print("\n" + "-"*60)
    print("Fitting 5-Seed Full-Train LightGBM (814 rounds)...")
    print("-" * 60)
    ds_full = lgb.Dataset(X_full, label=y_full, categorical_feature=selected_cats, free_raw_data=False)
    
    preds_full_seeds = []
    for s in SEEDS:
        t_s = time.time()
        prm = dict(
            objective="binary", metric="None", seed=s, bagging_seed=s,
            feature_fraction_seed=s, verbosity=-1,
            learning_rate=0.02, num_leaves=127, feature_fraction=0.85,
            bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=50,
        )
        bst = lgb.train(prm, ds_full, num_boost_round=814)
        pred = bst.predict(X_test)
        preds_full_seeds.append(pred)
        print(f"  Seed {s:5d} done in {time.time()-t_s:.1f}s")

    p_full_test = np.mean(preds_full_seeds, axis=0)

    # 5. Train 90-Day Specialist Models (5 Seeds)
    print("\n" + "-"*60)
    print("Fitting 5-Seed 90-Day Specialist LightGBM (430 rounds)...")
    print("-" * 60)
    ds_90d = lgb.Dataset(X_90d, label=y_90d, categorical_feature=selected_cats, free_raw_data=False)
    
    preds_90d_seeds = []
    for s in SEEDS:
        t_s = time.time()
        prm = dict(
            objective="binary", metric="None", seed=s, bagging_seed=s,
            feature_fraction_seed=s, verbosity=-1,
            learning_rate=0.02, num_leaves=63, feature_fraction=0.75,
            bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=100,
        )
        bst = lgb.train(prm, ds_90d, num_boost_round=430)
        pred = bst.predict(X_test)
        preds_90d_seeds.append(pred)
        print(f"  Seed {s:5d} done in {time.time()-t_s:.1f}s")

    p_90d_test = np.mean(preds_90d_seeds, axis=0)

    # Horizon Blend
    p_fwd_lgb_test = 0.48 * p_full_test + 0.52 * p_90d_test

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    # 6. Candidate 1: Pure Forward LightGBM Champion
    print("\n" + "="*60)
    print("Generating Candidate 1: Pure Forward LightGBM Dual-Horizon...")
    print("=" * 60)
    sub1 = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": p_fwd_lgb_test})
    sub1 = sub1.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    path1 = SUBMISSIONS_DIR / "CANDIDATE_forward_lgb_dual_0.5377.csv"
    sub1.to_csv(path1, index=False)
    print(f"✅ Saved Candidate 1: {path1}")

    # 7. Candidate 2: Forward Tree-Neural Champion
    print("\n" + "="*60)
    print("Generating Candidate 2: Forward Tree-Neural (Method B Fusion)...")
    print("=" * 60)
    resnet_test_file = SUBMISSIONS_DIR / "resnet_predictions_test.csv"
    assert resnet_test_file.exists(), f"Missing ResNet: {resnet_test_file}"
    p_resnet_test = pd.read_csv(resnet_test_file)["resnet_prob"].values

    r_fwd_lgb = rankdata(p_fwd_lgb_test) / len(p_fwd_lgb_test)
    r_resnet  = rankdata(p_resnet_test) / len(p_resnet_test)

    r_tree_neural = 0.90 * r_fwd_lgb + 0.10 * r_resnet
    p_tree_neural = (r_tree_neural - r_tree_neural.min()) / (r_tree_neural.max() - r_tree_neural.min())

    sub2 = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": p_tree_neural})
    sub2 = sub2.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    path2 = SUBMISSIONS_DIR / "CANDIDATE_forward_tree_neural_0.5385.csv"
    sub2.to_csv(path2, index=False)
    print(f"✅ Saved Candidate 2: {path2}")

    # 8. Candidate 3: Grand Blend with Ratul's v6 Stacker
    print("\n" + "="*60)
    print("Generating Candidate 3: Grand Blend (50% Forward Tree-Neural + 50% v6)...")
    print("=" * 60)
    v6_file = Path("kaggle_v6/output/submission.csv")
    assert v6_file.exists(), f"Missing v6: {v6_file}"
    sub_v6 = pd.read_csv(v6_file).set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

    corr = sub2["fraud"].corr(sub_v6["fraud"], method="spearman")
    print(f"  Spearman correlation with v6: r = {corr:.4f}")

    r_sub2 = rankdata(sub2["fraud"]) / len(sub2)
    r_v6 = rankdata(sub_v6["fraud"]) / len(sub_v6)

    r_grand = 0.50 * r_sub2 + 0.50 * r_v6
    p_grand = (r_grand - r_grand.min()) / (r_grand.max() - r_grand.min())

    sub3 = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": p_grand})
    sub3 = sub3.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()
    path3 = SUBMISSIONS_DIR / "CANDIDATE_grand_blend_forward_and_v6.csv"
    sub3.to_csv(path3, index=False)
    print(f"✅ Saved Candidate 3: {path3}")

    # 9. Integrity Verification
    print("\n" + "="*60)
    print("Integrity Verifications:")
    print("=" * 60)
    for name, pth in [("Pure Forward LGBM", path1), ("Forward Tree-Neural", path2), ("Grand Blend v6", path3)]:
        chk = pd.read_csv(pth)
        assert len(chk) == 262648, f"{name}: Wrong row count"
        assert chk["fraud"].isna().sum() == 0, f"{name}: Contains NaNs"
        assert list(chk["transaction_id"]) == list(sample_sub["transaction_id"]), f"{name}: IDs mismatch"
        assert (chk["fraud"] >= 0.0).all() and (chk["fraud"] <= 1.0).all(), f"{name}: Bounds error"
        print(f"  {name:<25}: 262,648 rows | 0 NaNs | min={chk['fraud'].min():.6f}, mean={chk['fraud'].mean():.6f}, max={chk['fraud'].max():.6f}")

    print(f"\n🎉 ALL CANDIDATES GENERATED AND VERIFIED SUCCESSFULLY in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
