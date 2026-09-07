#!/usr/bin/env python3
"""
scripts/40_generate_method_b_and_grand_blend.py

Generates:
1. CANDIDATE_method_b_rank_0.5365.csv:
   - Propagate LightGBM test predictions independently (+/-60m, w=0.50, cust+dev)
   - Propagate Tabular ResNet test predictions independently (+/-60m, w=0.50, cust+dev)
   - 90% Rank(Prop_LGBM) + 10% Rank(Prop_ResNet)
   - Local validation PR-AUC: 0.5365 (Projected LB: ~0.5670)

2. CANDIDATE_grand_blend_method_b_and_v6.csv:
   - 50% Method B Champion + 50% Ratul v6 Stacker (local probe 0.5370)
   - The ultimate multi-paradigm ensemble combining Dual-Horizon Trees + Neural Manifold + Learned Stacker.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import PROCESSED_DIR, TIME_COL, SUBMISSIONS_DIR, SAMPLE_SUBMISSION_CSV
from src.model import sliding_loo_blend, PROP_W, PROP_WINDOW_MINUTES, PROP_ENTITIES


def main():
    print("=" * 75)
    print("GENERATING METHOD B CHAMPION (0.5365) & ULTRA GRAND BLEND WITH V6")
    print("=" * 75)

    # 1. Load Data
    t0 = time.time()
    df = pd.read_pickle(PROCESSED_DIR / "features.pkl")
    test_df = df[df["is_test"]].copy()
    del df
    print(f"Loaded test frame ({len(test_df)} rows) in {time.time()-t0:.1f}s")

    # 2. Load cached test predictions
    lgb_test_cache = PROCESSED_DIR / "lgb_champ_test.npy"
    resnet_test_file = SUBMISSIONS_DIR / "resnet_predictions_test.csv"
    v6_file = Path("kaggle_v6/output/submission.csv")

    assert lgb_test_cache.exists(), f"Missing LGBM cache: {lgb_test_cache}"
    assert resnet_test_file.exists(), f"Missing ResNet test: {resnet_test_file}"
    assert v6_file.exists(), f"Missing v6 submission: {v6_file}"

    p_lgb_test = np.load(lgb_test_cache)
    df_resnet = pd.read_csv(resnet_test_file)
    p_resnet_test = df_resnet["resnet_prob"].values
    sub_v6 = pd.read_csv(v6_file)

    assert len(p_lgb_test) == len(p_resnet_test) == len(sub_v6) == len(test_df) == 262648

    # 3. Independent Propagation
    print("\nApplying independent propagation to test predictions...")
    ent_test = {e: test_df[e].values for e in PROP_ENTITIES}
    timestamps_test = test_df[TIME_COL]

    t1 = time.time()
    prop_lgb_test = sliding_loo_blend(p_lgb_test, ent_test, timestamps_test, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
    print(f"  LightGBM test propagation complete ({time.time()-t1:.1f}s)")

    t1 = time.time()
    prop_resnet_test = sliding_loo_blend(p_resnet_test, ent_test, timestamps_test, w=PROP_W, window_minutes=PROP_WINDOW_MINUTES)
    print(f"  ResNet test propagation complete ({time.time()-t1:.1f}s)")

    # 4. Method B Rank Blend (90% LGBM + 10% ResNet)
    print("\nComputing Method B Rank Blend (90% LGBM + 10% ResNet)...")
    r_lgb = rankdata(prop_lgb_test) / len(prop_lgb_test)
    r_resnet = rankdata(prop_resnet_test) / len(prop_resnet_test)

    r_method_b = 0.90 * r_lgb + 0.10 * r_resnet
    p_method_b = (r_method_b - r_method_b.min()) / (r_method_b.max() - r_method_b.min())

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    out_b = pd.DataFrame({"transaction_id": test_df["transaction_id"].values, "fraud": p_method_b})
    out_b = out_b.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

    path_cand_b = SUBMISSIONS_DIR / "CANDIDATE_method_b_rank_0.5365.csv"
    out_b.to_csv(path_cand_b, index=False)
    print(f"✅ Saved Method B Champion: {path_cand_b}")
    print(f"   Summary: min={out_b['fraud'].min():.6f}, mean={out_b['fraud'].mean():.6f}, max={out_b['fraud'].max():.6f}")

    # 5. Grand Blend: Method B + v6 Stacker
    print("\nComputing Ultra Grand Blend (50% Method B + 50% v6 Stacker)...")
    sub_v6 = sub_v6.set_index("transaction_id").loc[sample_sub["transaction_id"]].reset_index()

    corr = out_b["fraud"].corr(sub_v6["fraud"], method="spearman")
    print(f"   Spearman correlation between Method B and v6: r = {corr:.4f}")

    r_b = rankdata(out_b["fraud"]) / len(out_b)
    r_v6 = rankdata(sub_v6["fraud"]) / len(sub_v6)

    blend_ranks = 0.50 * r_b + 0.50 * r_v6
    blend_probs = (blend_ranks - blend_ranks.min()) / (blend_ranks.max() - blend_ranks.min())

    out_grand = pd.DataFrame({"transaction_id": out_b["transaction_id"], "fraud": blend_probs})
    path_grand = SUBMISSIONS_DIR / "CANDIDATE_grand_blend_method_b_and_v6.csv"
    out_grand.to_csv(path_grand, index=False)
    print(f"✅ Saved Ultra Grand Blend: {path_grand}")
    print(f"   Summary: min={out_grand['fraud'].min():.6f}, mean={out_grand['fraud'].mean():.6f}, max={out_grand['fraud'].max():.6f}")

    # 6. Assertions
    for name, pth in [("Method B", path_cand_b), ("Grand Blend", path_grand)]:
        chk = pd.read_csv(pth)
        assert len(chk) == 262648, f"{name}: Wrong row count {len(chk)}"
        assert chk["fraud"].isna().sum() == 0, f"{name}: Contains NaNs"
        assert list(chk["transaction_id"]) == list(sample_sub["transaction_id"]), f"{name}: IDs mismatch"
        assert (chk["fraud"] >= 0.0).all() and (chk["fraud"] <= 1.0).all(), f"{name}: Bounds error"
    print("\nAll integrity checks passed successfully!")


if __name__ == "__main__":
    main()
