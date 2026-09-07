#!/usr/bin/env python3
"""
scripts/36_blend_tree_neural_and_v6.py

Generates the Grand Blend between:
1. Our Tree-Neural Champion (0.56548 Public LB, Rank 1):
   submissions/CANDIDATE_tree_neural_propagated_0.5359.csv
2. Ratul's v6 Learned Stacker (0.5370 local probe):
   kaggle_v6/output/submission.csv

Correlation between the two: r = 0.4672 (Spearman rank).
Blend ratio: 50% Tree-Neural Champion + 50% v6 Learned Stacker.
"""

from pathlib import Path
import pandas as pd


def main():
    path_champ = Path("submissions/CANDIDATE_tree_neural_propagated_0.5359.csv")
    path_v6 = Path("kaggle_v6/output/submission.csv")
    out_path = Path("submissions/CANDIDATE_blend_056548_and_v6.csv")

    assert path_champ.exists(), f"Missing champion: {path_champ}"
    assert path_v6.exists(), f"Missing v6 submission: {path_v6}"

    print(f"Loading champion: {path_champ}")
    sub_champ = pd.read_csv(path_champ)
    print(f"Loading v6: {path_v6}")
    sub_v6 = pd.read_csv(path_v6)

    assert len(sub_champ) == len(sub_v6) == 262648, f"Mismatch row count: {len(sub_champ)} vs {len(sub_v6)}"
    assert (sub_champ["transaction_id"] == sub_v6["transaction_id"]).all(), "transaction_id order mismatch!"

    # Correlation
    corr = sub_champ["fraud"].corr(sub_v6["fraud"], method="spearman")
    print(f"Spearman rank correlation: {corr:.4f}")

    # Percentile rank normalization
    r_champ = sub_champ["fraud"].rank(pct=True)
    r_v6 = sub_v6["fraud"].rank(pct=True)

    # 50/50 blend
    blend_ranks = 0.50 * r_champ + 0.50 * r_v6

    # Scale to [0, 1] probability range
    blend_probs = (blend_ranks - blend_ranks.min()) / (blend_ranks.max() - blend_ranks.min())

    out_df = pd.DataFrame({
        "transaction_id": sub_champ["transaction_id"],
        "fraud": blend_probs
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved grand blend to: {out_path} ({len(out_df)} rows)")

    # Assertions
    assert out_df["fraud"].isna().sum() == 0, "Found NaNs in blend!"
    assert (out_df["fraud"] >= 0.0).all() and (out_df["fraud"] <= 1.0).all(), "Values out of [0, 1]!"
    print("Verification passed successfully.")


if __name__ == "__main__":
    main()
