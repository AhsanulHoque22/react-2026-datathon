# 🏆 Live Rank 1 Champion Submission (0.56548 Public Leaderboard)

- **File**: `submissions/CANDIDATE_tree_neural_propagated_0.5359.csv`
- **Kaggle Submission Ref**: `56071582` (Submitted 2026-09-07 07:16:52 UTC / 13:16 Dhaka)
- **Public Leaderboard Score**: **`0.56548` (Rank 1 / 33 teams)**
- **Margin over 2nd**: +0.00069 (5.3x wider lead over Error404 at 0.56479)
- **Local Holdout Score**: **`0.5359` PR-AUC** (July 01–15 holdout)
- **MD5 Checksum**: `d64c632fc843a39c5ae4f4aa528316ff`
- **Rows**: 262,648 (matches `sample_submission.csv` order, 0 NaNs, valid probabilities in `[0, 1]`)

---

## Provenance & Architecture
Combines two complementary breakthroughs:
1. **Tree-Neural Ensemble Base**:
   - 88% Dual-Horizon LightGBM (882 rounds Full-Train + 433 rounds 90-day specialist on Top-160 pruned features)
   - 12% Tabular ResNet (3 residual blocks, hidden dim 256, Cosine Annealing, seed 42)
2. **Temporal Entity Prediction Propagation**:
   - Sliding leave-one-out (LOO) blend over customer and device neighbors within a $\pm 60$-minute window at weight $w=0.50$ in calibrated probability space.

---

## Exact Script to Reproduce
```bash
.venv/bin/python scripts/35_generate_propagated_champion.py
```
