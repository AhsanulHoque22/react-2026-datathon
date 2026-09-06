# Candidate Submissions Registry

This file tracks all generated candidate submission files, their validation scores, MD5 checksums, and git commits so any historical candidate can be recovered or reproduced at any point.

| Candidate File | Local Tail PR-AUC | Git Tag / Commit | MD5 Checksum | Status |
|---|---|---|---|---|
| `CANDIDATE_multimodel_multihorizon_0.5282.csv` | **0.5282** | pending commit | `9ee7c0095d28c796059717e2e48ffc06` | **Active Top Candidate** (Multi-Horizon Blend) |
| `CANDIDATE_pruned_top160_lgb_0.5269.csv` | 0.5269 | tag: `candidate-0.5269` (`ab2fe4e`) | `2fc4b6148a2cecaa26c0258a5c137b5e` | Archived (Single Full-Train LGB) |
| `CANDIDATE_sanzid_local_0.5246.csv` | 0.5246 | tag: `candidate-0.5246` (`00c922d`) | `b649c49c45494999b78911ed71d9a265` | Archived (All 216 features) |
| `submission_baseline.csv` | 0.1664 | tag: `baseline` | — | Submitted (0.16644 Public LB) |
| `BEST_0.53948_2329b1b.csv` | 0.5204 | commit `2329b1b` | `334ce114e8f761538871e691ce0227f4` | **Current Live Leaderboard Score: 0.53948** |

## How to Recover / Reproduce Any Candidate:
1. **Direct CSV use:** The CSVs in `submissions/` are preserved locally.
2. **Reproduce from code:**
   ```bash
   # Example: to reproduce candidate 0.5282:
   git checkout candidate-0.5282
   .venv/bin/python scripts/20_multi_model_horizon_blend.py

   # Example: to return to candidate 0.5269 code state:
   git checkout candidate-0.5269
   .venv/bin/python scripts/18_train_pruned_model.py
   ```
