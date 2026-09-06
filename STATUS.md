# REACT 2026 Datathon — Team Status Dashboard

Last updated: **2026-09-07 01:20 Dhaka** · deadline **2026-09-07 23:59:59 Dhaka** (~22.5h left)

> Modeling plan and rationale: [`PLAN.md`](./PLAN.md) · Candidate registry: [`submissions/CANDIDATE_REGISTRY.md`](./submissions/CANDIDATE_REGISTRY.md)

---

## 🏆 Current Champion Candidate — Ready for Morning Submission (06:00 AM Reset)

| Metric / Attribute | Value |
|---|---|
| **Local Held-Out Tail PR-AUC** | **0.5288** (All-time repository high) |
| **Lift over Live Best (0.5204)** | **+0.0084** (4.2× the 0.0020 seed-noise std) |
| **Projected Public Leaderboard** | **~0.5478 – 0.5510** (Target: ~0.550) |
| **Current Live LB Score** | 0.53948 (11th place) |
| **Artifact** | [`submissions/CANDIDATE_tuned_horizon_0.5288.csv`](./submissions/CANDIDATE_tuned_horizon_0.5288.csv) |
| **MD5 Checksum** | `6d87e95b5dab95654c19b95fec97fa54` |
| **Git Branch & Tag** | Branch: `candidate/tuned-horizon-0.5288`, Tag: `candidate-0.5288` (`2a9476f`) |
| **Exact Reproduction Command** | `git checkout candidate-0.5288 && .venv/bin/python scripts/24_generate_0.5288_candidate.py` |

### Architecture of the 0.5288 Champion:
* **Top 160 Pruned Feature Core (`data/processed/optimal_pruned_features.csv`):** Solved the feature dilution bottleneck by stripping out 60+ near-zero gain features, restoring deep tree growth to 882 boosting rounds.
* **Dual-Horizon Specialist Ensemble:**
  * **48% Full Train Model:** Jan 01 $\to$ Jul 01 (672k rows, `leaves=127, min_data=50, FF=0.85, lr=0.02`, iter 882, PR-AUC = 0.5269). Learns lifetime customer spending distributions and historical baseline z-scores.
  * **52% Tuned 90-Day Specialist:** Apr 01 $\to$ Jul 01 (346k rows, `leaves=63, min_data=100, FF=0.75, lr=0.02`, iter 433, PR-AUC = 0.5272). Learns July's daytime fraud, location bursts, and smurfing dynamics.
  * **Rank Correlation between the two models:** $r = 0.614$, providing genuine structural diversity.
  * **Blend Stability:** Plateau across $w \in [0.325, 0.600]$ all score `0.5288`.

---

## 🔒 Current Live Best (Holding)

| Metric / Attribute | Value |
|---|---|
| **Public LB Score** | **0.53948** |
| **Position** | 11th place |
| **Local Validation (Jul 1–15)** | 0.5204 |
| **Artifact** | `submissions/BEST_0.53948_2329b1b.csv` (md5 `334ce114e8f761538871e691ce0227f4`) |
| **Code State** | commit `2329b1b`, git tag `best-0.53948` |

---

## ⏰ Submission Budget & Schedule

* **Daily Limit:** 5 submissions per day.
* **Reset Time:** **06:00 AM Dhaka Time (00:00 UTC)** — in ~4.5 hours.
* **Competition Deadline:** **Tonight, 7 Sep 2026, 11:59:59 PM Dhaka Time** (~22.5 hours remaining).
* **Morning Plan (Sep 7):**
  1. At 06:00 AM reset, submit `submissions/CANDIDATE_tuned_horizon_0.5288.csv` using slot 1/5.
  2. Verify the empirical +0.019 offset on the live Kaggle leaderboard.
  3. Reserve the remaining 4 submission slots for final calibration before the 11:59 PM deadline.

---

## 📊 Score Progression History

| Candidate / Run | Local Tail PR-AUC | Public LB | Status / Notes |
|---|---|---|---|
| Baseline v1 (Raw fields only) | 0.5001 | 0.16644 | Initial submission |
| Ratul Full Pipeline | 0.5133 | 0.51309 | Baseline behavioral features |
| Ratul Class Weight Fix | 0.5204 | **0.53948** | Removed `scale_pos_weight` (Current LB Champion) |
| Sanzid Local Expanded (216 feats) | 0.5246 | — | Candidate generated (`00c922d`) |
| Top 160 Core (Pruning Search) | 0.5269 | — | Fixed feature dilution (882 rounds) |
| Multi-Horizon Blend (Full + Untuned 90d) | 0.5282 | — | Broke repo record (`59f5443`) |
| **🏆 Tuned Multi-Horizon Blend** | **0.5288** | **[~0.548–0.551 proj.]** | **Stored & tagged `candidate-0.5288`** |

---

## 🔬 Confirmed Engineering Discoveries & Settled Axes

### 1. What Worked (Confirmed Wins)
* **Feature Pruning to Top 160 Core:** Expanded 227-feature set caused tree dilution and premature stopping at 139–232 rounds (PR-AUC 0.5193). Pruning to the Top 160 non-diluting features restored tree depth to 882 rounds and lifted PR-AUC to `0.5269`.
* **Multi-Horizon Window Specialization:** While recency weighting inside a single model failed, ensembling separate time-horizon models preserves both macro baselines (Jan–Jul) and recent daytime/smurfing shifts (Apr–Jul). Lifted score from `0.5269` to `0.5282`.
* **Specialist Regularization Tuning:** Calibrating `num_leaves=63, min_data=100, FF=0.75` for the 346k-row 90-day specialist lifted the specialist from `0.5259` to `0.5272`, pushing the blend to **`0.5288`**.
* **Removing `scale_pos_weight`:** Retained from earlier analysis; PR-AUC is a rank metric, and unweighted logloss (`spw=1.0`) produces optimal probability calibration.

### 2. Settled Negative Results (Do Not Retry)
* **Secondary / Interaction Feature Expansion:** Appending 9 location-surge and customer-familiarity features (`loc_surge_accel`, `cust_loc_visit_share`, `dev_loc_prior_uses`, etc.) introduced feature dilution, lowering rounds and dropping blend PR-AUC from `0.5288` to `0.5255`.
* **DART Boosting:** Slower (608s) and underperformed standard GBDT (`0.5234` vs `0.5269`).
* **Custom Focal Loss:** Early-stopped too early (166 rounds) and underperformed cross-entropy (`0.5256` vs `0.5269`).
* **Full Train Hyperparameter Sweeps:** The incumbent configuration (`num_leaves=127, min_data=50, FF=0.85, lr=0.02`) is already at the global optimum on Full Train (`0.5269`).
* **Ultra-Recent Specialist Windows:** 45-day (`0.5220`) and 60-day (`0.5229`) underperformed 90-day (`0.5272`) due to sample size constraints.

---

## 📁 Candidate Preservation & Checksums

| Candidate File | Local Tail | Git Tag / Commit | MD5 Checksum | Status |
|---|---|---|---|---|
| [`CANDIDATE_tuned_horizon_0.5288.csv`](./submissions/CANDIDATE_tuned_horizon_0.5288.csv) | **0.5288** | `candidate-0.5288` (`2a9476f`) | `6d87e95b5dab95654c19b95fec97fa54` | **Active Champion Candidate** |
| [`CANDIDATE_multimodel_multihorizon_0.5282.csv`](./submissions/CANDIDATE_multimodel_multihorizon_0.5282.csv) | 0.5282 | `candidate-0.5282` (`59f5443`) | `9ee7c0095d28c796059717e2e48ffc06` | Archived (Un-tuned 90d Blend) |
| [`CANDIDATE_pruned_top160_lgb_0.5269.csv`](./submissions/CANDIDATE_pruned_top160_lgb_0.5269.csv) | 0.5269 | `candidate-0.5269` (`ab2fe4e`) | `2fc4b6148a2cecaa26c0258a5c137b5e` | Archived (Single Full-Train LGB) |
| [`CANDIDATE_sanzid_local_0.5246.csv`](./submissions/CANDIDATE_sanzid_local_0.5246.csv) | 0.5246 | `candidate-0.5246` (`00c922d`) | `b649c49c45494999b78911ed71d9a265` | Archived (All 216 features) |
| `BEST_0.53948_2329b1b.csv` | 0.5204 | commit `2329b1b` | `334ce114e8f761538871e691ce0227f4` | **Current Live LB: 0.53948** |
