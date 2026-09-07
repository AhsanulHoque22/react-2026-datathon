# Method Summary: Multi-Horizon Tree-Neural Hybrid with Temporal Entity Propagation

**Team**: Overfit & Overcaffeinated | **Competition**: REACT 2026 Datathon (Tabular Fraud Detection)  
**Leaderboard Result**: 0.56548 Public Leaderboard | **Local PR-AUC**: 0.5359 | **Metric**: Average Precision (PR-AUC)

---

### 1. Approach Overview
Detecting payment fraud under severe class imbalance (1.76% prevalence) and temporal regime drift requires moving beyond standard decision trees. Our solution combines three synergistic pillars:
1. **Causally Sound Behavioral Engineering**: 160 scale-free, strictly-prior features capturing individual spending baselines, short-term velocity acceleration, and bipartite entity novelty.
2. **Dual-Horizon Tree-Neural Hybrid**: Blending leaf-wise gradient-boosted trees (LightGBM) across two distinct training horizons with a continuous deep neural manifold (Tabular ResNet) to resolve orthogonal decision-boundary artifacts.
3. **Unsupervised Temporal Prediction Propagation**: A post-inference leave-one-out (LOO) diffusion process that exploits empirical transaction burst clustering without label leakage.

---

### 2. Data Handling & Causally Sound Feature Engineering
Strict temporal causality is enforced throughout: for every transaction at time $t$, engineered features access information strictly before $t$ (broken by `transaction_id`). All features mechanically pass point-in-time leakage assertions; no per-entity target encoding or future information is used. From 220+ initial candidates, feature pruning isolated the **Top 160 core features**:

- **Scale-Free Behavioral Deviations**: Rather than raw amounts (which drift), amount is normalized against personal history:
  - `cust_amt_ratio`: Amount divided by the customer's prior expanding mean (single highest-gain feature).
  - `cust_amt_robust_z`: Robust MAD-based z-score around expanding median: $(\text{amt} - \text{median}) / (\text{MAD} + \epsilon)$.
- **Traffic-Matched Trailing Windows**: Sub-hour and trailing windows (5m, 15m, 30m, 1h, 3h, 6h, 12h, 24h, 72h, 168h) using `closed="left"`. High-throughput devices and merchants leverage sub-hour windows to catch automated credential-stuffing bursts.
- **Velocity Burst Acceleration**: Normalized velocity ratios like `cust_burst_accel` ($(\text{cnt}_{1\text{h}} \times 24) / (\text{cnt}_{24\text{h}} + 1)$) detect abrupt spikes relative to recent personal baselines.
- **Counterparty Expansion & Novelty**: Tracks real-time account takeover signals (`is_new_device_for_customer`, `is_new_location_for_customer`) and shared device concentration (`device_id_nunique_customer_for_device_prior`).

---

### 3. Model Architecture: Dual-Horizon Tree-Neural Hybrid
Empirical audits revealed an acute summer regime shift: modern attacks shifted heavily toward device-velocity bursts rather than customer-amount anomalies. To adapt without discarding rare historical categoricals, we deployed a hybrid ensemble:

- **Dual-Horizon LightGBM Trees**:
  - *Full-Train Horizon (Jan 1 – Jul 15)*: 882 boosting rounds, `num_leaves=127`, `lr=0.02`, `feature_fraction=0.85`, `min_data_in_leaf=50`. Captures macro patterns and long-tail categories.
  - *90-Day Specialist Horizon (Apr 16 – Jul 15)*: 433 boosting rounds, `num_leaves=63`, `lr=0.02`, `feature_fraction=0.75`, `min_data_in_leaf=100`. Discards stale winter relationships and focuses on recent summer velocity attacks.
  - *Horizon Blend*: $P_{\text{LGBM}} = 0.48 \cdot P_{\text{Full}} + 0.52 \cdot P_{90\text{d}}$ (Local PR-AUC: **0.5284**).
- **Tabular ResNet (Neural Manifold Regularization)**:
  - *Architecture*: Continuous numerical features are standardized (`StandardScaler`, clipped to $[-5, 5]$); categorical features pass through learnable entity embeddings ($d \approx 1.6 \times \text{card}^{0.56}$). The combined representation feeds into 3 Residual Blocks (`LayerNorm` $\to$ `Linear(256, 512)` $\to$ `GELU` $\to$ `Dropout(0.15)` $\to$ `Linear(512, 256)` $\to$ `Dropout(0.15)` $\to$ Skip Connection).
  - *Training*: Trained for 8 epochs using AdamW (`lr=1e-3`, `wd=1e-4`) with `CosineAnnealingLR`.
  - *Tree-Neural Blend*: $P_{\text{raw}} = 0.88 \cdot P_{\text{LGBM}} + 0.12 \cdot P_{\text{ResNet}}$ (**0.5298** PR-AUC). The continuous neural manifold smooths orthogonal tree step boundaries, rescuing borderline false negatives.

---

### 4. Post-Processing: Temporal Entity Prediction Propagation
Fraud in payment rails is inherently bursty. Analysis revealed strong empirical sibling clustering: $P(\text{sibling fraud} \mid \text{current fraud})$ is **3.87x baseline** on devices and **2.63x** on customers. To reflect this, we apply an unsupervised leave-one-out (LOO) temporal diffusion process over a sliding $\pm 60$-minute window ($|t_j - t_i| \le 30$ min, $j \neq i$):

$$P_{\text{final}}(i) = (1 - w) \cdot P_{\text{raw}}(i) + w \cdot \frac{1}{2} \left[ \bar{P}_{\text{LOO, cust}}(i) + \bar{P}_{\text{LOO, dev}}(i) \right]$$

with $w = 0.50$. Transactions without in-window siblings (singletons, 95.3% of test rows) pass through completely unchanged; the blend refines only the 4.7% of transactions occurring in bursts. The operation touches zero labels and fits zero test parameters. This elevates high-confidence burst clusters and suppresses isolated noise, driving local PR-AUC to **0.5359** and Kaggle Public LB to **0.56548**.

---

### 5. Validation Strategy, Technical Decisions & Results
- **Validation Discipline**: Evaluated on a chronological holdout tail (**July 1 – July 15**, 59,465 transactions). Seed noise was measured at $\sigma \approx 0.0020$; innovations were retained only if lifts exceeded this noise floor.
- **Key Negative Findings (Avoided Traps)**:
  - *Class Weighting*: Removing `scale_pos_weight=55` gave +0.026 LB. Weighted loss distorts ranking calibration under PR-AUC.
  - *Forward Features*: Discarded forward-looking window experiments (+0.024 local lift) to maintain strict rule compliance.

| Milestone / Iteration | Key Technical Innovation | Local PR-AUC (Jul 1-15) | Kaggle Public LB |
| :--- | :--- | :---: | :---: |
| **0. Baseline** | Raw tabular features only, default LightGBM | 0.1664 | 0.1664 |
| **1. Feature Pipeline** | Expanding baselines, robust MAD z-scores, trailing windows | 0.5090 | 0.5105 |
| **2. Objective Calibration** | Removed `scale_pos_weight` (rank optimization) | 0.5204 | 0.5395 |
| **3. Feature Pruning** | Information-density pruning to Top-160 core features | 0.5269 | — |
| **4. Dual-Horizon Trees** | 48% Full-Train (882r) + 52% 90-Day Specialist (433r) | 0.5284 | — |
| **5. Tree-Neural Fusion** | 88% Dual-Horizon LGBM + 12% Tabular ResNet | 0.5298 | — |
| **6. Temporal Propagation** | Sliding LOO temporal propagation ($\pm 60$m, $w=0.50$) | **0.5359** | **0.56548** |
