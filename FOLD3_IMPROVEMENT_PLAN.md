# Fold 3 Improvement Plan

## Executive Summary

The current best public score is approximately **0.53948 PR-AUC**. The repository documentation also contains an earlier figure of `0.5349`, but the latest Git history records `0.53948` after removing `scale_pos_weight`.

The central problem is not ordinary LightGBM under-tuning. Fold 3 represents a different fraud regime from folds 0-2:

| Window | PR-AUC | Fraud rate |
|---|---:|---:|
| Fold 0 | 0.7748 | 1.57% |
| Fold 1 | 0.7883 | 1.86% |
| Fold 2 | 0.7810 | 1.74% |
| Fold 3 | 0.5029 with the previous weighted pipeline | 1.56% |

The fraud rate remains stable, but the relationship between fraud and the available behavioral features changes sharply near July. The model learned strong January-June signals that weaken or disappear in July. Test begins on July 16, so fold 3 is much more representative of the leaderboard than the average of the four folds.

The highest-value strategy is:

1. Repair and standardize the evaluation harness.
2. Re-establish a no-class-weighting baseline on multiple recent validation windows.
3. Remove or replace features that extrapolate badly into test.
4. Build a July-specialist feature set centered on short-term velocity, diversity, acceleration, and categorical interactions.
5. Blend a full-history generalist with a recent-regime specialist only if out-of-fold validation confirms the blend.

A score in the `0.70` range is not currently supported by the evidence. A move toward `0.55-0.63` is more realistic unless a major missing fraud mechanism is discovered.

## Evidence-Based Diagnosis

### 1. July is a behavioral shift, not a prevalence shift

The raw fraud rates by validation period are:

| Period | Rows | Fraud rows | Fraud rate |
|---|---:|---:|---:|
| Jan 1-May 21 | 514,603 | 9,237 | 1.795% |
| May 21-Jun 4 | 53,200 | 834 | 1.568% |
| Jun 4-Jun 18 | 54,069 | 1,003 | 1.855% |
| Jun 18-Jul 2 | 54,448 | 945 | 1.736% |
| Jul 2-Jul 15 | 51,538 | 804 | 1.560% |

The positive rate does not collapse in fold 3. The feature-to-label relationship does.

### 2. Major signal decay occurs in July

Measured changes from the signal-decay audit include:

| Signal | Earlier recall | July recall |
|---|---:|---:|
| Amount greater than 3000 | approximately 54% | approximately 29% |
| Night transaction | approximately 37% | approximately 16% |
| New device for customer | approximately 58% | approximately 29% |
| Customer robust amount anomaly | approximately 41-54% | approximately 31% |
| Customer log amount anomaly | approximately 43-49% | approximately 21% |

The signal does not usually reverse direction. It becomes weaker and less concentrated. This explains why simply changing regularization or increasing model capacity has limited value.

### 3. Raw amount behavior changes

The median fraudulent amount changes substantially:

| Period | Median fraudulent amount |
|---|---:|
| Before July | approximately 2,524 BDT |
| July | approximately 1,064 BDT |

Absolute amount thresholds learned from earlier months therefore transfer poorly. Relative and recent-context representations are more appropriate.

### 4. Short-term device velocity survives the regime shift

The July univariate AP audit found that several short-window device features become stronger:

| Feature | June AP | July AP | July/June |
|---|---:|---:|---:|
| `dev_amtsum_6h` | 0.0939 | 0.1704 | 1.82x |
| `dev_cnt_6h` | 0.1328 | 0.1615 | 1.22x |
| `dev_amtsum_1h` | 0.1088 | 0.1515 | 1.39x |
| `dev_amtsum_24h` | 0.0669 | 0.1315 | 1.96x |
| `cust_amtsum_6h` | 0.0848 | 0.1369 | 1.62x |
| `cust_amtsum_24h` | 0.0443 | 0.0768 | 1.73x |

This is the strongest evidence for the next feature direction: recent burst, velocity, diversity, and acceleration features should receive priority over more expanding lifetime aggregates.

### 5. Categorical relationships change

The pre-July versus July fraud-rate correlation by category is unstable for some fields:

| Feature | Correlation of category fraud rates |
|---|---:|
| `merchant_category` | 0.61 |
| `device_type` | 0.099 |
| `location` | 0.269 |
| `payment_method` | 0.92 |
| `transaction_type` | 0.705 |

Location becomes especially useful in fold 3. For example, `LOC_033` has approximately `5.77x` fraud lift in the July 2-15 window. This suggests that time-aware categorical interactions are worth testing.

## Repository Issues To Fix First

### Stale validation artifacts

`data/processed/cv_results.csv` contains the previous weighted-model results, while `src/model.py` now trains without `scale_pos_weight`. The documented `0.7117` mean therefore does not describe the current executable pipeline.

The current no-weighting fold-3 result is approximately `0.5116 +/- 0.0020` across seeds.

### Stale class weighting in experiment scripts

The following scripts still use `scale_pos_weight` even though the production model no longer does:

- `scripts/03_recency_experiment.py`
- `scripts/05_hparam_search.py`
- `scripts/06_catboost_cv.py`
- `scripts/07_blend_check.py`
- `scripts/09_fix_graph_drift.py`
- `scripts/11_fold3_eval.py`

Their results should not be treated as valid evidence for the current model until rerun without weighting.

### Graph component-size extrapolation

The component-size features created in `src/features.py` are monotonically increasing. They have severe train/test drift:

- Graph PSI is approximately `9.9-10.9`.
- The train-versus-test adversarial classifier reaches AUC `1.0`.
- The top adversarial features are graph component sizes by orders of magnitude.

These features may be leakage-safe but are not distribution-safe. Raw component sizes encode dataset age and grow beyond the scale observed during training.

### Validation horizon mismatch

Fold 3 predicts up to approximately 13 days ahead. The actual test period spans July 16 through September 15, approximately 62 days after the end of training. The final validation design must include shorter recent windows and should explicitly test how performance changes with forecast horizon.

### Repeated reuse of fold 3

Fold 3 has only 804 fraud examples. Repeatedly selecting features, model parameters, and training rounds against the same fold can overfit the validation window, even when seed noise is controlled.

Use several recent validation windows and preserve one final recent window as an evaluation gate.

## Prioritized Experiment Plan

## Phase 0: Make Results Trustworthy

### 0.1 Standardize model parameters

Centralize the training parameters in one helper. Every experiment must use the same current default unless it explicitly declares a change.

Required default:

```text
objective=binary
no scale_pos_weight
learning_rate=0.05
num_leaves=63
feature_fraction=0.85
bagging_fraction=0.85
min_data_in_leaf=50
```

### 0.2 Re-run the baseline

Create a fresh result file containing:

- Fold 0 through fold 3
- July 1-8 validation
- July 8-15 validation
- July 15-16 validation if sufficient examples exist
- At least three seeds per configuration
- Full PR-AUC, history PR-AUC, cold-start PR-AUC
- Precision and recall at top-k

### 0.3 Track experiment identity

Every result row should contain:

- Git commit
- Feature-set name or hash
- Model parameters
- Seed
- Training period
- Validation period
- Number of positive examples
- Best iteration

### Acceptance rule

A candidate is a real improvement only if:

- Recent-window mean improves by at least `0.004`.
- At least two recent windows improve.
- No recent window falls by more than `0.003`.
- The improvement survives at least three seeds.

## Phase 1: Build a Fold-3 Error Taxonomy

Save out-of-fold predictions for fold 2 and fold 3. For every segment below, report row count, fraud count, fraud rate, PR-AUC, mean score, and recall in the top `0.5%`, `1%`, and `2%`:

- Amount decile
- Customer-relative amount ratio decile
- Customer, merchant, and device history count quantile
- Hour and day of week
- Location
- Merchant category
- Device type
- Payment method
- Transaction type
- New versus known customer-device pair
- Device velocity quantile
- Customer velocity quantile
- Location by transaction type
- Device type by transaction type

The main question is: **which July fraud groups are ranked below false positives?**

This should determine feature work instead of relying only on global feature importance.

## Phase 2: Remove or Replace Unstable Features

Run the following no-weighting ablation on all recent windows:

1. Current full feature set.
2. Drop all four raw graph component-size features.
3. Drop graph sizes and raw lifetime frequency/count features.
4. Keep only raw transaction facts, relative deviations, shares, and recent windows.
5. Add normalized graph-growth features.

Potential replacements for raw graph sizes:

- New graph nodes attached in the last 1h, 6h, 24h, and 7d.
- Component growth over the last 24h and 7d.
- Recent degree divided by lifetime degree.
- Component size divided by active nodes in a recent window.
- Number of distinct customers per device in recent windows.
- Number of distinct devices per customer in recent windows.

The desired signal is current suspicious expansion, not the absolute age of the graph.

## Phase 3: Add July-Regime Features

The next feature block should focus on short-term behavior.

### 3.1 More recent horizons

For customers, devices, and merchants, add strictly prior counts and amount sums over:

- 5 minutes
- 15 minutes
- 30 minutes
- 1 hour
- 3 hours
- 6 hours
- 12 hours
- 24 hours
- 3 days
- 7 days

The existing implementation covers `1h`, `6h`, `24h`, `72h`, and `168h`. The missing sub-hour and intermediate horizons are especially relevant for burst behavior.

### 3.2 Recent distribution features

For each entity and recent window, test:

- Recent mean amount
- Recent median amount
- Recent maximum amount
- Recent standard deviation
- Current amount divided by recent mean
- Current amount divided by recent median
- Current amount divided by recent maximum
- Current amount minus recent median
- Log-current amount minus log-recent median

### 3.3 Diversity and acceleration

Add strictly prior recent-window features for:

- Distinct customers per device
- Distinct devices per customer
- Distinct merchants per customer
- Distinct locations per customer
- Distinct transaction types per device
- Distinct locations per device
- Recent count divided by the preceding window count
- Recent amount sum divided by the preceding window amount sum
- Recent device degree divided by lifetime device degree
- Recent customer counterparties divided by lifetime counterparties

### 3.4 Sequence and repetition features

Test:

- Time to previous 2, 3, 5, and 10 transactions
- Time since the previous transaction with the same merchant
- Time since the previous transaction with the same location
- Same amount as a recent prior transaction
- Near-equal amount count in recent windows
- Amount change from the previous transaction
- Amount ratio to the previous transaction
- Count of consecutive transactions involving a device or location

These features should remain strictly prior and should never use fraud labels.

## Phase 4: Add Time-Aware Categorical Interactions

Use count or frequency encodings only, computed strictly from prior rows. Do not use per-entity fraud-rate encodings.

Priority interactions:

- `location x merchant_category`
- `location x transaction_type`
- `location x payment_method`
- `device_type x transaction_type`
- `device_type x payment_method`
- `merchant_category x transaction_type`
- Hour bucket x transaction type
- Hour bucket x device type
- Location x device type

High-priority checks from the raw analysis:

- `LOC_033 x transaction_type`
- ATM/web x transaction type
- Travel/education x payment method
- P2P/top-up x short device velocity

Use both recent-window frequency and lifetime frequency only if their drift is acceptable.

## Phase 5: Train a Recent-Regime Specialist

The first recency experiment was not fully diagnostic because it trained before the new regime was represented. Test recency adaptation with post-change data included in training.

Use a split around July 8:

- Train: all rows before July 8.
- Validate: July 8 through July 15 or July 16.

Compare:

- Full-history training without weighting.
- Recent 30-day window.
- Recent 60-day window.
- Recent 90-day window.
- Exponential decay half-life of 7 days.
- Exponential decay half-life of 14 days.
- Exponential decay half-life of 30 days.
- Exponential decay half-life of 60 days.

All variants must use no class weighting and at least three seeds.

The specialist should preferentially use:

- Short-window velocity features.
- Recent diversity features.
- Recent categorical interactions.
- Relative amount features.
- Stable raw categorical variables.

## Phase 6: Generalist/Specialist Ensemble

Train two independent models:

### Generalist

- Full available history.
- Stable relative and behavioral features.
- No raw graph component sizes.

### Specialist

- Recent history or recency weighting.
- July-oriented velocity and interaction features.
- More emphasis on recent categorical context.

Evaluate:

- Probability averaging.
- Percentile-rank averaging.
- Blend weights from `0.0` to `1.0` in increments of `0.1`.
- Conditional blending based on stable attributes such as transaction type, device type, location, or velocity availability.

Accept the blend only if it beats both component models on multiple recent windows.

## Phase 7: Re-test Alternative Models Correctly

The previous CatBoost comparison used the old weighted setup and is not final evidence for the current pipeline.

After the feature work, compare:

- LightGBM without class weighting.
- CatBoost without class weighting and without prohibited target statistics.
- XGBoost histogram trees without class weighting.
- A regularized linear model on stable transformed features.

The purpose is not to find the highest standalone score. It is to find complementary ranking errors. Use out-of-fold prediction correlation and blend performance to make the decision.

## Validation Design

### Required windows

Use both the existing folds and recent rolling windows:

- May 21-Jun 4
- Jun 4-Jun 18
- Jun 18-Jul 2
- Jul 2-Jul 15
- Jun 25-Jul 2
- Jul 2-Jul 8
- Jul 8-Jul 15

### Untouched gate

Reserve the most recent available validation slice for final model selection. Do not use it for feature discovery after the final comparison starts.

### Metrics

Primary:

- `average_precision_score`

Secondary:

- PR-AUC by history/cold-start group
- Precision at top `0.5%`
- Recall at top `1%`
- PR-AUC by amount, location, device type, and velocity segments
- Fold-to-fold spread and standard deviation

### Noise control

The measured single-fold seed standard deviation is approximately `0.0020`. Do not call a single-seed change of `0.001-0.002` an improvement.

## Expected Value and Priority

| Priority | Experiment | Expected value | Cost |
|---|---|---|---|
| P0 | Remove stale class weighting from all scripts | Makes evidence valid | Low |
| P0 | Re-run current no-weight baseline on recent windows | Establishes true baseline | Medium |
| P0 | Remove raw graph component sizes | Reduces test extrapolation risk | Low |
| P1 | Fold-3 segment error audit | Identifies missing fraud subtype | Medium |
| P1 | Sub-hour and short-window velocity features | Best evidence-backed feature family | Medium |
| P1 | Recent diversity and acceleration features | Targets surviving July behavior | Medium |
| P1 | July-trained specialist model | Direct regime adaptation | Medium |
| P1 | Generalist/specialist rank blend | Low-risk robustness | Medium |
| P2 | Time-aware categorical interactions | Captures changing subgroup behavior | Medium |
| P2 | Re-test alternative models without weighting | May provide complementary errors | High |
| P3 | Broad hyperparameter sweep | Low value before feature improvement | High |

## Do Not Spend Time On

- More tuning of `num_leaves` under the obsolete weighted setup.
- Raw absolute graph component sizes.
- Raw cumulative counts that primarily encode time.
- Selecting models from the easy-fold mean.
- Calibration for a rank metric.
- SMOTE unless a new validation result specifically justifies it.
- Per-entity historical fraud-rate encoding, given the documented competition restriction.
- Passing raw customer, merchant, device, or transaction IDs into the model.
- External data or reverse-engineering organizer logic.

## Final Submission Strategy

Before final training:

1. Select the feature set using multiple recent windows.
2. Freeze the model and blend weights.
3. Recompute all features chronologically through the test period using only non-label data.
4. Train through July 15.
5. Average predictions over multiple seeds.
6. Produce one stable generalist submission.
7. Produce one materially different specialist or specialist-blend submission.
8. Run all existing submission assertions.

The final model should prefer stationary ratios, recent-window behavior, recent diversity, and time-aware categorical context over unbounded lifetime counts.

## Relevant Files

- `PLAN.md`: project modeling plan and competition constraints.
- `STATUS.md`: current score history and experiment status.
- `src/features.py`: feature construction.
- `src/model.py`: LightGBM parameters and feature selection.
- `scripts/02_cv_eval.py`: walk-forward evaluation.
- `scripts/08_adversarial_validation.py`: train/test drift diagnostics.
- `scripts/10_signal_decay_audit.py`: temporal signal decay analysis.
- `scripts/11_fold3_eval.py`: fast fold-3 evaluator.
- `scripts/12_july_signal_hunt.py`: July univariate feature analysis.
- `scripts/13_class_weight_test.py`: class-weight experiment.
- `data/processed/feature_psi.csv`: feature drift measurements.
- `data/processed/adversarial_validation_importance.csv`: adversarial feature importance.
- `data/processed/univariate_ap_june_july.csv`: temporal univariate AP measurements.
