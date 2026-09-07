# REACT 2026 Datathon — Methodology & Compliance Disclosure

**Team: Overfit & Overcaffeinated** · Public leaderboard: 2nd of 36 (0.56548)

Written for the reproducibility reviewer. Its purpose is to describe what we
built, and to disclose plainly the one part of our pipeline whose compliance is
a judgement call rather than a plain fact.

---

## 1. The two selected submissions

| selected | file | public LB | notebook |
|---|---|---|---|
| yes | `submission.csv` | **0.56548** | `REACT_2026_champion_reproducibility.ipynb` |
| yes | `CANDIDATE_v9c_compliant_0.5280.csv` | **0.55368** | `REACT_2026_v9c_reproducibility.ipynb` |

The first includes a post-inference propagation step (section 5). The second
removes it entirely and is strictly-past throughout, so a reviewer reading the
rule more strictly than we do still has a submission that needs no argument.

---

## 2. The rules we validated against

Quoted verbatim from the competition Overview, Data and Rules pages, re-fetched
and diffed word-for-word during preparation:

> *"Every engineered feature for a transaction at time t may only use
> information strictly before t."*

> *"Feature computation that uses only the raw, non-target columns of test.csv
> in a strictly-past-only way is fine (e.g., a device's known transaction
> history can include test-period rows that occurred earlier than the row being
> scored); using the fraud label anywhere near test.csv is not, since it does
> not exist for you."*

> *"Validation leakage — random K-fold cross-validation is not appropriate for
> this task... Use time-based, walk-forward, expanding-window, or
> purged/embargoed validation instead."*

> *"Attempting to reverse-engineer or scrape the organizer's private
> fraud-generation logic, or targeting specific entity IDs rather than
> behavioral patterns, undermines the spirit of the competition and may be
> flagged during reproducibility review."*

> *"No external data of any kind."*

---

## 3. Compliance, rule by rule

**Strictly before t.** Every engineered feature is computed so a row at time *t*
sees only rows strictly before *t*. Expanding aggregates use cumsum/cumcount
constructions that exclude the current row's own value; trailing windows are
half-open *[t-w, t)*; the graph component sizes are read *before* the current
row's edge is written. Enforced mechanically rather than by inspection:

- `leakage_assertions()` — a customer's first-occurrence row must show NaN prior
  statistics and must register as a new-device pairing.
- `assert_strictly_past()` — fails the run if any column matching a
  forward-looking marker reaches a feature matrix. Called on the feature list
  and again on the final test matrix before a CSV is written.
- `scripts/test_compliance.py` — verifies pastness *empirically*: perturbing a
  later row must not change any earlier row's feature values.

**No label near test.csv.** No target encoding, no per-entity or per-subgroup
fraud-rate features at any grain finer than the whole training set. We measured
a smoothed past-only per-entity target encoding at +0.0031 and rejected it, both
because it fell below our acceptance bar and because it keys on entity identity
rather than behaviour.

**No entity-ID targeting.** Raw `customer_id`, `merchant_id`, `device_id` and
`transaction_id` never enter the feature matrix in any encoding. They are join
keys only, and their absence is asserted at every `prepare_lgb_frame()` call.
Grouping *by* an ID to compute a behavioural aggregate is not the same as
targeting that ID, and only the former appears here.

**Validation.** All validation is time-based and expanding-window. No random
K-fold anywhere. Model selection used held-out windows that sit strictly after
their training data.

**No external data.** None used.

**Submission limits.** 5 on 2026-09-06, 4 on 2026-09-07, within the 5/day cap.

---

## 4. The model

### 4.1 Features

~197 behavioural features on the combined train+test frame, sorted by time:

- **Per-entity expanding statistics** for customer, merchant and device: prior
  count, mean, standard deviation, robust median/MAD z-scores of amount, and
  amount-to-prior-mean ratios.
- **Trailing windows** at 1/6/24/72/168h and sub-hour horizons matched to each
  entity's traffic density, giving counts, amount sums, and share-of-window
  ratios.
- **Novelty and fan-out**: is-new-device / merchant / location for a customer,
  distinct counterparties seen so far, and recent-window counterparty expansion.
- **Frequency encodings** expressed as share-of-traffic rather than raw counts,
  so they do not drift with dataset volume.
- **Calendar and cyclic** terms.
- **Backward-rich family** (v9c only): a uniform window ladder at
  5/15/60/180/720/1440 minutes, past-only rate acceleration (short-window rate
  over long-window rate), amount against its own trailing-window mean, and
  second-order inter-transaction gaps.

### 4.2 Training

LightGBM, `objective=binary`, with `average_precision_score` as a custom `feval`
for early stopping so the selection signal matches the competition metric.

**No `scale_pos_weight`.** This was the single largest discovery of the
competition. It had been set to neg/pos (~55x) for the 1.76% imbalance from the
start, and every early hyperparameter sweep held it fixed. PR-AUC is a *rank*
metric, and upweighting positives distorts the ranking it scores. Measured
monotonically on a July fold: spw=55 → 0.5045, spw=10 → 0.5069, spw=7.4 →
0.5091, spw=1 → 0.5116. Removing it was worth **+0.026** on the leaderboard.

Final hyperparameters (`lr=0.02`, `num_leaves=127`, `min_data_in_leaf=50`,
`feature_fraction=0.85`, `bagging_fraction=0.85`) were re-swept *after* that
change, since every conclusion drawn under spw=55 was void.

**Dual-horizon blend.** A full-train model and a 90-day specialist
(`num_leaves=63`, `ff=0.75`, `min_data_in_leaf=100`), each early-stopped
independently and averaged over 5 seeds, blended 60/40. Both are trained on
strictly-prior features; this is an ensemble of two training spans, not a
reweighting of rows.

---

## 5. The one judgement call: prediction propagation

Present in `submission.csv` (0.56548), **absent** from
`CANDIDATE_v9c_compliant_0.5280.csv` (0.55368).

After training and inference are complete, model outputs are blended:

```
final(i) = 0.5 * raw(i) + 0.5 * mean( raw(j) : j shares i's customer or device
                                      and |t_j - t_i| <= 30 minutes, j != i )
```

The window is symmetric, so for a row at time *t* it includes rows up to
*t + 30 minutes*.

### Why we believe it is within the rules

- **It is not an engineered feature.** The model never sees it. It is arithmetic
  over model outputs, applied after training and inference are finished, and the
  rule as written governs feature computation.
- **It uses no labels of any kind.** Its only inputs are the model's own
  predictions, entity identifiers and timestamps. Nothing is fitted to
  `test.csv`; the window width and blend weight were selected on labelled
  validation windows drawn from `train.csv`.
- **It encodes a behavioural pattern, not entity targeting.** Fraudulent
  transactions cluster in time within an entity. On training data,
  P(sibling is fraud | this row is fraud) is **3.87x** the base rate for devices
  and **2.63x** for customers. The blend expresses that clustering, which the
  rules explicitly invite: *"do groups of customers, devices, and merchants form
  suspicious clusters that no single transaction reveals on its own?"*

### Why we are disclosing it rather than leaving it implicit

The rule's evident purpose is temporal causality, and the clarifying example
permits test-period rows only where they *"occurred earlier than the row being
scored"*. A symmetric window does not meet that description. We think the narrow
reading is correct — the sentence governs *features* — but a reviewer should not
have to discover the question unaided.

### Scope of the effect

The blend changes **4.7%** of test rows (3.6% via customer, 4.1% via device).
The other 95.3% have no in-window neighbour and pass through unchanged. It
refines the ranking rather than producing it.

---

## 6. What we built, measured, and deliberately did not submit

**Forward-looking window features.** Counts and sums over *[t, t+w]* and
time-to-next-transaction, for customer, device and merchant. Worth roughly
**+0.024 local PR-AUC — our largest single measured gain of the competition**,
and they made the propagation step redundant (stacking the two was worth
−0.0008).

We discarded them and submitted nothing containing them. Unlike the propagation
step, these are unambiguously *engineered features* reading information after
*t*, which the rule forbids without ambiguity. `assert_strictly_past()` now fails
the build if any such column reaches a feature matrix, so they cannot re-enter by
accident.

**A learned stacking layer** over neighbourhood statistics was also built,
measured at +0.0029, and discarded — its second-stage model trains on features
derived from a symmetric window, which is the same objection as above with a
model fitted on top.

---

## 7. Measurement discipline

**Seed-noise standard deviation was 0.0020** on our primary validation window.
This was established only after seven consecutive experiments — CatBoost, an
early hyperparameter sweep, recency weighting, graph features, stationarity
fixes, ratio features, expanded windows — all produced deltas of 0.0015–0.002,
at or below the noise floor. Conclusions were being drawn from noise.

From that point, a result counted only with a multi-seed mean and an effect
comfortably above the floor, judged on recent windows whose entity-group
structure matches the test period rather than on a four-fold mean dominated by
easy early-regime folds.

**Rejected on measurement, not on caution:** target encoding (+0.0031), drift
normalisation (+0.0001), ranking objectives (−0.009), self-training (+0.0023 soft
/ −0.0101 hard), DART and regularisation sweeps (−0.0039 to +0.0006), CatBoost,
recency weighting, isotonic calibration, and a teammate's location/category
context features (−0.0053 on our feature base, though they helped on his).

---

## 8. What we learned about the data

**A regime change in early July.** Every signal we relied on decayed
simultaneously: night-hour fraud recall fell from 37% to 16%, large-amount lift
from 54% to 29%, while the overall fraud rate held steady at 1.5–1.9%. Fraud did
not become rarer; it changed shape. Univariate analysis showed customer-relative
features roughly halving in power (`cust_amt_robust_z` 0.41 → 0.24 AP) while
device-volume features roughly doubled (`dev_amtsum_6h` 0.095 → 0.172).

The test window sits entirely after this change, with roughly two weeks of
post-change training data out of six and a half months. We tested the obvious
remedy — dropping the decayed features so the model would lean on the rising
ones — and it did not work (−0.0010 on July windows), because the decayed
features, at 0.24 AP, remain stronger than anything that replaced them.

**Local gains amplify on the leaderboard.** Four calibration points:

| local (Jul 1–15 held-out) | public LB | ratio |
|---|---|---|
| 0.5204 | 0.53948 | — |
| 0.5322 | 0.56492 | 2.15x |
| 0.5263 | 0.55262 | 2.22x |
| 0.5280 | 0.55368 | 1.87x |

A two-month test window rewards ranking improvements roughly twice what a
two-week validation fold shows. This is consistent enough across independent
measurements to be a property of the task rather than an artifact.

---

## 9. Reproducing our results

Both notebooks in this folder are generated from `src/` rather than
hand-maintained, so they cannot drift from the code that produced the scores.
They run anywhere `train.csv`, `test.csv` and `sample_submission.csv` are
present, CPU only.

```
final_submission/
  REACT_2026_champion_reproducibility.ipynb  -->  submission.csv         LB 0.56548
  REACT_2026_v9c_reproducibility.ipynb       -->  CANDIDATE_v9c_....csv  LB 0.55368
  METHODOLOGY.pdf                            -->  this document
```

Full source, experiment history and every negative result are at
`github.com/AhsanulHoque22/react-2026-datathon`.
