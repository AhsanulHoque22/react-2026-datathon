# Overfit & Overcaffeinated
**REACT 2026 Datathon · Tabular Fraud Detection**

Public leaderboard: **0.56548** PR-AUC · 2nd of 36 · 3.4× the raw-column baseline

Catching fraud that moves — without ever seeing the future.

---

## What this is

A ranking model for transaction fraud, built for a 2-month-forward, blind test
window (no shuffled validation, no peeking at labels near test). The task is
not "is this transaction fraud" — it's "how densely does real fraud pack into
the top of a sorted list," scored by PR-AUC (average precision).

- **262,648** test transactions, 16 Jul – 15 Sep 2026
- **731,942** training transactions, 1 Jan – 15 Jul 2026
- **1.76%** fraud rate (1 in 57)
- **13** raw columns → **160** engineered features
- **~15 minutes**, single seed, CPU only, zero external data or pretrained models

## Pipeline

```
160 behavioural features
        │
        ▼
┌────────────────────────────────────┐
│  Dual-horizon LightGBM              			│
│  full-history (48%) + 90-day (52%)  			│
└────────────────────────────────────┘
        │
        ▼
  + Tabular ResNet (12%)  ← wrong about different rows than the trees
        │
        ▼
  Neighbour propagation (±30 min, same customer/device)
  score = 0.5 × own + 0.5 × neighbours' average
        │
        ▼
     ranked queue
```

### Why each piece is there

**160 behavioural features, not 13 raw columns.** Every feature compares a
transaction to that customer's, device's, or merchant's *own past* — not to
the dataset at large. A transaction that is globally unremarkable can be
wildly abnormal for one specific customer, and only the second framing catches
it. This single change was the largest gain in the project: 0.166 → 0.509
local holdout, more than every later change combined. Built on median/MAD
(`robust_z`) rather than mean/standard-deviation, so a single legitimate large
purchase doesn't blind the detector to real fraud in the same customer's
history.

**Two LightGBM models, not one.** Validating fold-by-fold across the training
period showed the most recent fold scoring visibly worse than the others.
The cause: fraud's signature changed in the last stretch of the training
window — a customer-relative amount signal roughly halved, a device-velocity
signal roughly doubled, while the overall fraud rate stayed flat. Dropping
the decayed signal made things worse (it was still the strongest single
feature available); down-weighting old rows monotonically hurt. So both
horizons are kept — full history (48%) and a 90-day specialist (52%) — with
the recent model given the larger vote, because its picture of "normal" is
current.

**A small neural network, folded in at 12%.** Alone it scores worse than the
trees (0.5173). It's included anyway because it is wrong about *different
transactions* than the trees are — trees split on hard thresholds, the
network learns softer, blurrier patterns. 12% is where testing showed the
benefit peak; past that point its weaknesses start dragging the blend down.
Built as a residual network (3 blocks, hidden 256) for the additional depth
that buys — this specific architectural choice was not run head-to-head
against a plainer network; it is the design that was built and shipped, not
the winner of a documented ablation.

**Neighbour score propagation.** Fraud clusters — a same-device neighbour of
a fraud row is ~3.87× more likely to also be fraud (same-customer: 2.63×).
Every transaction's final score is half its own, half the average of its
neighbours within a ±30 minute window (excluding itself). This is arithmetic
on model outputs — entity IDs, timestamps, and prediction scores only, no
labels — and it's the step that won the score. 95.3% of rows have no
in-window neighbour and pass through untouched.

## The two decisions that mattered most

**Deleting class weighting was the single biggest gain (+0.026 on the
leaderboard).** `scale_pos_weight` (treating each fraud row as 55 legitimate
rows) is the standard fix for class imbalance — and it was in the project
plan from round one, held fixed through every hyperparameter sweep. It is
also wrong for this task: it's a tool for *deciding* (block this card, freeze
that account), and this task never decides — it only orders a queue. Weighting
made the model flag generously, crowding the top of the ranking with false
alarms. Every step down in weight improved the score monotonically, with no
optimum in the middle — evidence of a real mechanism, not noise. Removing it
entirely voided every conclusion drawn while it was on, and the whole
hyperparameter search had to be re-run.

**Measuring our own noise floor.** Re-running an identical model with only
the random seed changed moved the score by 0.0020. Seven earlier experiments
— CatBoost, an early hyperparameter sweep, recency weighting, graph features,
stationarity fixes, ratio features, expanded windows — had each shown deltas
of 0.0015–0.002, all inside that wobble. None of them were real findings.
From that point, nothing was adopted unless measured across multiple seeds
and clearing at least 2× the noise floor. Four subsequent ideas that scored
positive (target encoding +0.0031, a learned stacker +0.0029, self-training
+0.0023, drift normalisation +0.0001) were rejected on this rule.

## Result

| Step | Local holdout | Public LB |
|---|---|---|
| Raw columns (baseline) | 0.1664 | 0.1664 |
| + Behavioural features | 0.5090 | — |
| + Class weighting removed | 0.5204 | — |
| + Dual horizon | 0.5284 | — |
| + Tabular ResNet | 0.5298 | — |
| + Neighbour propagation | 0.5359 | **0.56548** |

All 33 active teams finished between 0.50 and 0.565 — the ceiling appears to
be a property of the data, not a gap in method.

**Where it breaks:** a first-ever transaction on a fresh device has no
history and no neighbours — both core mechanisms are blind to it. The model
also loses ~0.019 AP over 60 days of training staleness, and the test window
runs 62 days past the training cutoff.

## Compliance

Every engineered feature uses only information strictly before the
transaction it describes. Enforced inside the submitted notebook by three
assertions run on every call: a forward-looking-name guard, a
first-transaction-must-be-blank check, and a ban on raw entity IDs / the
fraud label reaching the feature matrix. Raw customer, device, and merchant
IDs are used only to *group* rows for behavioural comparison — never as model
inputs.

One disclosed judgement call: the neighbour propagation window is symmetric
and technically reads ~4.7% of rows from slightly after the scored
transaction's time. It uses no labels — only model outputs, entity IDs, and
timestamps, fitted only on labelled training-period windows. A strictly
past-only variant, submitted alongside, scores 0.55368.

## Repo structure

```
notebook/         the submitted champion notebook (source of truth)
docs/
  LESSONS.md            full technical write-up, lesson by lesson
  SLIDES_EXPLAINED.md   what's on each slide and why
scripts/
  test_compliance.py    empirical past-only check (perturb a later row,
                         require no earlier row's features move)
```

**Note on figures floating elsewhere in the repo:** `STATUS.md` and
`METHOD_SUMMARY.md` describe an earlier, separate compliant pipeline
(v8c/v9c, ~197 features) — not the submitted champion (160 features). The
champion notebook is the only source that matches what was actually entered.

## Docs

- [`docs/LESSONS.md`](docs/LESSONS.md) — full technical rationale, written to
  be read cold months later: the competition rules, the metric, the dataset,
  the feature design, the model stack, and both of the findings above in
  detail.
- [`docs/SLIDES_EXPLAINED.md`](docs/SLIDES_EXPLAINED.md) — a slide-by-slide
  annotation of the presentation deck, with the reasoning and likely
  follow-up questions behind each number on screen.
