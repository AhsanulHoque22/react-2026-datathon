# Talking points — lines that prove the work is correct

Running collection for the onsite final. Each entry is a line worth saying out
loud, plus the evidence behind it so it survives a follow-up question.

Ahsanul is curating these while working through the material. Append, don't
reorder — the numbering is just arrival order, not priority.

---

## 1. We made the rule violation impossible, not just absent

> "We then added a check in the code that crashes the program if those features
> ever come back by accident."

**Why it lands:** anyone can *say* their features are compliant. This says we
made non-compliance a build failure. It moves the claim from "we were careful"
to "the code will not run otherwise" — which is the difference between a promise
and a guarantee.

**The context it belongs to:** we built forward-looking features (counts over
*[t, t+w]*, time-to-next-transaction), measured them at **+0.024 local PR-AUC —
our largest single gain of the competition** — and discarded them unsubmitted
because they read information after *t*.

**The evidence, if pushed** — corrected 9 Sep 2026 to cite the champion
notebook (the submitted artifact) rather than the repo:

*Inside the champion notebook, all three called, not merely defined:*
- `assert_strictly_past(SELECTED_FEATURES)` — fails the run if a column matches
  `FORWARD_MARKERS = ("_seconds_to_next", "_accel_", "_amt_vs_sym_",
  "_gap_fwd_vs_bwd")`. Those four names are a fossil record: you only guard
  against `_gap_fwd_vs_bwd` if that family once existed.
- `leakage_assertions(df)` — a customer's first-ever row must show NaN for
  `cust_amt_mean_prior` and must register `is_new_device_for_customer == 1`.
- `prepare_lgb_frame()` — asserts every one of
  `ID_COLS = ["customer_id", "merchant_id", "device_id", "transaction_id"]` is
  absent from the matrix, plus the label. This is Rule 4 enforced in code.

*In the repository only — do not imply a judge can find it in the notebook:*
- `scripts/test_compliance.py` — verifies pastness *empirically*: perturb a
  later row, and no earlier row's features may change. The strongest of the
  four, because it reads no column names at all.

**The strongest version of the point:** the team that measured the largest
forward-looking gain of anyone and deleted it is not the team trying to get
something past you.

---

## 2. The reproduction gap is explained by our own measurement

> "Our notebook reproduces the method exactly and the score to within our
> measured noise floor of 0.0020. The 0.0018 gap between the header and the
> recorded run is single-seed variation on different hardware — we measured that
> floor early in the competition and it governs how we read every result,
> including this one."

**Why it lands:** it turns the most awkward question a reproducibility reviewer
can ask — *"your notebook does not reproduce its own stated score"* — into a
demonstration of rigor. The discrepancy is not explained away, it is explained
*by a number we measured before we needed it.*

**The context:** the champion notebook header states local PR-AUC 0.5359; its
recorded Kaggle run printed 0.5341. Difference: 0.0018. The team's measured
seed-noise standard deviation is 0.0020.

**The evidence, if pushed:**
- The champion trains single-seed (SEED=42), pinning tree sampling, feature
  sampling, torch initialisation and numpy. Deterministic on identical hardware;
  the neural network does not reproduce bit-for-bit across CPU/GPU.
- The 0.0020 floor was established after seven consecutive experiments — CatBoost,
  the first hyperparameter sweep, recency weighting, graph features, stationarity
  fixes, ratio features, expanded windows — all produced deltas of 0.0015–0.002
  and were therefore all unmeasurable.
- That floor is why target encoding (+0.0031), a learned stacker (+0.0029),
  self-training (+0.0023) and drift normalisation (+0.0001) were all rejected
  despite being positive.

**The strongest version:** we can tell you the precision of our own instrument,
and this gap is inside it.
