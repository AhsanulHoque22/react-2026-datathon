# Slides explained

One section per slide: what is on screen, what every number means, where it came
from, why the slide exists, and the question it invites.

`speech.md` is what you **say**. This is what you need to **know**.
Lesson references point at `docs/LESSONS.md`.

---

# 1 · Title

**On screen:** team name, the one-line positioning, the three-stage pipeline as
chips, three headline numbers.

| Element | Meaning |
|---|---|
| **0.56548** | Public leaderboard score of the submitted entry, `SANZID_champion_0.56548.csv` |
| **3.4×** | 0.56548 ÷ 0.16644 — the finished model against the raw-columns baseline |
| **0** | No external data, no pretrained models. Both were competition rules; the zero is a claim of compliance, not a boast |

**Why it exists:** the score is on screen before you speak, so the judges are not
waiting to find out whether you did well.

**Say nothing here.** Let it sit while you are introduced.

---

# 2 · We don't catch fraud. We decide who gets checked first.

**On screen:** a diagram of 262,648 transactions being scored and sorted into a
queue, with the top band marked *reviewed today*. Two stat blocks. One line
naming the metric.

| Element | Meaning |
|---|---|
| The queue diagram | The actual deliverable. You submit a score per row; the organisers sort by it and measure how much real fraud sits at the top |
| **1.76%** | Fraud rate in training — 12,886 of 731,942 rows. One in 57 |
| **0.0176** | What random ordering scores on PR-AUC. The floor. It equals the fraud rate, which is not a coincidence |
| "PR-AUC" | `sklearn.metrics.average_precision_score`, fixed by the organisers |

**Why it exists:** every later decision follows from *only the order matters*. If
a judge does not accept this framing, nothing after it lands.

**The point to land:** a fraud team reviews a few hundred alerts a day and works
down your list until home time. You are writing that list.

**Read:** Lesson 7 (the metric) · Lesson 1 (the rules behind the framing).

**Likely question — "why not accuracy?"** 98.24% of rows are legitimate, so
predicting "never fraud" is 98.24% accurate and useless. Accuracy rewards
ignoring rare events.

---

# 3 · The test period starts where our data ends

**On screen:** a timeline — a solid training block, a hard divider, a dashed test
block. Three stat blocks underneath.

| Element | Meaning |
|---|---|
| **731,942 rows, 1 Jan – 15 Jul**, labelled | What we can learn from |
| **262,648 rows, 16 Jul – 15 Sep**, blind | What we are scored on. The `fraud` column simply does not exist in this file |
| The hard divider | Zero overlap. Not a random split — a wall in time |
| **13** | Raw columns. No text, no images. The whole dataset is one flat table |
| **99.8%** | Share of test rows whose **device** already appears in training. Customer is 92.4%, merchant 99.9% |
| **5/day** | Submission cap. About ten leaderboard checks across the whole ~40-hour contest |

**Why it exists:** it converts the task from "detection" to "forecasting", which
justifies every time-based choice later — the specialist model, the validation
strategy, the staleness measurement.

**The number worth knowing cold: 99.8%.** It is the precondition for the entire
approach. Almost every test transaction belongs to a device we already have
months of history for, which is why history-based features are possible at all.
If it had been 10%, this would have been a different competition.

**Read:** Lesson 4 (the dataset).

---

# 4 · No single transaction here is suspicious

**On screen:** ten rows of a real fraud burst, timestamps and amounts in
monospace, every row flagged. Two stat blocks. A three-line punchline.

| Element | Meaning |
|---|---|
| Device **D000775** | A real device from `train.csv`, 16 May 2026 |
| Twelve transactions, seven minutes | The full burst is twelve rows; ten fit on the slide |
| **340–780 tk** | The range of the burst |
| **486 tk** | Median **legitimate** transaction across the whole training set |
| **2,310 tk** | Median **fraud** transaction |

**Why it exists:** the brief asks for one strong qualitative example. This is it,
and it is real rather than illustrative.

**Why this burst and not a bigger one:** because it defeats the obvious
heuristic. Median fraud is nearly five times median legitimate, so "large amount"
genuinely *is* a signal and a judge will assume that is what you built. This
burst runs *below* typical fraud — its largest transaction sits around the 58th
percentile of ordinary legitimate traffic. An amount filter never sees it.

**What does catch it:** twelve transactions in seven minutes, one device, one
customer, cycling through seven merchant categories and five transaction types.
That is a compromised device being drained, and it is only visible across rows.

**Read:** Lesson 4 (the data) · Lesson 5 (why every single-column signal misses
this).

**Likely question — "did your model actually catch this one?"** Answer honestly:
this is from the training period, so it illustrates the pattern the features were
built for, not a held-out prediction. What generalises is the measurement —
**21% of all fraud rows have another fraud on the same device within ±30
minutes**, across the whole training set, not cherry-picked.

---

# 5 · We stopped describing transactions. We described behaviour.

**On screen:** a 13 → 160 flow, two contrasting cards about the same transaction,
two score bars.

| Element | Meaning |
|---|---|
| **13 raw columns** | amount, hour, category, device type, etc. |
| **160 behavioural features** | Each one a comparison against that customer's, device's or merchant's **own past** |
| **6,550 tk / 98th percentile** | A real fraudulent transaction (customer C032720). Globally unremarkable — about 14,600 legitimate transactions are bigger |
| **64× normal** | The same transaction against that customer's own prior mean of 101.62 tk |
| **0.1664 → 0.5090** | Raw columns vs behavioural features, on the local holdout |

**Why it exists:** this is the single largest jump in the project, and it came
from asking a better question, not from a better model.

**The line that carries it:** the raw amount column alone scores **0.256**
univariate; the customer-relative version of the *same quantity* scores
**0.415**. Same data, better question.

**Read:** Lesson 6 (the whole feature lesson) · the MAD sidebar.

**Likely question — "why median and MAD rather than mean and standard
deviation?"** Because one legitimate large purchase destroys a mean-based score.
Add a single 40,000 tk fridge to that customer's history and the standard
z-score of the known fraud falls from **138.9 to 0.16** — invisible. The
median/MAD version goes from 693 to 408 and still catches it.

---

# 6 · The whole solution, in one pass

**On screen:** the pipeline figure full width, a formula strip, three cards.

| Element | Meaning |
|---|---|
| The figure | Four blocks: 160 features → dual-horizon LightGBM → tabular ResNet → temporal burst diffusion → decision gateway |
| **0.48 / 0.52** | Full-history model vs 90-day specialist. The specialist gets the **bigger** vote |
| **0.88 / 0.12** | Tree blend vs neural network. A plain weighted average of probabilities |
| **±30 min** | The neighbour pass window, symmetric |
| **0.5173** | What the neural network scores **alone** — worse than the trees |
| **~15 minutes on CPU** | End-to-end runtime of the submitted notebook |

**Why it exists:** the brief asks for approach and architecture, and asks you to
lead with the conclusion. This is the conclusion; slides 8–10 are the evidence.

**Careful — four blocks, five stages.** The figure merges the two tree models and
their blend into one "Dual-Horizon LightGBM" box. Lesson 8 describes five stages.
Say **four** when pointing at the screen.

**Why the neural network is there at all:** not for accuracy — it loses to the
trees on its own. It is in because it is wrong about *different rows*. Trees split
on hard thresholds; the network learns smooth representations of the categorical
attributes. 12% is where measurement put the benefit peak.

**Read:** Lesson 8 (the five stages, the two-advisor framing).

---

# 7 · Fraud has siblings

**On screen:** three lift bars, the propagation rule as a visual, a before/after
diagram of five scores.

| Element | Meaning |
|---|---|
| **1.00×** | Base rate — the reference line |
| **2.63×** | P(neighbour is fraud \| this row is fraud) for the same **customer** |
| **3.87×** | Same, for the same **device**. Measured across all 731,942 training rows |
| **0.50 + 0.50** | Final score = half its own, half the average of its neighbours |
| **±30 min** | Window, either side |
| **excluded** | A transaction is not in its own neighbour average. It cannot vouch for itself |
| **95.3%** | Test rows with no in-window neighbour — they pass through completely unchanged |
| Before/after circles | Five mildly-odd scores (~0.38–0.44) that individually flag nothing, becoming ~0.59–0.63 together |

**Why it exists:** it is the step that won the score, and the organisers explicitly
invited it — *"do groups of customers, devices and merchants form suspicious
clusters that no single transaction reveals on its own?"*

**Merchant propagation is dead** and was dropped: its lift is 0.76×, i.e. below
the base rate.

**Read:** Lesson 8 Stage 5 (the mechanism) · Lesson 3 (the compliance position).

**Likely question — "is that not leakage?"** The window is symmetric, so it reads
rows after *t*. Three answers: the model never sees it (it is arithmetic on model
outputs after inference is complete, and the rule governs engineered features);
it uses **no labels** — only predictions, entity ids and timestamps; and window
and weight were fitted on labelled windows inside `train.csv`, never on test. It
changes 4.7% of rows, we disclosed it in writing, and we submitted a strictly-past
pipeline at 0.55368 alongside.

---

# 8 · Our biggest gain was deleting one line

**On screen:** four bars showing the class-weight sweep, a "why it hurt" card, the
leaderboard gain, a closing note.

| Element | Meaning |
|---|---|
| **weight 55 → 0.5045** | `scale_pos_weight` set to the negative/positive ratio, the textbook default |
| **10 → 0.5069**, **7.4 → 0.5091** | Intermediate settings |
| **removed → 0.5116** | No weighting at all |
| **+0.026** | The gain on the public leaderboard, larger than every feature built that day combined |

**Why it exists:** it is the strongest finding in the project and it was missing
from the draft entirely.

**Why the monotonic shape matters:** every step down improves the score. There is
no optimum in the middle to find. A result that improves all the way to zero is a
*mechanism*, not a lucky measurement.

**Why it hurt:** class weighting is a tool for **deciding** — block this card,
freeze that account. We never decide; we order a queue. Weighting makes the model
flag generously, which crowds the top of the list with false alarms, and the
cleanliness of that top *is* the score.

**The uncomfortable half — say it anyway:** it was in the plan from round one,
every hyperparameter sweep held it fixed, so it went untested through most of the
competition. When it came out, every conclusion drawn under it was void and the
whole search had to be re-run.

**Read:** Lesson 9 (the emergency-room analogy) · its sidebar (ranking vs
weighting are not alternatives — ranking is the task, weighting was a switch).

---

# 9 · Seven experiments. All of them noise.

**On screen:** the bathroom-scale analogy, the noise floor as a stat block, the
acceptance rule, four rejected results as bars.

| Element | Meaning |
|---|---|
| **0.0020** | Standard deviation of the score when re-running an identical model with only the random seed changed |
| The seven | CatBoost · first hyperparameter sweep · recency weighting · graph features · stationarity fixes · ratio features · expanded windows. Every one produced a delta of 0.0015–0.002 |
| **+0.0031 / +0.0029 / +0.0023 / +0.0001** | Target encoding · learned stacker · self-training · drift normalisation |

**Why it exists:** it is the part of the work that generalises beyond a datathon,
and it is what makes every other number on the deck trustworthy.

**The point that lands:** all four rejected results are **positive**. Rejecting
something that scored negative is easy. Rejecting things that appear to help —
because they do not appear to help *enough to be sure* — is the discipline.

**Where the randomness comes from:** the model samples 85% of rows and 85% of
features when building each tree, so it is slightly different every build.

**Read:** Lesson 10.

**Likely question — "does your notebook reproduce its own score?"** The header
says 0.5359; the recorded run printed 0.5341. The gap is **0.0018 — inside the
0.0020 floor we measured**. Single-seed variation on different hardware. The
notebook reproduces the method exactly and the score to within our own measured
precision.

---

# 10 · Fraud changed shape in July. Here is what we did about it.

**On screen:** a three-cell diagnosis strip, a bordered "our fix" panel with a
48/52 split bar, two "tested and rejected" cards.

| Element | Meaning |
|---|---|
| **0.41 → 0.24** | Univariate AP of `cust_amt_robust_z` — "is this amount odd for this person?" — before and after July. Halved |
| **0.095 → 0.172** | Univariate AP of `dev_amtsum_6h` — device volume in six hours. Doubled |
| **1.5 – 1.9%** | Fraud rate, unchanged all year |
| The 48/52 bar | The fix: a second model on the last 90 days, given the bigger vote |
| **−0.0010** | Cost of dropping the decayed features — the obvious fix, tested, worse |
| **0.5449 → 0.5308** | Recency weighting at none / 60d / 30d / 14d / 7d half-lives. Monotonic loss |

**Why it exists:** it justifies the two horizons on slide 6, and it shows you
diagnosed the data rather than guessing.

**The sentence underneath it all:** fraud did not get *rarer*, it got
*different*. Old fraud was a transaction unusual for its customer. New fraud is
volume pushed through a device.

**Why dropping the dead features failed:** 0.24 is still much bigger than 0.172.
The decayed signal remained the strongest thing in the feature set, and nothing
stronger was hiding behind it.

**Why recency weighting failed:** down-weighting old rows does not make the model
younger. It discards signal while the staleness remains.

**Read:** Lesson 5 (what the signal-decay numbers mean) · Lesson 8 Stage 3.

---

# 11 · 0.56548 — 3.4× the baseline

**On screen:** a seven-bar progression chart, three cards.

| Bar | Meaning |
|---|---|
| **0.1664** | Baseline — 13 raw columns |
| **0.5090** | Behavioural features |
| **0.5204** | After removing class weighting |
| **0.5284** | Dual horizon |
| **0.5298** | + Tabular ResNet |
| **0.5359** | + Propagation (local holdout) |
| **0.56548** | Public leaderboard, the submitted entry |

The first six are **local holdout** (Jul 1–15); the last is the **public
leaderboard**. They are different measurements, which is why the last bar is
styled differently.

| Card | Meaning |
|---|---|
| Lone wolves | A first-ever transaction on a fresh device has no history and no siblings — both strongest mechanisms are blind |
| **−0.019** | AP lost over 60 days of training staleness, measured by walking the training cutoff backwards. The test window runs 62 days past training |
| 0.50–0.565 | Where all 33 active teams landed. The ceiling is a property of the data |

**Why it exists:** the brief asks for the result **and** a quick failure
analysis, and says explicitly to land the result before the cutoff.

**This is the slide you must reach.** Never cut it.

**Read:** Lesson 5 (the ceiling) · Lesson 10 (the reproduction gap).

---

# 12 · You don't review a transaction. You review an account.

**On screen:** three operational tiers, a two-line statement, the takeaway.

| Element | Meaning |
|---|---|
| Auto-block | Top of the queue — burst attacks caught as a group |
| Step-up verification | The rescued middle — rows that only look suspicious once their siblings are considered |
| Pass through | The 95.3% with no suspicious neighbours. Zero friction |
| "strictly past-only" | Every feature is a running aggregate over entity history — exactly what a streaming fraud system already maintains |

The tiers are **qualitative on purpose.** The draft had numeric thresholds
(P > 0.80, 0.35–0.80); no threshold analysis was ever run, so they were removed.

**The takeaway line:** *we won by measuring our own noise before believing our own
results.*

**Read:** Lesson 3 (the compliance framing behind "strictly past-only").

**Likely question — "is this deployable?"** The feature layer ports directly:
these are running counts and windowed sums per entity, which production systems
already keep. Two honest caveats — the neighbour pass needs a 30-minute window,
so in production it is either a short review delay or a revisable alert; and the
−0.019 staleness measurement says retraining is needed on a cadence of weeks, not
months.

---

# 13 · Appendix divider

Marks everything after it as outside the timed talk. Lists what is available:
features, models, compliance, rejected ledger. Its job is to tell judges the
detail exists without you spending time on it.

---

# 14 · Appendix A · Features

| Element | Meaning |
|---|---|
| **42 / 40 / 40** | merchant- / customer- / device-relative features. A near-balanced three-entity design — the merchant carries the *most* |
| **13** | Fan-out features — distinct counterparties per entity |
| **4** | Graph cluster-size features |
| **5** | Raw columns kept, including `account_age_days` (no signal alone, useful in combination) |
| The anatomy diagram | `dev` · `_amt_vs_recent_max` · `_5m` — entity, measurement, window. Once you can read this, you can read all 160 names |
| The mini bar chart | Median transactions per entity: device **28**, customer **10**, merchant **10** |
| Nine horizons | 5m · 15m · 30m · 1h · 3h · 6h · 12h · 24h · 72h · 168h |

**The point of the slide:** windows are matched to how often each entity actually
transacts. Devices are three times busier, so devices get 5-minute windows and
customers get 30. A 5-minute customer window would be empty almost always — a
dead column crowding out live ones, since the model only sees a random 85% of
features per tree.

**And it closes the loop with slide 4:** the 5-minute device window is what sees
twelve transactions in seven minutes.

**Read:** Lesson 6.

---

# 15 · Appendix B · Models

| Model | Configuration |
|---|---|
| Full-history trees | 882 rounds · 127 leaves · lr 0.02 · feature fraction 0.85 · min leaf 50 · all 731,942 rows · **48%** |
| 90-day specialist | 433 rounds · 63 leaves · lr 0.02 · feature fraction 0.75 · min leaf 100 · **52%** |
| Tabular ResNet | 3 residual blocks · hidden 256 · dropout 0.15 · AdamW lr 1e-3 · 8 epochs, cosine schedule · alone **0.5173** |
| No sample weighting | none 0.5449 · 60d 0.5401 · 30d 0.5361 · 14d 0.5334 · 7d 0.5308 |

**Why the specialist is shallower** (63 leaves vs 127, min leaf 100 vs 50): it
trains on a fraction of the data, so it is constrained harder to stop it
memorising.

**What the ResNet embeds:** five categorical **attributes** — merchant category,
device type, location, payment method, transaction type — plus three crosses of
them. **No entity identifiers.** The champion notebook's own header calls these
"categorical entity embeddings", which invites the wrong question; the code is
clean.

**Read:** Lesson 8.

---

# 16 · Appendix C · Compliance

Four boxes and a disclosure banner.

| Box | Contents |
|---|---|
| The rules | Only data before *t* · test rows as history, never the label · no random K-fold · behaviour not entity IDs |
| How they hold | Windows are half-open · totals subtract the current row · cluster size read before linking |
| Three assertions, all called | First transaction must be blank · no forward-looking names · no raw IDs, no label |
| Built, then deleted | Forward-looking features scored **+0.024** — our largest single gain. Deleted, unsubmitted, now blocked by assertion |

**The distinction that matters:** those three assertions run **inside the
submitted notebook**. A fourth test — perturb a later row, require that no
earlier row's features move — lives in the repository only. Do not cite it as
something a judge can find in the notebook.

**The banner** is the disclosed judgement call: the neighbour window is
symmetric, it changes 4.7% of rows, it was stated in writing, and a strictly-past
submission at 0.55368 was entered alongside.

**The reserve line if pressed hard:** the team that measured the largest
forward-looking gain of anyone and deleted it is not the team trying to get
something past you.

**Read:** Lesson 3 · Lesson 1.

---

# 17 · Appendix D · The ledger

| Left column | The full progression table, holdout and public LB side by side |
|---|---|
| The dashes | Steps verified locally **without spending a leaderboard submission** — there were only 5 a day |
| Right column | Four rejected ideas, all positive, all discarded, and the 0.0020 noise floor that killed them |
| Also dead | CatBoost · ranking objectives (−0.009) · isotonic calibration · recency weighting · graph features (net-neutral) · dropping decayed features (−0.0010) |
| The exchange rate | Local gains arrive on the leaderboard roughly **doubled** — 2.15× · 2.22× · 1.87× across three independent calibration points |

**Why the exchange rate matters:** it was used as a forecasting tool. We predicted
0.554 for one submission and got 0.55262.

**Why ranking objectives failed** — the sharpest technical answer in the deck:
PR-AUC scores *one global ranking* over all 262,648 rows, while LambdaRank
optimises NDCG *within a group*. Grouping by day taught the model within-day
sorting and discarded the cross-day ordering the metric measures. No grouping
fixes it — the metric wants a single 700k-row group, which is infeasible.

**Read:** Lesson 10 · Lesson 5.
