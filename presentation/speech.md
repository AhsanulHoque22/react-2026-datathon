# Speech — 3-minute onsite talk

**421 words · ~2:48 at a normal speaking pace · 12 seconds of margin**

The brief is a hard stop at 3:00, so this is written to land the result at 2:36
and finish at 2:48. Practise out loud with a timer — silent reading runs about
40% faster than speaking and will lie to you.

Lesson references point at `docs/LESSONS.md`. Read the lesson before you rehearse
the slide; the script only says what fits in the time, the lesson is what lets
you answer the follow-up.

| Slide | Time | Runs to | Words |
|---|---|---|---|
| 1 · Title | — | 0:00 | walk-on |
| 2 · The problem | 16s | 0:16 | 43 |
| 3 · The gap | 10s | 0:26 | 23 |
| 4 · The burst | 20s | 0:46 | 47 |
| 5 · Features | 16s | 1:02 | 40 |
| 6 · Architecture | 22s | 1:24 | 51 |
| 7 · Regime change | 12s | 1:36 | 29 |
| 8 · Propagation | 16s | 1:52 | 39 |
| 9 · The deletion | 22s | 2:14 | 50 |
| 10 · Noise floor | 12s | 2:26 | 28 |
| 11 · Results | 18s | 2:44 | 36 |
| 12 · Close | 12s | 2:56 | 35 |

---

## Slide 1 · Title

**Say nothing.** Let it sit for two seconds while you're introduced. The score is
already on the screen.

---

## Slide 2 · The problem — 16s

> "We're not catching fraud. We're deciding who the fraud team checks **first**.
> Two hundred sixty-two thousand transactions, one-point-seven-six percent
> fraudulent. We're scored on how much real fraud lands at the **top** of that
> queue."

**Lesson 7** — the metric. Read the to-do-list and doubled-exam-marks analogies.
That lesson is also where the answer to *"why not accuracy?"* lives.

**Also Lesson 1** if you want the competition rules behind the framing.

---

## Slide 3 · The gap — 10s

> "And the test period starts the day our training data **ends**, then runs two
> months forward. So this is forecasting, not pattern-matching."

**Lesson 4** — the dataset. The 99.8% device-overlap figure is in there; it's the
fact that makes history-based features possible at all, and worth knowing if a
judge asks why you bet on them.

---

## Slide 4 · The burst — 20s

> "Here's device D-seven-seven-five. Twelve transactions in seven minutes, every
> one fraud. Look at the amounts — three-forty to seven-eighty taka. A normal
> transaction is four-eighty-six. Typical fraud is two thousand three hundred.
> **This fraud hides by being small and fast.** No single row is suspicious."

**Lesson 4** for where the numbers come from, **Lesson 5** for why every
single-column signal misses this burst.

*Don't rush this slide.* It's the qualitative example the brief explicitly asks
for, and it's the only moment the judges see real data.

---

## Slide 5 · Features — 16s

> "So we stopped describing transactions and described **behaviour**. Every
> feature compares a row to that customer's, device's or merchant's own past.
> Six and a half thousand taka is unremarkable — until you know it's
> **sixty-four times** this customer's normal."

**Lesson 6** — the whole feature lesson. Also the **MAD sidebar** if anyone asks
why you used median and MAD instead of mean and standard deviation (the fridge
example: one legitimate large purchase takes a standard z-score from 138.9 to
0.16).

---

## Slide 6 · Architecture — 22s

> "That gives 160 features feeding two gradient-boosted models — one on all
> history, one on only the last ninety days — blended forty-eight, fifty-two. A
> small neural network adds twelve percent, because it's wrong about
> **different rows** than the trees. Then a neighbour pass. Fifteen minutes on
> CPU, no pretrained models."

**Lesson 8** — the five stages, with the weight-guessing assistants and the two
advisors.

**Careful:** the figure shows **four** blocks, the lesson describes **five**
stages. The figure merges the two tree models and their blend into one box. Say
four when pointing at the screen.

---

## Slide 7 · Regime change — 12s

> "In July the fraud signature **changed** — customer signal halved, device
> velocity doubled. That's why we train a second model on recent data and give it
> the **bigger vote**."

**Lesson 5** for what the signal-decay numbers mean, **Lesson 8 Stage 3** for the
two-advisor framing.

The two rejected fixes are on the slide. If you have a spare second, add:
*"and yes, we tried dropping the dead features — it made things worse."*

---

## Slide 8 · Propagation — 16s

> "Fraud clusters. If a transaction is fraud, its neighbour on the same device is
> nearly **four times** likelier to be fraud too. So each score becomes half its
> own, half its neighbours' within thirty minutes. It uses **no labels**."

**Lesson 8, Stage 5** for the mechanism. **Lesson 3** for the compliance
position — this is the disclosed judgement call, and *"it uses no labels"* is
the phrase that starts that defence.

---

## Slide 9 · The deletion — 22s

> "Our biggest gain was **deleting one line**. Class weighting — the standard fix
> for imbalance. But we never decide, we order a queue, and weighting made the
> model flag generously, crowding our top with false alarms. Fifty-five scored
> point-five-oh-four-five. Removing it, point-five-one-one-six. Worth **plus
> zero-point-zero-two-six** on the leaderboard."

**Lesson 9** — the emergency-room analogy, and the uncomfortable half about it
sitting untested in the plan from round one.

**Lesson 9's sidebar** if a judge seems confused about ranking versus weighting
being alternatives. They aren't — ranking is the task, weighting was a switch.

---

## Slide 10 · Noise floor — 12s

> "Then we measured our **own noise**. Re-running the same model with a different
> seed moved the score by point-zero-zero-two. Seven earlier experiments had all
> been **inside** that."

**Lesson 10** — the bathroom scale. This is the slide most likely to earn a
follow-up question, and Lesson 10 also explains the notebook's 0.5341 vs 0.5359
reproduction gap using this same number.

---

## Slide 11 · Results — 18s

> "Zero-point-five-six-five-four-eight. Second of thirty-six, from a
> point-one-six-six baseline. Where it breaks: a first-ever transaction on a
> fresh device has no history and **no siblings**. And the model loses
> point-zero-one-nine over sixty days of staleness."

**Lesson 5** for the data ceiling (all 33 teams inside 0.50–0.565), **Lesson 10**
for the reproduction gap if the notebook comes up.

**This is the slide you must reach.** The brief says land the result before the
cutoff, not after.

---

## Slide 12 · Close — 12s

> "Every feature is strictly past-only, so this runs in a live pipeline. Real
> fraud teams don't review a transaction — they review an **account**. We won by
> measuring our own noise before believing our own results."

**Lesson 3** for the compliance framing behind "strictly past-only".

---


## If you are running long

Cut in this order. Each cut leaves its section still populated:

1. **Slide 3** — fold the gap into slide 2: *"…and the test period starts where our data ends."*
2. **Slide 8** — the propagation step is already visible in the slide 6 figure
3. **Slide 10** — painful, but the noise floor survives in Q&A

**Never cut 4, 9 or 11.** Slide 4 is the required qualitative example, slide 9 is
your strongest finding, slide 11 is the result.

## Saying the numbers

- **0.56548** → "zero point five six five four eight" · slowly, it's the headline
- **0.5045 / 0.5116** → "point five oh four five" and "point five one one six"
- **+0.026** → "plus zero point zero two six"
- **taka**, not "BDT" — you're speaking, not writing
- Never read a table aloud. The slide shows the numbers; you say what they mean.

## Rehearsal

- [ ] Three full run-throughs out loud, timed
- [ ] One where someone stops you dead at 3:00 — find out what you actually lose
- [ ] One with slide 3 skipped, as the recovery path if you start slow
- [ ] Decide who speaks. A 3-minute talk does not survive a handover.
- [ ] Read Lessons 7, 9 and 10 before the Q&A — they carry the three questions
      most likely to be asked
