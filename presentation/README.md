# Presentation

| File | What it is |
|---|---|
| `slides.html` | **The deck.** Open in a browser, press `F` for fullscreen. Arrow keys or click to navigate. |
| `slides.pdf` | Exported deck, 17 pages, 16:9. Regenerate with the command below. |
| `slides_draft.html` / `slides_draft.pdf` | The original draft, kept for reference. Not the deck to present. |
| `figures/` | `pipeline.jpg` — the architecture figure, recovered from the draft PDF and used on slide 6. |
| `speech.md` | **The 3-minute script.** Timed per slide, with a lesson reference for each. |
| `SLIDES_EXPLAINED.md` | What every element and number on each slide means, and the question it invites. |

`slides.html` is **fully self-contained** — the pipeline figure is embedded as a
data URI, so the single file works from a USB stick with no `figures/` folder
beside it and no network.

Regenerate the PDF after editing:

```
google-chrome --headless --no-pdf-header-footer \
  --print-to-pdf=slides.pdf --virtual-time-budget=12000 "file://$PWD/slides.html"
```

---

## The flow

The deck follows the organisers' five required sections, in their order. The
section name is the tag at the top of every slide, so a judge can see which of
the five they are in.

### 1 · Problem & framing

| # | Slide | Carries |
|---|---|---|
| 2 | We don't catch fraud, we decide who gets checked first | The job, the 1.76% base rate, PR-AUC, and that random scores 0.0176 |
| 3 | The test period starts where our data ends | Train/test split, the 2-month blind gap, 99.8% device overlap, the 5/day cap |
| 4 | No single transaction here is suspicious | **The qualitative example the brief asks for** — device D000775, a real burst from `train.csv` |

### 2 · Approach / architecture

| # | Slide | Carries |
|---|---|---|
| 5 | We described behaviour, not transactions | 13 columns → 160 features, and the 0.166 → 0.509 jump |
| 6 | The whole solution, in one pass | The pipeline figure, full width, with the exact blend weights beneath it |
| 7 | Fraud has siblings | The propagation step, the 3.87×/2.63× clustering lift, before/after diagram |

### 3 · Key technical decisions

| # | Slide | Carries |
|---|---|---|
| 8 | Our biggest gain was deleting one line | Removing class weighting, the monotonic table, +0.026 |
| 9 | Seven experiments. All of them noise. | The 0.0020 noise floor and the four positive results rejected because of it |
| 10 | In July, fraud changed shape | The regime change — why the architecture has two horizons |

### 4 · Results & leaderboard performance

| # | Slide | Carries |
|---|---|---|
| 11 | 0.56548 — 3.4× the baseline | Full progression chart, plus the two failure modes and the data ceiling |

### 5 · Impact & conclusion

| # | Slide | Carries |
|---|---|---|
| 12 | You don't review a transaction, you review an account | Operational tiering, live-pipeline relevance, the one-line takeaway |

Slides 13–17 are appendix, explicitly marked as outside the timed talk.

**Why the architecture comes before the decisions.** The brief says to lead with
the conclusion, then the evidence. Slides 5–7 show *what we built*; slides 8–10
justify *why each piece is there*. Slide 6 carries a one-line "why two horizons"
answer so it stands alone, and slide 10 then gives the measurement behind it.

## Cutting to 3 minutes

Twelve main slides in 180 seconds is ~15 seconds each. That works only because
each slide carries one idea and one or two spoken sentences. **Practise with a
timer.**

If you run long, cut in this order:

1. **Slide 3** (the gap) — fold it into slide 2 in one sentence
2. **Slide 7** (propagation detail) — the step is already visible on slide 6
3. **Slide 9** (noise floor) — painful to lose, but it survives in Q&A

Never cut slides 4, 8 or 11. Slide 4 is the qualitative example the brief asks
for, slide 8 is the strongest finding, and slide 11 is the result — the brief
says explicitly to land the result before the cutoff. Cutting any of them also
empties one of the five required sections.

## Design rules applied

- **Fixed 1600×900 stage**, scaled to fit the screen. Layout is identical on any
  projector and in the PDF — no reflow surprises on the day.
- **Minimum body text 22px at 1600px wide**, headlines 68–82px, key figures
  64–132px. Readable from the back of a room.
- **Every slide leads with its conclusion.** Headline is the claim; the slide
  body is evidence.
- Diagrams instead of paragraphs: ranked-queue diagram, train/test timeline, the
  real fraud burst, bar charts for every comparison, an architecture flow, a
  before/after propagation figure.
- White background throughout; emphasis carried by colour, not by dark panels.
  Flame orange for the key phrase, deep sky blue for secondary emphasis. One
  highlight per slide, never more.
- Only the page number in the footer — nothing competing with the content.

## Claims removed from the draft

These appeared in the draft but could not be verified against the champion
notebook, so they are gone rather than defended in front of a judge:

| Draft claim | Why it was dropped |
|---|---|
| "Sub-10ms inference SLA" | No latency measurement exists anywhere in the project |
| "Pruned from 220+ candidates" | No count in the repo produces 220 |
| "+0.14 PR-AUC" for a feature family | Not traceable to any measurement |
| "Verified across 5 seeds" | The champion is **single-seed** (SEED=42) |
| "Robust quantile scaling" | The champion standardises and clips, it does not quantile-scale |
| "~20,000 duplicate timestamps resolved" | Not verifiable from the champion |
| Numeric tier thresholds (P>0.80, 0.35–0.80) | No threshold analysis was ever run; the tiers are now qualitative |
| "12% by rank" | The champion blends **plain probabilities**, not ranks |
| "(<10ms)" on the pipeline figure | Painted out of the image — no latency measurement exists |
| "160 features pruned from 197" | Repo docs say ~197; the champion's own list is 160 |

Everything now on a slide is either printed by the champion notebook, computed
directly from `train.csv`, or recorded in the experiment log — and where a figure
comes from the log rather than the notebook, it is a measurement the team made,
not a claim about the artifact.

## The pipeline figure

Slide 6 uses the draft's own architecture blueprint (`figures/pipeline.jpg`),
recovered from the draft PDF since the source `figures/` folder was missing. Two
edits before use:

- **Cropped off the duplicated label row.** The original repeated "160 Causal
  Features / Dual-Horizon LightGBM / Tabular ResNet / Temporal Burst Diffusion"
  both above and below the diagram.
- **Painted out "(<10ms)".** No latency measurement exists anywhere in the
  project, so it cannot be defended if asked.

The exact blend weights are not in the figure, so they sit in a formula strip
directly beneath it — the figure shows the *shape* of the solution, the strip
gives the numbers.

## Still to do before the onsite round

- [ ] Export **PPTX** as well as PDF (the brief requires both formats)
- [ ] Both files on a **USB stick**, top level, clearly named
- [ ] **Email both to the organizing committee ahead of the round** — explicit
      instruction in the brief, and the one with a deadline you can miss
- [ ] Three timed run-throughs, out loud, one with a hard cutoff at 3:00
- [ ] Decide who speaks — a 3-minute talk does not survive a handover
