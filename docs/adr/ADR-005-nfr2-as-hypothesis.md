# ADR-005: NFR-2 is reclassified from an acceptance criterion to a pre-registered hypothesis

**Status:** Accepted (W1–W3) · **Deciders:** all four members · **Related:** ETH-5 (negative-result policy)

---

## Context

The brief states NFR-2 as an acceptance criterion:

> **NFR-2 — Cooperative benefit scaling:** QMIX–IQL gap in task-completion rate **maintained or
> increased** as N scales from 5 to 10 to 20, quantifying that cooperative credit assignment becomes
> more important as the fleet grows.

There are two problems with this as an acceptance criterion.

**1. It is not an engineering outcome we control.** Whether the QMIX−IQL gap widens with fleet size
is a property of the algorithms and the environment, not of our implementation effort. We can build
QMIX correctly, tune it well, and evaluate it rigorously, and the gap can still narrow. There are
principled reasons to expect it might:

- ~~QMIX's monotonic mixing constraint limits the class of joint value functions it can represent.
  The cost of that restriction grows with the number of agents, and it is exactly the setting where
  the literature reports QMIX struggling.~~ **Amended W3 after the literature review — this argument
  is contested.** Hu et al. (arXiv:2102.03479, [P3]) find that the monotonicity constraint can
  *improve* sample efficiency in **purely cooperative** tasks, and that reported gains from
  relaxing it are largely attributable to code-level optimisations rather than the relaxation
  itself. Our warehouse task is purely cooperative — the regime they examine. The
  representational-limitation argument is therefore weaker than we first claimed, and may point the
  other way. See `07_literature_review.md` §6.1.
- The mixer's input (global state) and output space grow with N, making the credit-assignment
  problem harder to fit at the same sample budget.
- IQL, despite being theoretically unsound under non-stationarity, is empirically robust and
  sometimes degrades more gracefully than value-decomposition methods at scale.
- Our compute budget is fixed and free-tier. At N=20 both algorithms may be under-trained, which
  compresses any gap toward noise.

**The decision below does not depend on which way the gap goes** — it rests only on the outcome
being outside our control. But the correction matters and is kept visible rather than deleted: a
pre-registration is only credible if we are equally willing to publish updates that cut against our
own earlier reasoning. Net effect of [P3]: a *widening* gap (H1 supported) is somewhat more
plausible than we had assumed when this ADR was drafted.

**2. As written, it creates a bad incentive.** An acceptance criterion is something you are supposed
to *make* true. The only levers available for making this particular statement true — after seeing
results — are the reward weights, the floor plan, the episode horizon, or the evaluation manifests.
Every one of those is a form of tuning against the evaluation set (RSK-04). A requirement that can
only be satisfied by compromising the experiment's integrity is a badly-specified requirement.

This matters more here than in a typical project because the data is self-generated: nothing
physically prevents regenerating an evaluation manifest that happens to produce the desired curve.

## Decision

**Reclassify NFR-2 as pre-registered hypothesis H1.**

> **H1 (pre-registered W3, before any training run):** The QMIX−IQL task-completion-rate gap is
> non-decreasing as fleet size increases from N=5 to N=10 to N=20.

The **deliverable** is no longer "H1 is true." The deliverable is:

1. The measured gap at each fleet size, with bootstrap confidence intervals on the paired difference.
2. A stated verdict — H1 supported, contradicted, or inconclusive given the CIs.
3. An analysis of *why*, referencing the mechanisms above, whichever way it falls.

**A contradicted H1 is a passing result**, provided the measurement is sound. What would constitute
failure is: no measurement, an unsound measurement, or a measurement quietly adjusted until it
agreed with the hypothesis.

**Binding commitments** (these are what make the reclassification real rather than rhetorical):

- **ETH-5** is added to the ethical requirements: the negative-result policy is declared in W3 and
  honoured through W15.
- H1 is written down in W3, before any result exists, in this ADR and in the requirements spec.
- Evaluation manifests are sealed and hashed in W4 (DR-3), before training. Git history must show
  the `eval-manifests-sealed` tag preceding the first training commit — an examiner can verify this
  in one command.
- Sealed-manifest evaluation is run **once per algorithm × fleet size**. Iteration happens on
  training seeds only (CRISP-DM adaptation §3.2).
- If the sanity floor is breached — QMIX losing to IQL at N=5, where cooperation should clearly help
  and the problem is small enough to train properly — that is treated as a suspected implementation
  bug (RSK-05), cross-checked against the epymarl oracle (ADR-002) *before* being reported as a
  finding. This is the one legitimate reason to re-examine the pipeline, and any such re-examination
  is disclosed in the report.

## Consequences

**Positive**
- Removes the structural incentive to tune against the evaluation set — the single biggest threat to
  this project's credibility (RSK-04, scored red).
- Turns an uncontrollable requirement into a controllable one: rigorous measurement is something the
  team can actually guarantee.
- Makes the scalability analysis a genuine investigation. "Cooperative benefit does not scale as
  expected at N=20, and here is the mechanism" is a stronger and more interesting result than a
  tidy monotone curve that nobody can reproduce.
- Aligns the project with pre-registration practice from experimental science — itself a
  demonstrable learning outcome, and directly relevant to the Responsible AI theme.

**Negative / accepted**
- Deviates from the brief as literally written. Mitigated by documenting the deviation here, in the
  requirements change log (§7), and stating it openly in the final report rather than silently
  substituting our own criterion.
- The final presentation may have to show a result that is less immediately impressive. Accepted;
  the honesty is the contribution, and it should be presented as such rather than apologised for.

**Neutral**
- No change to any implementation work. The training, evaluation, and statistical machinery are
  identical either way — only the interpretation of the outcome changes.

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| Keep NFR-2 as a hard acceptance criterion | Creates an incentive to tune on the evaluation set; makes project success depend on an outcome nobody controls. |
| Drop NFR-2 entirely | Discards the project's most interesting research question. The scalability analysis is the signature contribution (§1 of the brief). |
| Weaken it to "gap ≥ 0 at all N" (QMIX merely never worse) | Still an outcome we do not control, just a lower bar. Same incentive problem, less scientific content. |
| Keep it as a criterion but allow documented failure | Effectively what we are doing — but naming it a hypothesis up front is clearer, and pre-registration is what gives the eventual result its credibility. |
