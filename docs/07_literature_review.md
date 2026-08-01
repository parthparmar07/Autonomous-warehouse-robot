# Literature Review — Five Supporting Papers

**Phase:** Initiation (W1–W3) · **Status:** Draft for review · **Owner:** R2 + R4

PDFs are in [`papers/`](../papers/). Every paper below was selected because it settles or challenges
a specific project decision — not for topical adjacency. Each entry ends with **what we take** and,
where applicable, **what it changes**.

> **One paper contradicts an argument we had already written into ADR-005.** That is recorded in
> §6 rather than quietly absorbed, and the ADR has been amended.

---

## P1 — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018).
**ICML 2018.** arXiv:[1803.11485](https://arxiv.org/abs/1803.11485) ·
[`papers/01-qmix-rashid-2018.pdf`](../papers/01-qmix-rashid-2018.pdf)

**What it says.** Agents must act on local observations at execution time, but training can be
centralised where global state is available. QMIX estimates the joint action-value as a non-linear
combination of per-agent values that condition only on local observations, and *structurally*
enforces monotonicity of Q_tot in each Q_a. Monotonicity is what makes decentralised argmax
equivalent to centralised argmax — the consistency property that lets CTDE work at all. The mixing
weights are produced by a hypernetwork conditioned on global state, and non-negativity is enforced
by passing hypernetwork outputs through an absolute-value activation.

**What we take.**
- The reference architecture we implement in W7 (OBJ-1, ADR-002): per-agent recurrent Q-network,
  hypernetwork mixer, `abs()` on mixing weights, TD loss on Q_tot.
- The precise justification for **FR-2**. Our monotonicity property test (perturb one Q_a, assert
  ΔQ_tot ≥ 0) tests the paper's central structural claim, not an arbitrary invariant.
- The reason CTDE is the right frame for warehouse robots: the paper's motivating condition —
  centralised training possible in simulation, decentralised execution required in deployment —
  is exactly our §9 deployment pattern (each robot runs its own Q-network on a companion computer).

---

## P2 — Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks

Papoudakis, G., Christianos, F., Schäfer, L., & Albrecht, S. V. (2021).
**NeurIPS 2021, Datasets & Benchmarks Track.** arXiv:[2006.07869](https://arxiv.org/abs/2006.07869) ·
[`papers/02-benchmarking-marl-papoudakis-2021.pdf`](../papers/02-benchmarking-marl-papoudakis-2021.pdf)

**What it says.** A systematic comparison of three algorithm classes — independent learning
(including IQL), centralised multi-agent policy gradient, and value decomposition (including QMIX) —
across a range of cooperative tasks. It open-sources **EPyMARL** (the library named in our brief)
and two environments built specifically around *coordination under sparse rewards*, one of which is
Robotic Warehouse (RWARE), a grid warehouse with item delivery — structurally close to our domain.
Its central methodological point is that algorithm comparisons in MARL are frequently confounded by
implementation details such as parameter sharing, so those details must be reported.

**What we take.**
- **Direct justification for our baseline set (OBJ-2).** IQL vs. value decomposition is the
  established axis of comparison, not one we invented.
- **Evidence that our environment family is a recognised benchmark**, supporting the brief's §6
  claim that "the environment itself is the dataset."
- **A reporting obligation.** Because implementation details confound comparisons, our MLflow
  provenance tags (FR-8) must capture parameter-sharing choice, network sizes, and training budget —
  not just the algorithm name. We will hold QMIX and IQL identical on every axis except the mixer.
- **A warning about sparse rewards.** The paper's environments were built because sparse-reward
  coordination is hard. Our reward-sparsity concern is already W4 design issue #3; this confirms it
  deserves that status.

---

## P3 — Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL

Hu, J., Jiang, S., Harding, S. A., Wu, H., & Liao, S. (2021–2023).
**ICLR Blogposts Track 2023.** arXiv:[2102.03479](https://arxiv.org/abs/2102.03479) ·
[`papers/03-monotonicity-tricks-hu-2021.pdf`](../papers/03-monotonicity-tricks-hu-2021.pdf)
*(This is the PyMARL2 paper. Note the venue: a blog track, not a main-conference paper — we cite it
for its empirical ablations, which are extensive, rather than for theoretical authority.)*

**What it says.** Two findings, both awkward for the conventional story.

1. **Implementation details dominate.** Reported gains of QMIX variants that relax the monotonicity
   constraint are "significantly affected by various code-level optimizations." With normalised
   optimisations applied, plain QMIX outperforms those variants on SMAC.
2. **Monotonicity helps rather than hurts.** "Beyond the common wisdom from these works, the
   monotonicity constraint can improve sample efficiency" in purely cooperative tasks, with a
   theoretical argument for why.

**What we take.**
- **A concrete, named risk for ADR-002.** If our from-scratch QMIX underperforms IQL, the most
  likely cause is a missing implementation trick, not a scientific finding. This sharpens RSK-05
  from "cross-check against epymarl" to a specific checklist to implement and record: reward
  standardisation/normalisation, gradient clipping, Adam vs. RMSProp, ε-anneal schedule, replay
  buffer size, target-update cadence, and parameter sharing. **Action for W7:** log every one of
  these in MLflow so a later underperformance can be diagnosed rather than guessed at.
- **A correction to our own reasoning** — see §6.

---

## P4 — Safe Multi-Agent Reinforcement Learning via Shielding

Elsayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., & Feng, L. (2021).
**AAMAS 2021.** arXiv:[2101.11196](https://arxiv.org/abs/2101.11196) ·
[`papers/04-safe-marl-shielding-elsayedaly-2021.pdf`](../papers/04-safe-marl-shielding-elsayedaly-2021.pdf)

**What it says.** MARL is increasingly used in safety-critical settings requiring that *no unsafe
state is ever visited*, including during learning — a guarantee standard MARL does not provide. The
paper presents two shielding approaches: **centralised shielding** (one shield monitors the joint
action) and **factored shielding** (multiple shields, each responsible for a subset of agents, based
on a factorisation of the joint state space). Two results matter to us: both approaches guarantee
safety during learning **without compromising the quality of learned policies**, and factored
shielding **scales better in the number of agents** than centralised shielding.

**What we take.** This is the strongest external validation of [ADR-003](adr/ADR-003-safety-layer-action-masking.md).

- **Our safety layer is a factored shield in the extreme case** — one shield per robot, whose
  correctness depends only on that robot's cell and the static corridor mask. That gives us the
  right technical vocabulary for the report and locates our design in an established literature
  rather than presenting it as ad hoc.
- **It validates the train-time masking choice.** The paper's central empirical claim is that
  shielding *during learning* does not degrade policy quality. That is precisely the objection
  ADR-003 had to argue against on first principles (RSK-06: "safety overrides disrupt
  coordination"); we can now cite evidence rather than reason alone.
- **Its scalability result supports our N=5→20 study.** Factored shielding scaling better with agent
  count is a good reason to expect our per-robot mask to remain cheap at N=20 — relevant to NFR-7
  (latency), since the mask is evaluated per robot per step in the service.
- **It sets the honest boundary.** The paper guarantees safety *with respect to a specified
  property in a modelled environment*. That is exactly the scope of our claim, and exactly why
  ETH-2's non-certification statement is necessary.

---

## P5 — PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., & Choset, H. (2019).
**IEEE Robotics and Automation Letters (RA-L).** DOI:
[10.1109/LRA.2019.2903261](https://doi.org/10.1109/LRA.2019.2903261) ·
arXiv:[1809.03531](https://arxiv.org/abs/1809.03531) ·
[`papers/05-primal-mapf-sartoretti-2019.pdf`](../papers/05-primal-mapf-sartoretti-2019.pdf)

**What it says.** Multi-agent path finding is core to warehouse automation, but state-of-the-art
centralised planners scale poorly past a few hundred agents and cannot replan online at the speeds
real deployments need. PRIMAL learns **fully decentralised** policies combining RL with imitation of
an expert MAPF planner, so agents plan reactively under partial observability while exhibiting
implicit coordination. Results extend to randomised worlds with up to 1024 agents, and the learned
policy is validated in a hybrid simulation of a factory mockup with real and simulated robots. The
paper credits "careful reward shaping and environment sampling" as essential.

**What we take.**
- **Domain grounding.** It establishes decentralised learned coordination in warehouse-like grids as
  a legitimate research direction, and its centralised-planners-scale-poorly framing is the argument
  for why our rule-based greedy + A* baseline should be *expected* to degrade as N grows — which is
  the mechanism our scalability analysis (OBJ-3) is designed to measure.
- **Independent support for reward shaping as first-order.** PRIMAL names reward shaping as
  essential to making this work. That is our OBJ-5 / Optuna sensitivity study, and it justifies
  treating reward weights as a research variable rather than a hyperparameter to set once.
- **A design contrast worth documenting.** PRIMAL trains *one* policy, copied onto any number of
  agents, so it "naturally scales to different team sizes." We instead train **separately per fleet
  size** (OBJ-3). Ours is the right choice for the question we are asking — it isolates the effect
  of N on learnability, whereas a single shared policy would confound it. But PRIMAL's approach is
  the stronger *deployment* pattern, and it belongs in the model card's future-work section.
- **A caution on imitation.** PRIMAL needed expert demonstrations to make decentralised MAPF learn.
  We use no imitation. If QMIX plateaus well below 80% at N=5, warm-starting from our rule-based A*
  dispatcher is a principled, literature-backed contingency — a better response than more tuning.

---

## 6. What this review changes

Three concrete updates. This section exists so the review demonstrably feeds the project rather than
sitting beside it.

### 6.1 ADR-005's stated mechanism is contested — amended

[ADR-005](adr/ADR-005-nfr2-as-hypothesis.md) argued that H1 (the QMIX−IQL gap widens with N) might
fail, and offered as its first reason:

> "QMIX's monotonic mixing constraint limits the class of joint value functions it can represent.
> The cost of that restriction grows with the number of agents, and it is exactly the setting where
> the literature reports QMIX struggling."

**P3 provides evidence against this**, finding that the monotonicity constraint can *improve* sample
efficiency in purely cooperative tasks, and that apparent gains from relaxing it are largely
attributable to code-level optimisations.

Our warehouse task is purely cooperative — the regime P3 examines. So the argument as written is
weaker than we claimed, and possibly backwards.

**This does not change the ADR's decision**, which rests on H1 being an outcome we do not control,
not on any prediction about direction. It does change the reasoning, and the reasoning has been
amended: the representational-limitation argument is now presented as *contested, with P3 cited
against it*, and the surviving reasons (sample budget at N=20, harder credit assignment, IQL's
empirical robustness) carry the argument. The ADR now also notes that P3 makes a *widening* gap
somewhat more plausible than we had assumed.

Recording this rather than silently deleting it is the point: the pre-registration in ADR-005 is
only meaningful if we are equally willing to publish updates that cut against our earlier reasoning.

### 6.2 ADR-002 gains a concrete implementation checklist

P3's finding that code-level optimisations dominate turns "our custom QMIX might have a bug" from a
vague worry into a specific W7 checklist, all values logged to MLflow: reward normalisation,
gradient clipping, optimiser choice, ε-anneal schedule, replay buffer size, target-update cadence,
parameter sharing. Added to ADR-002 and to RSK-05's mitigation.

### 6.3 ADR-003 gains a citation and a name

Our safety layer is a **factored shield** (P4). The ADR now cites P4 for the empirical claim that
shielding during learning does not degrade policy quality — replacing a first-principles argument
with evidence, and strengthening ETH-3.

---

## 7. Coverage and gaps

| Project element | Covered by | Adequate? |
|---|---|---|
| QMIX algorithm, CTDE, monotonicity (OBJ-1, FR-1, FR-2) | P1, P3 | ✅ |
| IQL / value-decomposition comparison (OBJ-2) | P2 | ✅ |
| Warehouse grid domain, decentralised coordination | P2 (RWARE), P5 | ✅ |
| Scalability in agent count (OBJ-3) | P4, P5 | ⚠️ Partial — P5 scales one shared policy; neither studies *per-N retraining*, which is our specific question. Reasonable as a novelty claim for the report. |
| Safety layer / shielding (OBJ-4, FR-3, NFR-3, ETH-3) | P4 | ✅ |
| Reward shaping sensitivity (OBJ-5) | P3, P5 | ✅ |
| **ETH-1 — worker displacement, wages, injury rates** | **none** | ❌ **Gap** |
| **ETH-2 — ISO 3691-4 personnel protection zones** | **none** | ❌ **Gap** |
| **ETH-4 — energy per delivery, motor-power reference** | **none** | ❌ **Gap** |

The three gaps are real and expected: they are not machine-learning questions, and no arXiv paper
will close them. They need labour-economics and occupational-safety literature, the ISO standard
text itself, and an engineering reference for AMR motor power. **Sourcing these is a W5 task
(milestone M3), separate from this review.** Flagging them now rather than discovering the gap in
W13 is the reason this table exists.

---

## 8. Citation list (BibTeX-ready)

```
[P1] Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S.
     (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement
     Learning. ICML 2018. arXiv:1803.11485.

[P2] Papoudakis, G., Christianos, F., Schäfer, L., & Albrecht, S. V. (2021). Benchmarking
     Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks. NeurIPS 2021
     Datasets and Benchmarks Track. arXiv:2006.07869.

[P3] Hu, J., Jiang, S., Harding, S. A., Wu, H., & Liao, S. (2023). Rethinking the Implementation
     Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning.
     ICLR Blogposts Track 2023. arXiv:2102.03479.

[P4] Elsayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., & Feng, L. (2021).
     Safe Multi-Agent Reinforcement Learning via Shielding. AAMAS 2021. arXiv:2101.11196.

[P5] Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., & Choset, H.
     (2019). PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning.
     IEEE Robotics and Automation Letters, 4(3). doi:10.1109/LRA.2019.2903261. arXiv:1809.03531.
```
