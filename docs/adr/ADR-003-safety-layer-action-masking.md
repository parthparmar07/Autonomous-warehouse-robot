# ADR-003: The safety layer is action masking at selection time, applied identically in training and evaluation

**Status:** Accepted (W1–W3) · **Deciders:** R1 + R4 (co-owners), all members · **Resolves:** an ambiguity in the brief

---

## Context

The brief describes the safety layer two different ways:

- **§4 / FR-3:** "pre-action check: block any action leading to a corridor cell"
- **§9 (deployment):** "The safety layer runs as a **post-processing step on the returned action
  vector**, overriding any action that would move a robot into a human-worker corridor."

These are not the same mechanism, and the difference is consequential.

**Post-hoc override.** The policy selects a corridor-entering action; the wrapper replaces it with a
no-op or a random legal action. The agent's *chosen* action and its *executed* action diverge. Two
problems follow:

1. **The agent cannot learn the constraint.** If the replay buffer stores the chosen action, the
   agent is credited for an action it never took, and its Q-values are trained on a fiction. If it
   stores the executed action, the agent is credited for an action it never chose, and the policy
   gradient signal is corrupted in a different way. Either way, the override count never declines —
   the policy keeps trying, forever.
2. **It destroys the training diagnostic.** FR-3 requires logging overrides "for analysis," and the
   brief's own risk register wants the override count to reveal "the policy has not learned to avoid
   corridors." Under post-hoc override that number carries no information about learning, because
   learning about the constraint is structurally impossible.

**Action masking.** Before the argmax, Q-values for corridor-entering actions are set to −∞. The
action is never selected, so chosen and executed actions are always identical. The agent optimises
within the feasible set. Masked *attempts* — the number of times a positive-Q action was masked
away — remain fully observable and are a genuine learning signal.

The subtle failure mode masking must avoid: **applying the mask only at evaluation time.** Then the
policy is trained in an unconstrained world and deployed in a constrained one, a distribution shift
at exactly the moment safety matters. The CI gate would be testing behaviour the policy never
experienced during training.

## Decision

**Three layers, of which two are hard and non-ML.**

| Layer | Mechanism | Type | Purpose |
|---|---|---|---|
| **1. Corridor-entry penalty** | Negative reward term for attempting corridor entry | Soft (learned) | Reduces how often the mask must fire; shapes the policy toward the constraint |
| **2. Action masking at selection** | Corridor-entering actions set to −∞ before argmax, **in training and evaluation identically** | **Hard (non-ML)** | The policy structurally cannot command corridor entry |
| **3. Post-hoc service assertion** | Independent re-check of the emitted action vector in the FastAPI service | **Hard (non-ML), redundant** | Defence in depth; the artefact the CI gate exercises |

Plus the process layer:

| **4. Blocking CI gate** | 20 max-aggression safety episodes, zero intrusions required | Process | Survives every model update |

Layer 3 is deliberately redundant with layer 2. Redundancy is the point: it is the layer that
verifies the *deployed artefact* rather than the training loop, and it is what ETH-3's "tested
independently of the QMIX policy" claim rests on. It is tested with an adversarial policy that
always selects corridor-entering actions — a policy that would fail catastrophically without it.

**Logged diagnostic (FR-3):** per-episode count of masked attempts, i.e. how often the unmasked
argmax would have entered a corridor. Expected to decline during training as layer 1 does its work.
A flat or rising count means reward shaping needs adjustment (RSK-06's trigger).

**Design-before-train (W5, before W7):** the layer is specified and tested before any QMIX training
begins, so no policy is ever trained in an unconstrained world.

## Consequences

**Positive**
- **Backed by the shielding literature.** Elsayed-Aly et al. (AAMAS 2021, arXiv:2101.11196, [P4])
  formalise exactly this pattern. Our per-robot corridor mask is a **factored shield** in the
  extreme case — one shield per agent, correctness depending only on that agent's cell and the
  static corridor mask. Two of their results carry directly: shielding *during learning* guarantees
  safety **without compromising learned-policy quality** (evidence against RSK-06, which we
  previously had to argue against on first principles alone), and factored shielding **scales better
  in agent count** than centralised shielding (supports NFR-7 latency at N=20). Using their
  terminology also locates our design in an established literature rather than presenting it as ad hoc.
- Chosen action ≡ executed action, so the replay buffer is coherent and there is no train/execute mismatch.
- The safety property holds by construction, independent of model quality — it cannot be unlearned,
  and it holds even for a randomly initialised network. This is exactly the ETH-3 claim.
- The override count becomes a meaningful interpretability instrument rather than a constant.
- Training is easier: the agent never wastes exploration on actions it can never take.
- Satisfies both readings of the brief — §4's pre-action check *is* layer 2; §9's post-processing
  step *is* layer 3.

**Negative / accepted**
- Masking changes the effective action space, so reported Q-values are only meaningful within the
  feasible set. Worth a sentence in the model card.
- Layers 2 and 3 must share one implementation of "which cells are corridors" or they can drift
  apart. Mitigated by both importing the same `corridor_mask` derived from the floor-plan JSON,
  with a test asserting they agree on randomised states.
- If every action from some cell is masked, the robot has no legal move. The env must guarantee a
  no-op/stay action is always legal — an explicit env invariant, tested in W4.

**Boundary of the guarantee (must appear verbatim in the model card):**

> The safety layer guarantees the policy never *commands* an action that enters a human-worker
> corridor. It does **not** guarantee that a physical robot never *ends up* in a corridor — that
> would additionally require sensing, actuator fault handling, latency guarantees, and hardware
> emergency stops, none of which are within this project's simulation-only scope.

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| Post-hoc override only (literal §9 reading) | Breaks the learning signal and the FR-3 diagnostic; creates a chosen/executed action mismatch in the replay buffer. |
| Masking at evaluation only | Train/deploy distribution shift precisely where safety matters; the CI gate would test unseen behaviour. |
| Reward penalty only (soft constraint) | An ML guarantee. Directly contradicts ETH-3 — a large enough throughput reward can buy its way through any finite penalty. |
| Constrained MDP / Lagrangian methods | Principled, but yields a *probabilistic* constraint. NFR-3 demands zero intrusions, and the added complexity is unjustified for a constraint that is O(1) to check exactly. |
| Terminate the episode on corridor entry | Teaches avoidance but permits the intrusion to occur first — fails NFR-3 by construction. |
