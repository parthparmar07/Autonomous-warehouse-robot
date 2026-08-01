# ADR-002: Custom PyTorch QMIX as the primary implementation; epymarl as cross-check oracle only

**Status:** Accepted (W1–W3) · **Deciders:** R2 lead, all members · **Related:** the brief's risk register lists custom PyTorch as the *contingency*; this ADR inverts that

---

## Context

The brief permits either "epymarl (PyMARL2 fork)" or "a custom PyTorch implementation," listing
epymarl first and custom PyTorch as the fallback if installation proves difficult.

Assessment of epymarl as a 2026 dependency:

- Effectively unmaintained; pinned to legacy `gym` (not Gymnasium), `sacred` for configuration, and
  older `protobuf`/`numpy` majors.
- Its dependency constraints conflict with a modern PyTorch and with the MLflow/Evidently/ONNX stack
  this project must also install. Resolving that is not a one-afternoon job.
- Its runner/controller abstraction obstructs clean ONNX tracing, and ONNX export of the per-agent
  Q-networks is a required deliverable (EO-3).
- Debugging a non-converging N=20 run means debugging inside framework internals — precisely the
  situation where RSK-01 needs the team to move fast.

Against that, QMIX itself is small. The whole algorithm is:

- a per-agent recurrent Q-network (GRU over local observation + last action + agent id),
- a hypernetwork mixer taking the global state and producing mixing weights, passed through `abs()`
  to enforce ∂Q_tot/∂Q_a ≥ 0 (this is the entire monotonicity mechanism),
- an episode-level replay buffer,
- a target network,
- a TD loss on Q_tot.

That is roughly 300 lines including tests. IQL — required anyway as a baseline (OBJ-2) — shares
around 70% of it: the same agent network, replay, target network, and training loop, minus the
mixer. Building IQL in W6 therefore delivers most of QMIX's infrastructure before W7 begins.

## Decision

**Write QMIX in PyTorch ourselves. Primary, not fallback.**

epymarl is retained for exactly one purpose: if our QMIX fails to beat IQL at N=5 after tuning, that
is far more likely an implementation bug than a scientific finding, and we install epymarl in an
isolated environment to cross-check our mixer's behaviour on a small problem. It is a debugging
oracle, never a source of reported results.

Implementation order that makes this low-risk:
1. **W6** — IQL: agent network, replay buffer, target network, ε-greedy, training loop. Validated as
   a baseline in its own right.
2. **W7** — QMIX: add the hypernetwork mixer and switch the loss from per-agent Q to Q_tot. Reuses
   everything from step 1.

The monotonicity property (FR-2) gets a dedicated property test rather than being assumed:
perturb one Q_a upward over randomised inputs and assert ΔQ_tot ≥ 0.

## Consequences

**Positive**
- No dependency-resolution risk; the entire stack installs from current PyPI on Python 3.11.
- ONNX export is direct — the agent network is a plain `nn.Module` with no framework wrapper.
- Every component is authorable and defensible in the viva. For a course project this is a learning
  outcome, not just an engineering convenience.
- Failures are debuggable in our own code at N=20, which is where RSK-01 will actually bite.
- Retires one risk from the brief's register entirely (epymarl compatibility) rather than mitigating it.

**Negative / accepted**
- We own correctness. A subtly wrong mixer could produce plausible-but-wrong results — mitigated by
  the monotonicity property test, by the IQL comparison acting as a sanity floor, and by the epymarl
  oracle path.
- ~300 lines of R2's W7 budget spent on implementation rather than experiments.

**Neutral**
- We implement the QMIX paper's architecture directly, so hyperparameter guidance from the
  literature transfers.

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| epymarl as primary | Dependency conflicts with the required MLOps stack; awkward ONNX path; opaque debugging; realistically two weeks at risk on the critical path. |
| PyMARL2 upstream | Same problems, less maintained than the epymarl fork. |
| RLlib multi-agent (QMIX included) | Heavy dependency, opinionated abstractions, and its QMIX is less faithful to the paper. Overkill for a 4-action grid world. |
| Custom, but skip IQL and go straight to QMIX | Discards the shared-infrastructure ramp and the sanity floor. IQL is required as a baseline anyway. |
