# ADR-001: Two environments — NumPy grid world for training, PyBullet for replay only

**Status:** Accepted (W1–W3) · **Deciders:** all four members · **Supersedes:** the brief's risk-register mitigation, which framed this as a contingency

---

## Context

The brief specifies "a custom PyBullet + grid-discretised warehouse environment" and lists a 3D
visualisation demo as a deliverable (EO-4). Read literally, that suggests one environment doing both
jobs: stepping physics for training and rendering for the demo.

MARL sample complexity makes that impossible within the semester:

| Fleet size | Env steps needed (order of magnitude) |
|---|---|
| N=5 | ~1–2 M |
| N=10 | ~3–5 M |
| N=20 | ~8–15 M |

| Environment | Throughput | Time for 2 M steps |
|---|---|---|
| PyBullet, 20 rigid bodies, GUI off | ~10²–10³ steps/s | hours to ~2 days |
| NumPy grid world, no rendering | ~10⁴–10⁵ steps/s | minutes to ~1 hour |

Training all nine (algorithm × fleet size) configurations in PyBullet would consume the entire
semester on N=5 alone, and would make NFR-6 (< 3 h CPU reproduction from a clean clone)
unachievable.

The brief's own risk register anticipates this and offers the split as a *mitigation*. We are
promoting it from contingency to baseline architecture, because treating it as a fallback invites
someone to start building the coupled version first.

## Decision

Two artefacts, one of which is canonical.

**`WarehouseEnv`** — pure NumPy/Python grid world. **Canonical.** Drives 100% of training and 100%
of evaluation. Every number in every report comes from here.

**`PyBulletRenderer`** — consumes a *recorded trajectory* (a sequence of already-decided states) and
replays it in 3D. It participates in no training step and computes no metric.

The renderer is a **replay device, not a simulator**. It has no dynamics of its own to disagree
with. This is what makes the split safe: with two live simulators you would have a consistency
problem, but a replay device can only ever show what the canonical env already produced.

**FR-7** enforces the boundary with an import-graph test: no module under `src/train/` or
`src/eval/` may import the renderer.

## Consequences

**Positive**
- Nine training configurations become feasible; NFR-6's timing target becomes achievable.
- The 3D demo deliverable is fully satisfied — the demo shows real trained-policy behaviour, just
  rendered after the fact rather than during training.
- Renderer breakage (a real risk on Windows — see ADR-004) cannot block training, evaluation, or CI.
- Matches standard practice in the MARL literature the project cites: SMAC and RWARE both train on
  abstracted dynamics rather than full physics.

**Negative / accepted**
- The policy is validated against grid-discretised kinematics, not rigid-body dynamics. Wheel slip,
  inertia, momentum, and continuous-time collision are **not** modelled. §3.2 of the brief already
  places continuous-action path planning out of scope, but this must be stated as a limitation in
  the model card, not left implicit.
- A sim-to-real deployment would require bridging that gap; the project makes no claim about it.
- Trajectory recording adds a small storage cost per demo episode (negligible; demo episodes only).

**Neutral**
- The trajectory format becomes a real interface between R1 (env) and the demo, and needs a schema.

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| PyBullet for everything | Infeasible — see the throughput table. This is the decisive constraint. |
| PyBullet for training at N=5 only, NumPy for N=10/20 | Results across fleet sizes would not be comparable, destroying the scalability analysis that is the project's signature contribution. |
| NumPy only, no 3D demo | Abandons a stated deliverable (EO-4) and the strongest presentation asset, for a saving the replay design already achieves. |
| Two live simulators kept in sync | All the cost of PyBullet plus a consistency-verification burden, for no benefit. |
