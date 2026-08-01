# Feasibility Study — Project 25

**Phase:** Initiation (W1–W3) · **Status:** Draft for review · **Owner:** R2 (MARL Algorithms Lead)

---

## 1. Executive verdict

**Feasible, with two scope caveats and one reframed requirement.**

| Dimension | Verdict | Note |
|---|---|---|
| Technical | ✅ Feasible | Contingent on the dual-environment split (ADR-001). Single-environment PyBullet training is **not** feasible in 15 weeks. |
| Economic | ✅ Feasible | ₹0 achievable. No line item requires payment. |
| Schedule | ⚠️ Feasible with risk | 15 weeks is adequate for N=5 and N=10. **N=20 is a stretch goal, not a commitment.** |
| Operational | ✅ Feasible | Runs on a 4-core/8 GB student laptop for N=5; Colab T4 for N=10/20. |
| Legal / licensing | ✅ Feasible | All dependencies permissively licensed (BSD-3 / Apache-2.0 / MIT). No dataset licence risk — data is self-generated. |
| Ethical / safety | ✅ Feasible | Simulation-only scope removes physical risk entirely; the residual obligation is *not overclaiming*. |

**Reframed requirement:** NFR-2 (cooperative benefit grows with N) is not a controllable engineering
outcome and is reclassified as a pre-registered research hypothesis (ADR-005).

---

## 2. Technical feasibility

### 2.1 The binding constraint: environment throughput

MARL sample complexity, not model size, is what makes or breaks this project. Realistic budget:

| Fleet size | Env steps needed (QMIX, order of magnitude) | Notes |
|---|---|---|
| N=5 | ~1–2 M | Converges reliably; CPU-feasible |
| N=10 | ~3–5 M | GPU strongly preferred |
| N=20 | ~8–15 M | Non-stationarity + credit assignment both degrade; convergence uncertain |

Now compare candidate environments:

| Environment | Realistic throughput | 2 M steps takes | Verdict |
|---|---|---|---|
| PyBullet, 20 rigid bodies, GUI off, stepped physics | ~10²–10³ steps/s | hours to ~2 days | ❌ Infeasible for the sweep |
| NumPy grid world, vectorised, no rendering | ~10⁴–10⁵ steps/s | minutes to ~1 hour | ✅ Feasible |

A single-environment design using PyBullet for training would consume the entire semester on N=5
alone and would make NFR-6 (< 3 h CPU reproduction) unachievable.

**Resolution (ADR-001).** Two artefacts, one canonical:

- `WarehouseEnv` — NumPy/Python grid world. **Canonical.** Drives 100% of training and 100% of
  evaluation. Every reported number comes from here.
- `PyBulletRenderer` — consumes a *recorded trajectory* and replays it in 3D. Participates in no
  training and no metric. Because it only replays, it cannot silently disagree with the env.

This satisfies the "PyBullet simulation with 3D visualisation demo" deliverable (OBJ-6, EO-4)
without paying the physics cost per training step. It is also standard practice in the MARL
literature the project cites (SMAC, RWARE both train on abstracted dynamics).

*Honest limitation to document in the model card:* the learned policy is validated against
grid-discretised kinematics, not rigid-body dynamics. Wheel slip, inertia, and continuous-time
collision are out of scope (§3.2 of the brief) and would require sim-to-real work not attempted here.

### 2.2 Algorithm implementation: epymarl vs. custom PyTorch

| Criterion | epymarl | Custom PyTorch QMIX |
|---|---|---|
| Maintenance status | Effectively unmaintained; pinned to legacy `gym`, `sacred`, `protobuf` | N/A |
| Python 3.11 install risk | High — dependency resolution conflicts with a modern torch | None |
| Lines to write | ~0 (but ~200+ lines of env adapter/config plumbing) | ~300 |
| ONNX export path | Awkward — controller/runner wrappers obstruct clean tracing | Direct: agent net is a plain `nn.Module` |
| Viva defensibility | Weak — "the library did it" | Strong — every component authored |
| Debuggability at N=20 | Poor — failures surface inside framework internals | Good |

QMIX itself is genuinely small: per-agent GRU Q-network, a hypernetwork mixer whose weights are
passed through `abs()` to enforce ∂Q_tot/∂Q_a ≥ 0, an episode replay buffer, a target network, and
a TD loss on Q_tot. That is ~300 lines with tests.

**Verdict (ADR-002): custom PyTorch is primary.** epymarl is retained only as a cross-check oracle
if our QMIX underperforms IQL at N=5 (which would indicate an implementation bug, not a finding).

### 2.3 Safety layer feasibility

The corridor-exclusion constraint is *state-local and deterministic*: given the corridor mask from
the floor-plan JSON and a robot's cell, the set of illegal actions is computable in O(1) with no
model involvement. It is therefore trivially implementable and — importantly — trivially *testable
in isolation from the policy*, which is exactly what ETH-3 requires.

Feasibility risk is not implementation but **specification**: the brief describes the layer as a
pre-action check (§4) in one place and a post-processing override (§9) in another. These have
materially different training dynamics. Resolved in ADR-003 (mask at action selection, applied
identically in training and evaluation, with a redundant post-hoc assertion in the service layer).

### 2.4 Deadlock: the real technical risk

The likeliest failure mode is not "the policy is bad" but **corridor deadlock** — two robots in a
one-cell-wide aisle, each blocking the other, with no action sequence resolving it. This is a
*floor-plan* property, not a learning problem: if the layout admits unresolvable configurations,
no algorithm will hit 80%.

Mitigation (W4, environment design): the floor-plan JSON is calibrated so that all aisles used for
task routing are ≥ 2 cells wide, or a swap/yield primitive exists. This must be verified during
EDA by measuring random-policy deadlock frequency **before** any training run. Listed as design
issue #1 in the W4 deliverable.

### 2.5 Team capability

Four students, one semester, no prior MARL assumed. The decomposition (env / algorithm / MLOps /
evaluation+RAI) gives each member an independently ownable vertical with a clear interface. The
riskiest role is R2 (algorithms); mitigated by implementing IQL first (W6) — IQL is a DQN per agent
and shares ~70% of its code with QMIX, so QMIX in W7 is an incremental step rather than a leap.

---

## 3. Economic feasibility

| Item | Solution | Cost |
|---|---|---|
| Dataset | Self-generated by simulation | ₹0 |
| Compute (N=5) | Student laptop CPU | ₹0 |
| Compute (N=10, 20) + Optuna | Colab / Kaggle free T4 | ₹0 |
| Libraries | PyTorch, PyBullet (BSD-3), MLflow, Optuna, Evidently, ONNX Runtime, FastAPI | ₹0 |
| Version control + CI | GitHub + Actions free tier (2,000 min/mo) | ₹0 |
| Registry | Docker Hub free tier | ₹0 |
| Hardware | Replaced entirely by simulation | ₹0 |
| **Total** | | **₹0** |

**Free-tier headroom check (the only real economic risk).** GitHub Actions' 2,000 min/month is
consumed by CI, not training. Budget: pytest + lint ≈ 3 min, 5-episode smoke run ≈ 2 min,
safety-gate suite ≈ 2 min → ~7–8 min per build. At ~30 builds/month that is ~240 min, comfortably
within quota. **Constraint: never run training in CI.** Training happens on Colab/laptop; CI only
validates. This is a hard rule, recorded in the risk register (RSK-09).

Colab free-tier session limits (~12 h, pre-emptible) mean long N=20 runs must checkpoint. Budget:
checkpoint every 50k steps and support `--resume`.

---

## 4. Schedule feasibility

| Phase | Weeks | Assessment |
|---|---|---|
| Initiation & requirements | W1–3 | ✅ Comfortable |
| Environment + EDA | W4 | ⚠️ Tight — this is the project's critical path. The env must be *correct and fast* before anything downstream. |
| Ethics & governance | W5 | ✅ Comfortable; runs partly in parallel |
| Baselines (rule-based + IQL) | W6 | ✅ Adequate, given the env is done |
| QMIX + tuning + safety + scalability | W7–10 | ⚠️ The densest block; N=20 lives here |
| Deployment | W11 | ✅ Adequate |
| Monitoring & CI/CD | W12 | ✅ Adequate (CI skeleton exists from W1) |
| RAI & closure | W13–15 | ✅ Adequate |

**Critical path:** W4 environment → W6 IQL → W7–8 QMIX N=5 → W9–10 scale-out. Any slip in W4
propagates to everything. Accordingly, W4 gets the largest share of team effort and the env is
frozen (feature-locked) at end of W4; later changes require a documented ADR because they
invalidate prior training runs.

**Descope ladder** (pre-agreed, in priority order, so descoping is a decision not a panic):
1. N=20 QMIX → report as "attempted, did not converge within budget" with the learning curve as
   evidence. N=5 and N=10 remain the core deliverable. *(Explicitly permitted by the brief's risk register.)*
2. Optuna study → reduce trial count and search only the two reward weights (the scientifically
   interesting axis) rather than the full 5-dimensional space.
3. Evidently drift dashboard → static report instead of live dashboard.

Never descoped: the safety layer, the sealed evaluation protocol, and the Responsible AI report.

---

## 5. Operational feasibility

Minimum viable machine: 4-core CPU, 8 GB RAM, ~5 GB disk — met by the specified student laptop.
Memory is dominated by the episode replay buffer; at N=20 with a large buffer this can grow
unexpectedly, so buffer size is capped by config and measured in W4.

The delivered system is a Dockerised FastAPI service taking a global state snapshot and returning a
fleet action vector, with ONNX runtime inference. Operationally simple: stateless, no database, no
GPU required at inference.

---

## 6. Legal, licensing, and data feasibility

- **Licences:** PyBullet BSD-3, PyTorch BSD-3, MLflow/Optuna/Evidently Apache-2.0, FastAPI MIT. All
  permit academic use, modification, and redistribution. No copyleft obligations.
- **Data protection:** no personal data is processed. Human workers exist in the simulation only as
  static corridor annotations in a JSON floor plan — not as tracked individuals. GDPR/DPDP
  considerations are therefore not engaged, and the report should say so explicitly rather than
  leaving it unaddressed.
- **Standards:** ISO 3691-4 is referenced as a *framing device*. The project performs no conformity
  assessment and must not imply one.

---

## 7. Safety-critical framing

Although this project is simulation-only and can injure no one, it is deliberately engineered as
though it were safety-critical, because the *architecture* is the transferable contribution.

**Principle: the safety property must not depend on the learned model.**

| Layer | Mechanism | Can the policy defeat it? |
|---|---|---|
| 1. Reward shaping (corridor-entry penalty) | Soft — aligns the policy with the constraint, improves learning | Yes, in principle |
| 2. Action masking at selection | Hard — illegal actions are removed from the action set in training *and* execution | No |
| 3. Post-hoc service-layer assertion | Hard — redundant check on the emitted action vector | No |
| 4. CI blocking gate | Process — 20 max-aggression safety episodes must show zero intrusions or the build fails | No |
| *(production, out of scope)* | Hardware E-stops, proximity sensors, formal conformity assessment | No |

Layers 2–4 are non-ML. Layer 1 exists only to reduce how often the mask must fire, and the
per-episode override count is logged as a training diagnostic (FR-3): a policy that constantly
attempts corridor entry has not learned the constraint, which signals that reward shaping needs
adjustment. This makes the safety layer double as an interpretability instrument.

**The honest boundary:** the layer guarantees the policy never *commands* corridor entry. It does
not guarantee a physical robot never *ends up* in a corridor — that would require sensing, latency
guarantees, and actuator fault handling, none of which exist here. This distinction goes in the
model card verbatim.

---

## 8. Feasibility conclusions carried into design

| # | Conclusion | Recorded as |
|---|---|---|
| 1 | Train on the NumPy grid env; PyBullet replays trajectories for the demo only | ADR-001 |
| 2 | Custom PyTorch QMIX is primary; epymarl is a cross-check oracle only | ADR-002 |
| 3 | Safety = action masking, identical in train and eval, plus redundant post-hoc assert | ADR-003 |
| 4 | Python 3.11 toolchain (pybullet has no cp313 wheels) | ADR-004 |
| 5 | NFR-2 is a pre-registered hypothesis, not an acceptance criterion | ADR-005 |
| 6 | Never run training in CI — free-tier minutes are for validation only | RSK-09 |
| 7 | Floor plan must be verified deadlock-tolerant during W4 EDA, before training | W4 design issue #1 |
| 8 | Environment feature-freeze at end of W4; later changes require an ADR | Milestone plan, M2 gate |
| 9 | N=20 is a stretch goal with a pre-agreed descope path | Risk register RSK-01 |
