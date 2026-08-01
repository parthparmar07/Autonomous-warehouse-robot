# Requirements Specification — Project 25

**Phase:** Initiation (W1–W3) · **Status:** Baselined at end of W3 · **Owner:** R4, reviewed by all

> **Baseline rule.** Once this document is baselined at the end of W3, changes require an ADR and a
> note in the change log (§7). This matters because several requirements (FR-4, NFR-6) are about
> *pre-registration* — they lose their meaning if edited after seeing results.

---

## 1. Objectives

| ID | Objective | Success measure |
|---|---|---|
| OBJ-1 | Implement custom PyTorch QMIX in a grid-discretised warehouse env with 4 movement primitives and a task queue; train N=5 | Task-completion rate ≥ 80% on the sealed N=5 evaluation manifest |
| OBJ-2 | Compare QMIX vs. IQL vs. rule-based across N=5/10/20 on completion rate, delivery time, collision rate | All three metrics reported for all 9 (algorithm × N) cells with bootstrap CIs |
| OBJ-3 | Scalability analysis: train each algorithm separately per fleet size and characterise metric scaling | Scalability curves + written analysis section |
| OBJ-4 | Hard safety layer preventing corridor entry regardless of policy output | Zero policy-induced intrusions across 20 dedicated safety episodes; CI gate green |
| OBJ-5 | Optuna hyperparameter + reward-shaping sensitivity study | Completion-vs-collision Pareto frontier reported |
| OBJ-6 | Full lifecycle portfolio (MLflow, Evidently, Docker, CI gates, RAI report, model card, data card, 3D demo) | All artefacts present and reviewed |

---

## 2. Functional requirements

| ID | Requirement | Verification method | Evidence artefact |
|---|---|---|---|
| FR-1 | Each agent receives a local observation (local occupancy window, nearest-task bearing, current task assignment) and emits one discrete action from {forward, backward, left, right} via its individual Q-network. | Unit test on observation shape/content; assertion that the agent network never reads global state | `tests/test_observation_space.py` |
| FR-2 | The mixer is trained on global state (all robot positions + full task queue) and produces Q_tot that is monotonically non-decreasing in each Q_a. | Property test: perturb one Q_a upward, assert ΔQ_tot ≥ 0 over randomised inputs | `tests/test_mixer_monotonicity.py` |
| FR-3 | The safety layer overrides/masks any action moving a robot into a human-worker corridor cell, regardless of policy output, and logs every override. | Isolation test with an adversarial policy that always selects corridor-entering actions | `tests/test_safety_layer.py` |
| FR-4 | The task-queue generator is deterministic given a seed, enabling identical evaluation across runs and algorithms. | Determinism test: same seed → byte-identical queue; different seed → different queue | `tests/test_task_queue_determinism.py` |
| FR-5 | The decision service accepts a global state snapshot and returns the full fleet action vector produced by decentralised per-agent policies. | API contract test + latency test | `tests/test_service_contract.py` |
| FR-6 | The evaluation harness logs per-episode completion rate, average delivery time, collision count, and safety-override count to MLflow for every algorithm × fleet size. | Inspect MLflow run; assert all four metrics present for all 9 cells | MLflow experiment |
| **FR-7** *(added)* | The renderer replays a recorded trajectory in PyBullet and must not be reachable from any training or evaluation code path. | Import-graph test asserting no training module imports the renderer | `tests/test_render_isolation.py` |
| **FR-8** *(added)* | Every training and evaluation run records its environment-config hash, git commit, and seed to MLflow. | Assert the three tags exist on every run | MLflow run tags |

> FR-7 and FR-8 are additions arising from the feasibility study (ADR-001) and from NFR-6's
> reproducibility requirement. Without FR-8, "seeded and reproducible" is unverifiable after the fact.

---

## 3. Non-functional requirements

| ID | Category | Requirement | Acceptance criterion | Verification |
|---|---|---|---|---|
| NFR-1 | Task completion (N=5) | QMIX reaches high completion at N=5 and beats both baselines | ≥ 80% mean completion over the sealed 20-episode N=5 manifest; superiority over IQL and rule-based supported by bootstrap CIs on the paired difference | Evaluation harness + CI quality gate |
| NFR-2 | Cooperative benefit scaling | *(reclassified — see ADR-005)* Characterise the QMIX−IQL completion gap as N grows | **Hypothesis H1**, pre-registered: the gap is non-decreasing from N=5→10→20. Deliverable is the measured gap with CIs at each N; **either direction is a passing result** provided the measurement is sound. | Scalability analysis §; bootstrap CIs on paired differences |
| NFR-3 | Safety | Zero policy-induced corridor intrusions | 0 intrusions across 20 dedicated max-aggression safety episodes; blocking CI gate green on every build | `tests/test_safety_layer.py` + CI gate |
| NFR-4 | Collision rate | QMIX robot–robot collisions ≤ 50% of IQL at N=10; reward shaping demonstrably reduces collisions | Ratio ≤ 0.5 with bootstrap CI; Optuna study shows monotone effect of collision-penalty weight | Evaluation harness; Optuna study |
| NFR-5 | Scalability reporting | All three metrics for all three algorithms at N=5/10/20 with bootstrap CIs | 9 cells × 3 metrics populated; scalability section written | Evaluation report |
| NFR-6 | Reproducibility | Seeded runs; eval manifests committed before training; README reproduces headline results | < 1 h GPU / < 3 h CPU wall-clock from a clean clone; manifest commit predates first training commit in git history | Timed clean-clone run; git log check |
| **NFR-7** *(added)* | Inference latency | Fleet decision service returns an action vector fast enough for real-time control | p95 < 100 ms for N=20 on CPU | Latency test in W11 |
| **NFR-8** *(added)* | Training footprint | Total energy/CO₂e of all training runs measured and reported | CodeCarbon totals for all 9+ runs incl. Optuna trials, reported in ETH-4 | CodeCarbon logs |

**Why NFR-7 exists:** §9 claims the design "matches how the system would work on real robot
hardware." That claim is empty without a latency figure. **Why NFR-8 exists:** ETH-4 claims an
energy *benefit*; a benefit claim requires the training cost on the other side of the ledger.

---

## 4. Data requirements

| ID | Requirement |
|---|---|
| DR-1 | All training and evaluation data is generated by the simulation. No external dataset. |
| DR-2 | Task queues come from a seeded generator sampling shelf→dispatch pairs from the floor plan. |
| DR-3 | Sealed evaluation manifests (20 episodes each for N=5, 10, 20) are committed **before any training run**. Git history must show the manifest commit preceding the first training commit. |
| DR-4 | Per-step episode logs (observations, actions, rewards, safety-override flags, collision flags) are versioned via MLflow dataset tracking and linked to environment version + config. |
| DR-5 | The warehouse floor plan (shelf grid, corridor annotations, dispatch zones) is a versioned JSON config. Any change to it invalidates existing policies and requires retraining. |
| DR-6 | Raw episode logs are **not** committed to git (regenerable from seed + config + commit); only manifests, configs, and hashes are. |

**Sealing protocol (the integrity mechanism behind DR-3).**
1. W4: generate the three manifests, record their SHA-256 hashes in `data/manifests/HASHES.txt`.
2. Commit and tag `eval-manifests-sealed`.
3. Every evaluation run asserts the manifest hash matches before running; a mismatch aborts.
4. Any later change to a manifest is a protocol violation and must be disclosed in the report.

This is what makes the comparison credible: the evaluation set is fixed before anyone knows which
algorithm it favours.

---

## 5. Ethical requirements

| ID | Requirement | Acceptance criterion |
|---|---|---|
| ETH-1 | Analyse both productivity benefits and employment-displacement / wage / injury-rate implications of warehouse automation, using occupational-safety and labour-economics evidence — including evidence unfavourable to automation. | Section reviewed W13 |
| ETH-2 | Map the hard safety layer to ISO 3691-4 personnel-protection-zone requirements, with an explicit statement that simulation-based demonstration is **not** certification, and an enumeration of what production conformity would additionally require. | Framing reviewed W13 |
| ETH-3 | Document the safety layer's independence from the learned policy; verify it by a blocking CI gate; state that production deployment requires hardware-level safety stops. | Section reviewed W13; gate green on all builds |
| ETH-4 | Estimate energy saving from routing efficiency (robot-travel-metres per delivery, QMIX vs. IQL at N=10) using a literature motor-power reference, **net of** the CodeCarbon-measured training footprint. | Quantified two-sided estimate reviewed W13 |
| **ETH-5** *(added)* | State the negative-result policy: if QMIX fails to beat IQL, or H1 is contradicted, the finding is reported as measured with analysis of why. No post-hoc manifest changes, no unreported run selection. | Declared in W3, honoured through W15 |

---

## 6. Requirements traceability matrix

| Objective | Functional | Non-functional | Data | Ethical | Stakeholder | Primary evidence | Week |
|---|---|---|---|---|---|---|---|
| OBJ-1 | FR-1, FR-2, FR-8 | NFR-1, NFR-6 | DR-1, DR-2, DR-3 | ETH-5 | S2, S9 | MLflow N=5 QMIX run + eval report | W7–8 |
| OBJ-2 | FR-6, FR-8 | NFR-1, NFR-4, NFR-5 | DR-3, DR-4 | ETH-5 | S2, S9 | 9-cell comparison table | W6, W9–10 |
| OBJ-3 | FR-6 | NFR-2 (H1), NFR-5 | DR-3 | ETH-5 | S2, S9 | Scalability curves + analysis | W9–10 |
| OBJ-4 | FR-3 | NFR-3 | DR-5 | ETH-2, ETH-3 | S1, S3, S5 | Safety test suite + CI gate | W10 |
| OBJ-5 | FR-2 | NFR-4 | DR-4 | ETH-4 | S2, S3 | Optuna study + Pareto frontier | W8 |
| OBJ-6 | FR-5, FR-6, FR-7, FR-8 | NFR-6, NFR-7, NFR-8 | DR-4, DR-5, DR-6 | ETH-1…ETH-5 | all | Service, dashboards, CI, RAI report, model/data card, demo | W11–15 |

**Coverage check:** every FR and NFR maps to at least one objective and one verification artefact;
every stakeholder in the map traces to at least one requirement. No orphans.

---

## 7. Change log

| Date | Change | Rationale | ADR |
|---|---|---|---|
| W1–W3 | Added FR-7 (renderer isolation) | Dual-environment design needs an enforceable boundary | ADR-001 |
| W1–W3 | Added FR-8 (run provenance tags) | NFR-6 reproducibility is otherwise unverifiable | — |
| W1–W3 | Added NFR-7 (latency), NFR-8 (training footprint) | Substantiate the real-time claim (§9) and the energy claim (ETH-4) | — |
| W1–W3 | Reclassified NFR-2 as pre-registered hypothesis H1 | Outcome is not engineering-controllable; as an acceptance criterion it creates an incentive to fudge | ADR-005 |
| W1–W3 | Added ETH-5 (negative-result policy) | Makes the reclassification of NFR-2 binding rather than rhetorical | ADR-005 |
