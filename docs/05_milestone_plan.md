# Milestone & Delivery Plan — Project 25

**Phase:** Initiation (W1–W3) · **Status:** Draft for review · **Owner:** R3, agreed by all four members

---

## 1. Team roles

Four members, four independently ownable verticals with clean interfaces. Names to be filled in at
the W1 kickoff.

| Role | Owns | Interfaces it must publish | Member |
|---|---|---|---|
| **R1 — Environment & Simulation** | `WarehouseEnv`, floor-plan JSON, task-queue generator, collision detection, corridor annotations, PyBullet renderer | Observation/action/reward contract; env config hash | _TBD_ |
| **R2 — MARL Algorithms** | Rule-based dispatcher, IQL, QMIX (agent nets + mixer + replay + CTDE loop), Optuna study, ONNX export | Policy interface `act(obs, mask) -> action`; checkpoint format | _TBD_ |
| **R3 — MLOps & Deployment** | Repo scaffold, MLflow, Docker, FastAPI service, CI workflows + gates, Evidently monitoring, CodeCarbon | Run-tagging schema; service API schema | _TBD_ |
| **R4 — Evaluation & Responsible AI** | Evaluation harness, sealed manifests, bootstrap statistics, scalability analysis, safety test suite, RAI report, model card, data card | Metric definitions; manifest format | _TBD_ |

**Shared ownership:** the safety layer is co-owned by R1 (env-side masking) and R4 (test suite +
CI gate) — deliberately, so that no single person both implements and validates it.

---

## 2. Milestones

| ID | Milestone | Week | Definition of done (gate) |
|---|---|---|---|
| **M1** | Requirements baselined | end W3 | Stakeholder map, feasibility study, requirements spec + RTM, CRISP-DM adaptation, risk register, ADR-001…005 all reviewed. Repo scaffolded, venv reproducible, CI skeleton green. H1 pre-registered. |
| **M2** | **Environment frozen** ⚠️ *critical path* | end W4 | Env passes unit tests; throughput ≥ 10⁴ steps/s measured; deadlock frequency characterised and remediated; difficulty calibrated; **manifests sealed, hashed, tagged**; three design issues documented with remediation. |
| **M3** | Ethics & governance drafted | end W5 | Displacement analysis outline with sources; ISO 3691-4 mapping table; safety-layer independence design; sustainability accounting plan; data card draft. |
| **M4** | Baselines complete | end W6 | Rule-based (greedy + A*) and IQL at N=5 evaluated on the sealed manifest, MLflow-logged; IQL failure-mode analysis (deadlock patterns, collision clusters) written. |
| **M5** | QMIX converged at N=5 | end W8 | QMIX beats IQL on training seeds; Optuna study complete; **NFR-1 tested once on the sealed manifest**; learning curves logged. |
| **M6** | Safety layer verified | end W9 | 20 max-aggression safety episodes, zero intrusions; safety CI gate blocking; override-count diagnostic reported; ONNX export working. |
| **M7** | Scalability analysis complete | end W10 | All 9 cells (3 algorithms × N=5/10/20) evaluated with bootstrap CIs; scalability curves; H1 result reported whichever way it falls; N=20 either converged or documented as non-converged with evidence. |
| **M8** | Deployed | end W11 | Docker image on Docker Hub; FastAPI service passing contract + latency tests (NFR-7); 3D demo recorded; CodeCarbon totals collected. |
| **M9** | Monitored & gated | end W12 | Evidently dashboard with injected 3× high-demand scenario; PSI retraining trigger documented; CI runs pytest + smoke episodes + completion gate + safety gate, all blocking. |
| **M10** | Portfolio complete | end W15 | RAI report (ETH-1…ETH-5), model card, data card, README with verified reproduction timing, retrospective, final presentation with live demo and scalability chart. |

---

## 3. Week-by-week schedule

| Week | Focus | Key activities | Lead | Output |
|---|---|---|---|---|
| **W1** | Kickoff & scaffold | Roles agreed; repo, venv, CI skeleton; stakeholder map draft; **toolchain smoke test** (torch + pybullet import, tiny PyBullet scene) | R3 | Working repo |
| **W2** | Requirements | Feasibility study; FR/NFR/DR/ETH with acceptance criteria; RTM | R4 | Requirements draft |
| **W3** | Baseline & decisions | ADRs accepted; H1 pre-registered; CRISP-DM adaptation; risk register; **M1 gate** | R4/R3 | M1 |
| **W4** | ⚠️ Environment & EDA | `WarehouseEnv`; floor-plan JSON; task-queue generator; collision detection; corridor annotations; throughput benchmark; random-policy EDA; deadlock analysis; difficulty calibration; **seal manifests**; **M2 gate** | R1 | M2 |
| **W5** | Ethics & governance | Displacement literature review; ISO 3691-4 mapping; safety-layer design (pre-training, per ADR-003); sustainability plan; data card draft; **M3 gate** | R4 | M3 |
| **W6** | Baselines | Greedy + A* dispatcher; IQL at N=5; MLflow logging; IQL failure-mode analysis; **M4 gate** | R2 | M4 |
| **W7** | QMIX build | Agent GRU Q-nets; hypernetwork mixer with monotonicity test; episode replay buffer; target net; CTDE training loop; first N=5 run | R2 | QMIX v1 |
| **W8** | QMIX tune | Optuna: LR, mixer hidden size, γ, buffer, collision-penalty & delivery-bonus weights; Pareto frontier; convergence verified; sealed-manifest test; **M5 gate** | R2 | M5 |
| **W9** | Safety + scale-out I | Safety layer integrated and tested in isolation; 20 safety episodes; safety CI gate; ONNX export; QMIX + IQL at N=10; **M6 gate** | R1/R4 | M6 |
| **W10** | Scale-out II & analysis | N=20 attempt (checkpointed, resumable); bootstrap CIs; scalability curves; H1 test; analysis written; **M7 gate** | R2/R4 | M7 |
| **W11** | Deployment | FastAPI fleet-decision service; Docker build + push; latency test; PyBullet 3D demo recording; CodeCarbon aggregation; **M8 gate** | R3 | M8 |
| **W12** | Monitoring & CI | Evidently task-queue drift; 3× high-demand injection; PSI trigger; full CI with both blocking gates; **M9 gate** | R3 | M9 |
| **W13** | Responsible AI | ETH-1 displacement; ETH-2 ISO framing; ETH-3 safety independence; ETH-4 net energy estimate; ETH-5 negative-result disclosure | R4 | RAI report |
| **W14** | Cards & README | Model card (scalability results, safety design, simulation-only scope, limitations); data card; README with **timed** clean-clone reproduction | R3/R4 | Cards |
| **W15** | Closure | Retrospective; final presentation; live 3D demo rehearsal; scalability chart | all | M10 |

---

## 4. Critical path & float

```
W1 scaffold ──> W4 ENVIRONMENT ──> W6 IQL ──> W7-8 QMIX N=5 ──> W9-10 scale-out ──> W11 deploy
                    ^^^^^^^^^^^                                        |
                    zero float                                         v
W5 ethics ─────────────────────────────────────────────> W13 RAI report  (has float)
W12 monitoring/CI ──────────────────> W14 cards          (has float)
```

**Zero-float items:** W4 environment, W6 IQL, W7–8 QMIX N=5. A slip in any of these consumes the
project's entire buffer. **Float exists** in W5 (ethics), W11–12 (deployment/monitoring), and W14
(cards) — these can compress or run in parallel if needed.

**Buffer strategy.** W15 is deliberately light. If W10 slips, W11–12 compress into one week and W15
absorbs the overflow. If the slip exceeds one week, invoke the descope ladder.

---

## 5. Descope ladder (pre-agreed)

Invoke in order. Each step is a decision recorded in the retrospective, not an omission.

| Order | Descope | Impact | Trigger |
|---|---|---|---|
| 1 | **N=20 QMIX** → report as attempted-but-not-converged, with the learning curve as evidence | OBJ-3 partially met; explicitly permitted by the brief's risk register | End W10 with no convergence |
| 2 | **Optuna** → reduce trials; search only collision-penalty and delivery-bonus weights | OBJ-5 narrowed to the scientifically interesting axis | End W8 behind schedule |
| 3 | **Evidently** → static report instead of live dashboard | EO-5 partially met | End W12 behind schedule |
| 4 | **Docker Hub push** → local image + documented build | EO-3 partially met | W11 blocked |

**Never descoped:** the safety layer and its CI gate (NFR-3, ETH-3); the sealed evaluation protocol
(DR-3, NFR-6); the Responsible AI report (ETH-1…ETH-5). These are the project's integrity core — if
there is not enough time for them, there is not enough time for the project.

---

## 6. Weekly cadence

- **Monday:** 30-min stand-up — last week's gate status, this week's owner commitments, risk review.
- **Continuous:** all work on feature branches; PR requires CI green; the safety gate cannot be
  bypassed or marked `continue-on-error`.
- **Friday:** milestone-gate check against the definition of done; slippage triggers an explicit
  descope discussion rather than silent carry-over.
- **MLflow discipline:** no result is quoted in any document unless it has a corresponding MLflow
  run ID with env-config hash, git commit, and seed (FR-8).
