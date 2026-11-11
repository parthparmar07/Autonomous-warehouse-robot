# Model Card — Recurrent CTDE warehouse fleet policies (QMIX, VDN, IQL)

**Project 25 — Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent RL**
ATLAS SkillTech University, uGDX School of Technology
Team: Parth Parmar (2410244), Saket Sarvaiya (2408838), Swarnim Jambhrunkar (2409789)

## 1. Model details

| Item | Value |
|---|---|
| Type | Cooperative multi-agent deep Q-learning; centralised training, decentralised execution (CTDE) |
| Algorithms | **QMIX** (monotonic mixer), **VDN** (additive sum), **IQL** (independent targets, no mixer) |
| Actor (shared by all robots) | `SharedAgent` ([src/training/learner.py](../src/training/learner.py)): 2-layer MLP encoder (64) → 1-layer GRU (64) → 7 action values. Input = 348-feature egocentric observation + 7-way one-hot of the previous action. 52,359 parameters. |
| QMIX mixer (training only) | `MonotonicMixer`: hypernetworks conditioned on the global state produce mixing weights passed through `abs()`, so Q_tot is non-decreasing in every robot's Q. 157,153 parameters at N=5. |
| Action selection | Greedy over the action mask; masked actions get −∞ before the argmax |
| Training | Behaviour cloning from the centralised `NearestTaskDispatcher` planner → DAgger relabelling → online RL with masked Double-Q targets, recurrent sequence replay (burn-in 20, unroll 40) and an imitation regulariser (`bc_aux_weight` 0.25) |
| Framework | PyTorch; ONNX export with an action-parity check ([src/deployment/onnx_policy.py](../src/deployment/onnx_policy.py)) |
| Contracts | Environment `warehouse_v6_research`, observation `egocentric_range_v3_pose_feedback`, workload `mixed_stock_v1` |

## 2. Intended use

- **Intended:** research and teaching. It compares cooperative MARL credit-assignment methods in a
  simulated warehouse, and backs the project's live demo and decision service.
- **Not intended:** controlling physical robots, or any setting where people could be hurt. The
  simulator is not calibrated to real hardware, and the safety layer is not a certified safety
  function (see the [Responsible AI report](24_RESPONSIBLE_AI_AND_SUSTAINABILITY_REPORT.md), ETH-2/3).

## 3. Training data and procedure

All training data is synthetic; see the [data card](DATA_CARD.md). One policy is trained per
algorithm per fleet size, with training seed 11, on the shared recipe below. Only the demo budget
shrinks for larger fleets.

| Setting | Value |
|---|---|
| Stage / workload | C5, mixed stock (inbound put-away + outbound shipping), 600-tick episodes |
| Expert demonstrations | N=5: 60 episodes, 10 epochs, 2 DAgger rounds · N=10: 30 episodes, 8 epochs, 1 round · N=20: 20 episodes, 8 epochs, 1 round |
| Online RL | 6,000 environment steps, one update every 8 steps, ε from 0.1 to 0.05 (capped at 0.1 after BC) |
| Learner | lr 3e-4, γ 0.99, batch 8 sequences, target sync every 200 updates, gradient clip 10 (actor and mixer clipped separately) |
| Checkpoint selection | Best development evaluation (seeds 300000–300004), ranked by readiness → safety penalties → completion → throughput → energy |
| Hardware | N=5: RTX 4070 Laptop GPU · N=10/20: RTX 4050 Laptop GPU (6 GB) |

The published N=5 "outbound" checkpoints in `artifacts/warehouse_v6_research/` contain the **same
actor weights** as the mixed-stock N=5 checkpoints (same training run). They are repackaged with a
mixer sized for the outbound state; they were not trained separately.

## 4. Evaluation

### Sealed benchmark, N=5 (5 seeds, 200000–200004; earlier report)

| Policy | Completion | Mean delivery time (ticks) | Energy (EU) | Executed collisions | Avoidable depletions |
|---|---|---|---|---|---|
| Centralised planner (privileged) | 100.0% | 37.21 | 343.48 | 0 | 0 |
| IQL | 100.0% | 50.99 | 354.52 | 0 | 0 |
| VDN | 100.0% | 50.36 | 351.82 | 0 | 0 |
| QMIX | 99.0% | 46.51 | 351.12 | 0 | 0 |

### Scalability, N = 5 / 10 / 20 (20 sealed seeds, 200000–200019)

Fixed demand: 40 orders per shift at every N. Source:
`outputs/evaluation/scalability_N5_N10_N20.json` (script [evaluate_scalability.py](../scripts/evaluate_scalability.py)).
Brackets are 95% bootstrap CIs over the 20 scenarios.

| N | Policy | Completion | Delivery time (ticks) | Near-miss conflicts / episode | Executed collisions | Avoidable depletions / episode | Energy per delivery (EU) |
|---|---|---|---|---|---|---|---|
| 5 | Planner | 100% | 36.2 [34.3, 38.1] | 0.65 | 0 | 0 | 8.53 |
| 5 | IQL | 98.1% [94.5, 100] | 44.5 [40.2, 49.6] | 32.2 | 0 | 0 | 8.77 |
| 5 | QMIX | **99.8%** [99.3, 100] | 46.6 [41.1, 52.7] | 29.7 | 0 | 0 | 8.74 |
| 10 | Planner | 100% | 28.3 [27.8, 28.9] | 1.15 | 0 | 0 | 9.41 |
| 10 | IQL | 96.8% [94.3, 98.8] | 56.9 [50.4, 64.7] | 97.4 | 0 | 0 | 11.06 |
| 10 | QMIX | **97.8%** [95.9, 99.3] | **42.1** [40.1, 44.2] | 102.1 | 0 | 0.25 | 11.29 |
| 20 | Planner | 100% | 29.1 [28.3, 30.1] | 1.95 | 0 | 0 | 10.89 |
| 20 | IQL | **95.3%** [92.1, 97.6] | 91.5 [79.5, 105.0] | 177.2 | 0 | 0.30 | 15.31 |
| 20 | QMIX | 92.1% [88.3, 95.4] | 81.2 [71.6, 92.0] | 182.2 | 0 | 0.55 | 15.40 |

**Paired QMIX − IQL completion difference** (positive means QMIX is better):

| N | QMIX − IQL |
|---|---|
| 5 | +1.6 pp [−0.5, +5.5] |
| 10 | +1.0 pp [−2.1, +4.3] |
| 20 | −3.1 pp [−7.3, +1.0] |

**What the results show.**

1. **N=5: OBJ-1 / NFR-1 met.** QMIX completes 99.8% on the 20-seed sealed manifest, well above the
   80% target. Its advantage over IQL is not statistically resolved (the CI includes 0).
2. **N=10: the clearest QMIX advantage is speed, not completion.** QMIX delivers in 42.1 ticks vs
   56.9 for IQL, and the intervals do not overlap. Completion is similar.
3. **NFR-2 / H1, the cooperative-gap hypothesis.** The QMIX−IQL gap goes from +1.6 to +1.0 to
   −3.1 pp, so it does **not** grow with N. All three intervals include 0. ADR-005 pre-registered
   this as a hypothesis with either direction acceptable; the measured answer is "no detectable
   widening".
4. **NFR-4 not met.** The requirement is QMIX near-miss conflicts ≤ 50% of IQL's at N=10. The
   QMIX/IQL ratio is 1.05 [0.80, 1.37] at N=10 (0.92 at N=5, 1.03 at N=20). Neither learned policy
   *executes* a collision; the joint move resolver blocks every conflicting move.
5. **Crowding at N=20.** Demand is fixed while robots share 2 chargers and the same aisles. Learned
   delivery times roughly double, energy per delivery rises about 75%, and a few battery depletions
   appear. The privileged planner is unaffected (100%, 29 ticks). So the learned policies lose
   ground to the planner as N grows: QMIX − planner completion goes from −0.3 to −2.3 to −7.9 pp.
6. **Throughput is capped by demand.** All policies stay near 0.0667 orders per tick
   (40 orders / 600 ticks), so adding robots cannot raise it.

### Safety

There are two checks, both on the 20 sealed seeds at each fleet size.

**Learned and planner policies:**
- 0 safety violations and 0 corridor-entry attempts across all 180 benchmark episodes. The mask
  removes corridor moves before the policy chooses.
- 0 executed collisions.

**Adversarial policy** that ignores the mask and drives at any adjacent corridor cell:

| N | Episodes | Corridor-entry attempts blocked | Intrusions |
|---|---|---|---|
| 5 | 20 | 56,134 | **0** |
| 10 | 20 | 110,913 | **0** |
| 20 | 20 | 209,492 | **0** |

OBJ-4 / NFR-3 are met: zero policy-induced intrusions across dedicated maximum-aggression episodes.
The CI gate runs the safety tests on every push.

## 5. Limitations

- **One training seed per cell.** Confidence intervals cover scenario variance only, not
  run-to-run training variance, so differences between algorithms should be read as indicative.
- **Fixed demand across fleet sizes.** N=20 does the same 40-order shift as N=5 on the same map,
  so the scalability results measure congestion handling, not throughput scaling.
- **The planner is privileged.** It sees global state, while the learned policies see only a
  noisy 5 × 5 local window. Matching the planner is the ceiling, not the expected result.
- **Imitation-heavy.** The policies start from planner demonstrations and keep an imitation
  regulariser, so they inherit the planner's habits; RL mainly fine-tunes.
- **No out-of-distribution evaluation.** Map layouts and robot models are never varied, and
  high-demand regimes (3× arrivals) overwhelm even the planner (see the drift report).
- **Synthetic physics and battery model**, with no sim-to-real validation.

## 6. Ethical considerations

See [24_RESPONSIBLE_AI_AND_SUSTAINABILITY_REPORT.md](24_RESPONSIBLE_AI_AND_SUSTAINABILITY_REPORT.md).
In short: the reward has no term for the people working alongside the robots; the safety layer
is deterministic code independent of the model, but it is not certified; and any productivity
gain carries documented risks of work intensification and repetitive-strain injury.
