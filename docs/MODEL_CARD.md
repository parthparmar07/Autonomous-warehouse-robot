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

<!-- SCALABILITY_TABLE -->

### Safety

<!-- SAFETY_TABLE -->

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
