# Papers, original requirements and policy-controlled charging

Reviewed 2026-10-26 against the executable code. The user's new 30% rule overrides
the earlier 45% experiment. Document recommendations are evidence to assess, not
instructions to execute unrelated work or force an experimental outcome.

## Sources and interpretation

1. Khlifi et al., *On Diagnostics for Understanding Agent Training Behaviour in
   Cooperative MARL*, arXiv:2312.08468v1, supplied six-page PDF. Relevant material:
   PDF p2 (diagnostics and experiment), p3 Figure 2 (learning curves), p4 Figure 5
   and the sparsity discussion.
2. Bahrpeyma and Reichelt, *A review of the applications of multi-agent
   reinforcement learning in smart factories*, Frontiers in Robotics and AI,
   2022, DOI 10.3389/frobt.2022.1027340, supplied 21-page PDF. Relevant material:
   pp10–13 transportation, p15 Table 2, pp15–16 energy, pp16–19 discussion.
3. Supplied *warehouse plan (1)*, eight PDF pages. Requirements pp1–4; evaluation
   pp5–6; risks p6. Page 7 includes unrelated earthquake/ETAS text, treated as
   copy-paste contamination, not a warehouse requirement. Page 8 is empty.

The first is an experiment and diagnostics paper. The second is a literature
review of different tasks and algorithms, not one reproducible warehouse recipe.
The plan is the project's acceptance specification; its desired performance
advantages are hypotheses, not evidence that an algorithm must outperform another.

## Environment comparison

| Aspect | Diagnostics paper / RWARE | Factory review | Original plan | Current implementation |
|---|---|---|---|---|
| Physical task | Retrieve requested movable shelves, deliver to workstation, return shelves to empty locations | Many distinct systems: pathfinding, scheduling, pick/place, UAVs, energy | Fixed shelf-to-dispatch package orders | Fixed shelves; packages move through pickup/drop SERVICE transitions |
| Layout | Configurable RWARE grid and requested-shelf density | Grid, graph, continuous and real-world examples | Structured warehouse and forbidden human corridors; N=5,10,20 | 20x20 default, six rack blocks, 12 pickups, four drops, two chargers, one transfer bay |
| Observations | Local partial view | Both partial and full observations, depending on cited work | Local occupancy plus task bearing/assignment | 5x5 local synthetic sensing; 346 features, own task/energy, sensor configuration and executed-action feedback |
| Actions | Five in RWARE code: NOOP, forward, left/right rotation, load/unload | Choi example has five movements/stop; other studies differ | Four directional primitives | Seven: forward/backward, turn left/right, wait, service, request charge |
| Task allocation | RWARE requested shelves; not our fixed order dispatcher | Some papers learn allocation, others navigation or bidding | Queue and rule-based nearest-task baseline | Shared rule-based feasible greedy pair matching; actor does not allocate orders or select chargers |
| Batteries | No battery scheduling in the described RWARE study | Energy section includes smart-grid/EV systems; not a 30% warehouse prescription | Energy-efficiency reporting, not a specified battery/charger protocol | Synthetic energy costs, finite chargers/queues and new actor-controlled 30% request rule |
| Extra difficulty | Sparse deliveries and coordination | Different papers address different complications | Homogeneous robot core | Varied capacity/energy factor, faults, repairs, arrivals, sensor degradation and handover prerequisites |
| Physics/display | RWARE discrete simulation | Varies | PyBullet plus grid; risk section explicitly allows fast NumPy training and separate PyBullet demo | NumPy grid physics and 2D replay; required PyBullet 3D demo not implemented |

These are not identical benchmarks. A success percentage or return cannot be
compared directly across movable shelves and fixed packages, different episode
lengths, reward magnitudes, sensors, or allocations. RWARE actions/collisions and
observations were cross-checked against the [official environment](https://github.com/semitable/robotic-warehouse).

The original homogeneous-fleet specification should remain a separately measured
core condition. Capacity/consumption variation is an approved later project
extension, but it should not silently replace that core condition.

## Training comparison

The diagnostics paper uses EPyMARL, optimized hyperparameters, parameter sharing,
seven LBF scenarios, three RWARE scenarios, and ten seeds (PDF p2). Its Figure 2
plots curves through two million timesteps. It does not give a complete standalone
table of every optimizer/replay/epsilon setting; it refers to the benchmark setup.
Its one-seed diagnostic case studies are not ten independent training curves per
individual figure. It does not demonstrate our C0–C5 curriculum or charging rule.

For context, the [underlying EPyMARL benchmark, section 4.1](https://arxiv.org/html/2006.07869)
reports four million off-policy steps for RWARE, five evaluation-training seeds,
and 100 evaluation episodes at each of 41 checkpoints. Its longer budget must not
be substituted for what the supplied diagnostics paper's two-million-step figures
actually show. Neither number guarantees convergence in this custom environment.

Our actor is shared across robot slots, with an MLP and GRU of width 64, previous
issued actions and local observations. QMIX uses a state-conditioned monotonic
mixer; IQL uses shared recurrent utilities with independent TD terms. This matches
the paper's parameter-sharing comparison better than the plan's literal separate
DQN network per robot. Our actor has no explicit robot-ID input, another detail
that differs from the underlying EPyMARL benchmark's shared-network setting.

Current learner defaults: Adam LR 0.0003, gamma 0.99, batch size 8, replay 500
episodes, burn-in 20, unroll 40, target refresh every 200 optimizer updates. The
generic runtime YAML still defaults to 20k steps versus a 500k-step epsilon decay;
these are pilot defaults and must be overridden deliberately. The curriculum
script uses CUDA by default, 100k steps per stage, per-stage exploration decay,
and a 25k-step development interval. Its total default cap is 800k steps if every
stage passes. The separate research YAML allocates two million steps per run.

Current stage order is C0 -> C0.5 (full-map solo) -> C0.75 (two workers) -> C1 ->
C2 -> C3 -> C4 -> C5. C1–C3 retain ideal sensing. Generic C4/C5 generation still
matches, with more named training cases at C5. Generic and named-skill retention
checks are implemented; they are stricter than the plan's 80% delivery target.
We have not completed such a learned-policy progression.

Historical evidence: six assisted C0 V3 runs scored 100%, while one pure-RL C5
run each for QMIX/IQL scored zero after 50k steps. A V4 tiny assisted GPU check
reached 800 total steps and verified resume. These are not new V5 C5 results and
do not reproduce the paper's multi-seed benchmark.

## Reward comparison

RWARE's main signal is successful requested-shelf delivery. The official default
delivery reward is 1; reward sharing variants must be reported. The diagnostics
paper treats its warehouse setup as sparse reward and cautions that sparsity
alone need not explain failure.

The review's Choi et al. summary (PDF p13) describes positive reward for reducing
Manhattan distance to a target, while the PRIMAL discussion (pp11–12) describes
collision penalties and goal rewards. Other reviewed work uses tardiness or edge
costs. These are different objective choices. The review is secondary evidence
about those implementations; it supplies neither one shared reward equation nor
one shared training budget. Its wording about individual Qa/global Qtot must not
be read as two observed environmental rewards: these are value estimates in QMIX.

The project plan asks to tune collision/delivery weights and report task
completion, time and safety, but prescribes no complete numeric reward equation.

Our actual shared reward in `src/environment/physics.py` is:

```
r = 10 * deliveries_this_step
    - 0.001 * unfinished_orders
    - 0.01 * energy_used_this_step
    - 0.05 * movement_conflicts_this_step
    - 0.002 * sum(priority of overdue unfinished orders)
    - 10 * new_avoidable_battery_depletions
```

Delivery credit is sparse; costs are dense. There is no pickup reward, distance
progress reward, movement bonus, explicit stay penalty, or charging reward.
Waiting consumes a little energy and still incurs backlog/lateness costs.
This is not the same reward as either the RWARE benchmark or Choi's shaping.
Reward weights are hard-coded; current Optuna code tunes LR/gamma, not delivery
weights, collision weights or mixer width. OBJ-5 is therefore incomplete.

Do not copy a Manhattan-progress reward blindly into this shelf map. The useful
route can initially move away from the goal to get around a rack; goal changes
and loops also complicate shaping. Any proposed shaping needs a separately
labelled ablation and checks for loops/goal-switch exploitation. No reward
weights or shaping terms were changed in this task.

## Diagnostics to adopt from the first paper

Return and completion alone do not show whether a robot waits helpfully, turns
in circles, over-requests charging or loses service progress. The paper adds
policy entropy, update divergence and action-usage/task-switching plots. Its
RWARE discussion explicitly proposes an explanation beyond pure reward sparsity.
These are diagnostic hypotheses, not causal proof that one metric fixes learning.

Our runner currently records aggregate action counts and per-robot visited-cell
counts, but not full state-conditioned action distributions, frozen-probe policy
drift, or per-mode productive wait/service statistics. Add those before choosing
between longer exploration, imitation, reward shaping or another algorithm.

For Q-learning, distinguish the actual legal epsilon-greedy behaviour distribution
from a diagnostic softmax over Q-values. They are not the same policy. Report a
softmax temperature if used. Greedy entropy alone is zero for a unique greedy
action, while epsilon-greedy entropy is heavily controlled by the epsilon schedule
and mask size. Compare update divergence on the same observations, recurrent
histories and masks; otherwise changing encounters can look like policy change.
RWARE action 0 is NOOP in the [official action enum](https://raw.githubusercontent.com/semitable/robotic-warehouse/master/rware/warehouse.py).
High wait usage is not automatically laziness; condition it on work, queueing,
traffic and eventual deliveries. Raw action numbers differ from our action enum.

## Requirements audit

| Requirement | Current evidence/status |
|---|---|
| FR-1 local robot decisions | Implemented synthetic local observations and recurrent decentralized actions; seven-action extension is a deviation from four primitives |
| FR-2 global conditioning / monotonic mixer | Implemented and gradient-tested; critic includes fleet/queue but still omits some energy/lease/queue details |
| FR-3 corridor exclusion and logging | Masks plus physical backstop; independent 20-seed adversarial regression exists |
| FR-4 deterministic queues | Seeded generators, events, manifests and replay checks implemented; each new version needs its own frozen evaluation set |
| FR-5 global snapshot fleet API | Partial: session API owns global simulation, but `/predict` accepts local observation/mask batches and is stateless; not the literal global-snapshot request contract |
| FR-6 MLflow across fleets/metrics | Partial: JSON metrics exist and optional MLflow integration logs selected metrics; not the complete required per-step/metric lineage across all fleets |
| NFR-1 >=80% held-out N=5 / comparative benefit | Not achieved; historical full C5 0%; no V5 learned full-fleet result |
| NFR-2 scaling advantage | Unmeasured; separately trained N=10/N=20 comparisons pending |
| NFR-3 zero corridor intrusions / CI | Local independent regression implemented; remote CI execution not verified here |
| NFR-4 <=half IQL collision rate at N=10 | Unmeasured; physical shielding makes executed collisions zero for either learner, so also report attempted conflicts, blocked time and overrides |
| NFR-5 fleet-size curves with CIs | Not achieved; scripts/CI helpers are not completed experiments |
| NFR-6 reproducibility / wall-clock budget | Seeds, hashes and manifests exist; no reproduced successful result or verified <1h GPU/<3h CPU training promise |
| OBJ-5 tuning | Partial: LR/gamma tuning only; requested reward and mixer-width sensitivity not implemented/executed |
| OBJ-6 portfolio | 2D replay, API and ONNX path exist; PyBullet 3D, Evidently drift, CodeCarbon and complete deployment/lineage evidence remain outstanding |

Plan issues to correct in the written requirements: NFR-1/2 and ETH-4 assume a
favourable QMIX result. Preserve them as hypotheses and report negative results;
do not tune on the sealed test to force significance. NFR-4 is not evidence of
learned collision avoidance when the hard resolver blocks all impacts. The plan
also mentions both five- and ten-episode CI smoke runs in different places;
choose one explicit contract. Its 80% final target and our stricter development
gate are different thresholds, not interchangeable claims.

## Implemented 30% charging policy

Normal-path rule, using the existing configured trigger (default 0.30):

| State | Request charge? | Dispatcher behaviour |
|---|---|---|
| SOC >30% | Masked; direct request rejected | Allocate only feasible orders; no automatic charger request |
| SOC <=30%, current order or package | Masked until the order finishes | Preserve current order and physical cargo |
| SOC <=30%, active, unassigned and empty | Enabled if outside a reservation/cooldown and able to issue the action | Hold new assignments so the policy can choose REQUEST_CHARGE or wait |
| Already reserved/charging | No duplicate request | Existing station/queue controller remains responsible for reservation and release |
| Docked and SOC rises above 30% | SERVICE remains available | Continue the existing cycle toward 90%; do not terminate charging at 30% |

The same guard is enforced by masks and transitions; legal masked action values
are used during greedy selection and TD targets. The dispatcher no longer makes
an automatic request at 30%, or because an above-threshold order is infeasible.
The rule baseline now explicitly issues action 6. Charger choice and physical
queue assignment remain rule-based. The privileged forced-request helper remains
only for explicit fixtures/controllers, not normal dispatch or the actor/API flag.

This makes charging a real learned decision. Unlocking is not a reward and does
not guarantee learning: successful later deliveries must provide its value.
Training must include sufficient future demand and horizon after a charge, plus
reachable chargers and feasible initial energy. A finite batch can terminate
before charging after the final order is useful; do not require that terminal
charge to receive positive credit. Mission admission still checks route/service/
charger energy plus reserve. It does not guarantee a wandering policy can finish
an order; depletion stays an explicit failure rather than being hidden by rescue.
An above-threshold robot with no feasible order may have no productive task:
report/calibrate those cases instead of secretly bypassing the action rule.

The environment version is now `warehouse_v5_policy_charge`; observations remain
346-dimensional. V3/V4 weights and manifests cannot silently load into it. The
existing browser replay and earlier scores still describe older contracts.

## Recommended next experiment, in order

1. Resolve the known parking reward/oracle mismatch without adding generic motion
   bonuses or penalizing necessary waiting.
2. Establish homogeneous full-map solo and two-worker delivery before combining
   energy variation, faults and sensor corruption. Keep harder extensions as
   measured ablations, not prerequisites for the first core-plan result.
3. Test battery request timing with task completion followed by future orders;
   measure request SOC, completed charge cycles, productive post-charge service,
   and failures by role. No reward for requesting or charging alone.
4. Use matched QMIX/IQL conditions and several independent training seeds. The
   supplied diagnostics paper uses ten; the earlier EPyMARL benchmark uses five
   for evaluation. Budget in millions is a research starting point, not proof of
   sufficiency. Select checkpoints on development only.
5. Freeze fleet-specific tests and report completion/time/conflict/energy plus
   uncertainty for N=5, then separately trained N=10 and N=20. Do not claim either
   algorithm must beat a privileged planner or achieve a predetermined gap.

## Validation

Current CPU suite: 201 passed, three CUDA-only skipped, one known parking expected
failure. CUDA subset: 16 passed, including actual QMIX/VDN/IQL GPU updates and the
new charging boundary/action tests. Tests cover SOC boundary across capacities,
raw-action bypass, absence of automatic reservation on wait, order completion
before the next assignment, infeasible-mission fallback removal, arbitrarily high
charge Q-value masking, and continuing service through the 30%-to-90% charge cycle.
There is no new V5 learned full-warehouse training result in this comparison.

The privileged rule planner was additionally evaluated on development seeds
300000–300019 with N=5, C5 and horizon 400 under V5: 300/300 released orders
delivered across 20/20 successful episodes, with zero battery depletions. This
establishes feasibility for those scenarios, not learned QMIX performance or
independent-seed statistical readiness. Source/PDF hashes and measured results
are recorded under `outputs/paper_review_20261026/`.
