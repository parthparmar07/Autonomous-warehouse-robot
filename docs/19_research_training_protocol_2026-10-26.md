# V6 research protocol and live training evidence

This document supersedes the implementation status in document 18, but does not
replace its historical results or paper comparison. The current experiment is a
development study, not a completed replication or a final acceptance result.

## What the papers actually justify

Khlifi et al., *On Diagnostics for Understanding Agent Training Behaviour in
Cooperative MARL* (supplied arXiv:2312.08468v1), motivate parameter sharing,
multiple seeds, action-use/entropy/update-divergence diagnostics and investigating
failure beyond sparse rewards. Their plots extend to two million timesteps. The
underlying EPyMARL benchmark reports four million off-policy RWARE timesteps;
neither is a convergence guarantee for this different task.

Bahrpeyma and Reichelt's factory review (DOI 10.3389/frobt.2022.1027340) describes
different navigation, allocation and energy problems. Its examples motivate
testing imitation, progress feedback and explicit scheduling separately. It is
not a specification for one reward function, battery threshold or training budget.

Our fixed-package warehouse is different from RWARE's movable-shelf retrieval and
return problem. Scores cannot be compared directly. The user's warehouse plan
defines the project scope: homogeneous fleets first, local decisions with a
centralized training critic, repeated-seed measurements, held-out scenarios,
N=5/10/20 scaling, safety and energy reporting. A QMIX advantage is a hypothesis.
The new 30% charging rule comes from the user, not either paper.

## Environment and decision boundaries

The default 20 by 20 map contains six fixed rack blocks, twelve shelf-side pickup
bays, four dispatch bays, two chargers and one authorized transfer bay. Packages
originate at registered shelves and move only through physical pickup, carriage,
handover and delivery. Service normally requires three consecutive valid ticks.
The joint resolver prevents overlap, head-on swaps and forbidden-corridor entry;
attempted conflicts and blocked time therefore matter more than a trivially zero
executed-collision count.

The shared recurrent actor chooses forward/backward motion, left/right turns,
wait, service and request charge. A rule-based, energy-feasible greedy dispatcher
assigns orders and charger reservations. Allocation is not learned jointly. The
privileged cooperative planner is a baseline and demonstration generator, not
the learned actor. Training on its trajectories is explicitly imitation-assisted.

Each actor receives 348 features: a 5 by 5 egocentric range/occupancy window,
own task/energy/mode, sensor settings, normalized own position and two executed
action histories. Own localization is an explicit assumption; no global live
robot list, future event schedule or planner route enters the actor. Sensing
supports front/rear coverage, occlusion, dropout and latency. These are simulated
range sensors, not a learned camera system. The default core study uses ideal
range sensing; degraded sensing is a later robustness condition.

The centralized critic receives current layout, released orders, fleet and
charger information, now including capacity, energy factor, health, reservations,
queue ordering/ages and approach/retry/fault timers. Its N=5 state has 2,538
features. This improves conditioning but is not a formal proof that every hidden
simulator variable has been represented as a Markov state.

Battery units remain synthetic. Capacity is 100 in the homogeneous core. Empty
motion costs .2, loaded motion .3, turning .05, waiting .005 and service .1.
Charging adds 2 units/tick below 80% and .5 thereafter, releasing at 90%.
REQUEST_CHARGE is masked above 30%; at or below 30% it requires an active,
empty, unassigned robot without a reservation/cooldown. Existing orders continue.
Idle low-charge robots receive no new orders, so the actor must choose to request
charging. Docked SERVICE continues beyond 30%. Ordinary dispatch supplies no
automatic rescue; tests/controllers have an explicitly privileged forced hook.

## Task curriculum and gates

| Stage | Core task | Maximum additional RL joint steps |
|---|---|---:|
| C0 | One active worker, introductory nearby pickup and dispatch | 10,000 |
| C0.5 | One active worker on the full warehouse, two orders | 30,000 |
| C0.75 | Two active workers on the full warehouse, four orders; passing skill | 30,000 |
| C1 | Five active workers, ten orders; passing, bay clearance, reserve arrival and solo continuation | 50,000 |
| C2 | Homogeneous fleet with initial SOC 20–65%; charging cycle and battery recovery skills | 50,000 |

Worker identities and headings vary. CHARGE_CYCLE places a rotating worker at a
charger with 20% SOC and releases later orders. Passing requires completing a
charge and subsequently delivering, not just selecting the charge action.
COO02 now checks clearing service bays rather than an arbitrary parking point.
An idle robot is allowed to wait where it does not obstruct a service bay.

The core profile deliberately removes capacity variation, faults and sensor
degradation. The existing robust C3–C5 mechanisms remain available for subsequent
fault/repair, transfer, arrival, outage and degraded-sensing studies. Calling core
C2 mastery full C5 robustness would be incorrect. This launch trains through C2
only, and stops earlier on failed prerequisites.

Every 2,000 RL steps a cheap development screen checks current tasks and retained
skills. It can reject, never promote. A successful screen triggers the full gate:
20 generic scenarios per stage plus 20 worker/heading cases per named prerequisite.
Promotion requires two distinct full passing checks and at least 4,000 new RL
steps in that stage. A stage cap is a ceiling, not automatic permission to advance.
Final and OOD partitions are not used for these choices.

## Shared reward

The executable reward is the sum of these independently logged components:

```
r = 10 * new_deliveries
    - .001 * unfinished_orders
    - .01 * energy_consumed_this_tick
    - .05 * movement_conflicts_this_tick
    - .002 * sum(priority of overdue unfinished orders)
    - 10 * new_avoidable_depletions
    - .02 * idle_unreserved_empty_robots_on_service_bays
    + w * (.99 * Phi(next_state) - Phi(current_state))
```

Phi sums exp(-remaining_mission_work / 20) over active assigned robots. Remaining
work includes legal grid-path distances and remaining pickup/drop service ticks.
It is a centralized training signal; no waypoint is exposed to the actor. True
terminal states use Phi=0; time limits retain Phi and the learner bootstraps.
The shaped condition uses w=2; the unshaped control uses w=0. Delivery remains the
primary positive task event. Charging requests, pickups alone, motion and generic
waiting receive no standalone bonuses.

This follows the potential-difference construction in
[Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
Its policy-invariance result under its assumptions does not guarantee convergence
of recurrent QMIX with function approximation. Assignment changes, partial
observability, discounts and episodic termination must still be audited. Tests
check discounted telescoping, unchanged-state waiting/loops, service progress and
terminal delivery. No reward tuning against final tests is permitted.

## Training study actually launched

Source snapshot SHA256:
`786d956198ef6ff3ed30c7aabb35cd699a34c309d0e82b7a1b94f4a2408f27c2`.
The archive, per-job configurations, logs and live matrix are in
`outputs/research_training_20261026/`. The RTX 4070 Laptop GPU runs two jobs at a
time; training source/configuration is frozen while they execute.

| Condition | Algorithms | Seeds | Purpose |
|---|---|---|---|
| Assisted, potential weight 2 | QMIX, IQL | 11, 23, 37 | Main staged development comparison |
| Assisted, potential weight 0 | QMIX, IQL | 11 | Reward-shaping diagnostic control |
| Pure RL, potential weight 2 | QMIX, IQL | 11 | Demonstration diagnostic control |

The assisted condition uses 40 expert episodes, ten behavior-cloning epochs,
one 40-episode DAgger round and auxiliary imitation weight .2 **at each stage**.
Expert transitions and optimization time are logged separately from RL steps.
The pure-RL control disables all three assistance mechanisms. The two controls
have only one seed and cannot support robust significance claims.

All runs use a shared recurrent actor of width 128, QMIX mixer width 32 where
applicable, Adam LR .0003, gamma .99, batch size 8, replay capacity 128 episodes,
burn-in 20, unroll 40 and one update per 16 joint environment steps. Assisted
epsilon decays from .2 to .05 over 10,000 stage-local steps; pure RL starts at 1.
Actor, optimizer, replay and training counters warm-start between stages. Source,
scenario/configuration hashes and resumable state accompany each checkpoint.

The maximum staged budget is 170,000 RL joint steps per run, less with early
promotion or rejection. This is a measured development study, substantially below
the cited papers' million-step experiments. Additional iterations are justified
by learning curves, exploration, coverage and task diagnostics, not by an assumed
universal QMIX training requirement.

## Diagnostics, verification and remaining limits

Logs include per-robot action use conditioned on operating mode, reward-component
totals, actual legal epsilon-greedy behavior entropy, deliveries, backlog,
tardiness, energy, depletion, conflicts, deadlocks and completed charge cycles.
A frozen recurrent-history probe records softmax-Q entropy, legal-Q margins,
greedy actions and update KL on identical histories. Softmax-Q uses temperature 1
and is a diagnostic distribution, not the epsilon-greedy behavior policy.

Preflight passed 20/20 cases in each of five core stages and six named core
skills: 220 planner episodes total. It caught a planner defect where an active
joint plan forced WAIT at a delivery bay instead of SERVICE; this was fixed and
regression-tested before expert collection. The failing preflight was retained.
Planner success establishes executable feasibility, not learned-policy mastery.

Verification before launch: full CPU suite 209 passed and three CUDA-only skips;
subsequent planner regression selection three passed, including two new tests;
CUDA reward/diagnostic/update selection nine passed. There are no remaining
expected-failure exemptions in that tested suite. These checks do not establish
training convergence, statistical power or real-robot safety.

Open research limits include full-map generalization and forgetting; incomplete
named-role exposure in a small stage-local expert sample; hyperparameter/reward
search cost; larger-fleet charger congestion; action-space and task-allocation
deviations from the plan; uncalibrated energy/sensors; and remaining robust-stage,
OOD, N=10/20, final statistical, PyBullet and deployment evaluation. New V6 final
realizations were not all sealed before this development pilot began, so it is
not a preregistered confirmatory study. A later confirmatory study must seal its
protocol/manifests before training and retain all failed seeds.

Measured learning results are recorded separately in the generated
[live report](../outputs/research_training_20261026/live_results.md). It refreshes
while the finite launched study runs. The first seed-11 pair completed C0 and
stopped at the failed C0.5 development gate: QMIX 35/40 orders, IQL 31/40 on the
fixed full-map-solo development set. Separate, paired five-robot transfer checks
on seeds 308000–308019 measured:

| Test (not trained/promoted) | QMIX delivered | IQL delivered | QMIX / IQL avoidable depletions |
|---|---:|---:|---:|
| C1 core, full initial battery | 192/200 | 172/200 | 0 / 2 |
| C2 core, low initial battery | 176/200 | 123/200 | 23 / 47 |
| C5 robust | 56/300 | 31/300 | 45 / 32 |

These are one-seed, imitation-assisted, shaped development transfers, with
different accumulated C0 step costs (QMIX 8,001; IQL 4,256) before the same
30,000-step C0.5 budget. They do not establish a statistically reliable QMIX
advantage, a mastered charging stage or C5 robustness. The full-battery task is
substantially easier than the robust condition. Zero executed collisions is a
physical backstop, not learned collision avoidance. No successful C0 score should
be described as five-robot warehouse success. See
[the acceptance/failure plan](20_research_acceptance_and_failure_plan.md) for
the remaining tests and how to choose the next change.
