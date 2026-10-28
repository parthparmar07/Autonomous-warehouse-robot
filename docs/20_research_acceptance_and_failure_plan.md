# Research acceptance and failure analysis plan

Use this alongside the V6 protocol in document 19. A passing simulator, a planner
success, an assisted policy and a pure-RL policy are four different claims. A
historical 74-case design catalog is archived outside the repository; its
design-only entries are not automatically executable or passed.

## Acceptance layers

| Layer | Required evidence | Current limitation |
|---|---|---|
| Physical integrity | Unique positions/cargo ownership; valid stations; no teleportation; legal simultaneous resolution; deterministic replay | Synthetic discrete physics; no calibrated dynamics or real collision sensing |
| Decision integrity | No future-event or hidden-traffic leakage; legal charge tiers; actual executed-action feedback; recurrent reset/replay correctness | Public static map and own localization are assumptions |
| Task feasibility | Privileged planner completes the same registered scenarios; capacity and deadlines admit a solution | Planner cannot certify optimality; some adversarial cases intentionally infeasible |
| Learned elementary skills | Held-out worker/heading service, full-map navigation, passing, bay clearance, charging then delivery | Passing isolated skills does not establish composed shift operation |
| Composed operations | Positive throughput under concurrent traffic, later arrivals, faults and charger contention | Robust C3–C5 training and evaluation still required after core gates |
| Statistical performance | Independent training seeds; paired scenario streams; uncertainty over runs and scenarios; retained failures | Three-seed main pilot and one-seed controls are exploratory |
| Generalization | Frozen ID and OOD tests; mirrored/novel layouts and unseen event compositions; N=10/20 | No V6 final acceptance or scaling claim yet |
| Deployment | Recurrent online inference, explicit model compatibility, latency, 3D demonstration and reproducible environment | A browser animation does not validate learned control or physical readiness |

## Adversarial task matrix

| Concern | Situations to vary | Measurements and oracle |
|---|---|---|
| Navigation | All headings/worker slots; goals across racks; initially move away from target; blocked/unknown cells | Delivery, route efficiency, turn reversals, distinct cells, attempted conflicts; feasible planner comparison |
| Service | Approach from each side; interrupt/restart pickup/drop; two carriers competing for a bay | Consecutive physical service ticks, exactly-once package transitions and correct owner |
| Traffic | Opposing narrow-aisle traffic, passing pocket, intersection, convoy, cyclic rotation | Delivered orders, deadlock duration and attempted conflicts; zero overlap alone is insufficient |
| Useful waiting | No orders, future demand, queued charger, blocked aisle, reserve worker | Mode-conditioned WAIT and eventual work; do not punish necessary waiting or demand equal deliveries |
| Harmful inactivity | Assigned reachable work but stationary; spinning; shuttle loops; parked on service bay | Stagnation interval, repeated state/action cycles, missed orders and bay occupancy |
| Charge decision | SOC just above/equal/below 30%; current task; cargo; cooldown; reservation already held | Charge-action legality; no threshold bypass; no dropped current order |
| Charge execution | Both chargers, all headings, owner/queue changes, lease expiry, approach stall, slow taper | Request-to-dock time, queue wait, completed cycles, release SOC, subsequent delivery |
| Energy | Feasible reserve margin; mission infeasible before assignment; depletion en route; low SOC under congestion | Avoidable depletion, rejected assignments, energy/delivery, unfinished work; no free refill on repair |
| Handover | Unpicked reassignment versus loaded failed carrier; authorized transfer adjacent to bay | Cargo remains with carrier until actual transfer; receiver service and eventual delivery |
| Recovery | Fault before pickup, while loaded, while charging, then repair; temporary blockage | Recovery time, preservation/ownership, resumed work; distinguish injected from avoidable failures |
| Demand | Single stream, burst, one hot shelf, shared dispatch, no demand, persistent overload | Throughput, backlog slope, weighted tardiness, censored latency and station utilization |
| Sensing | 360-degree, front/rear sectors, rack occlusion, dropout, latency, changed configuration | Completion degradation and observation privacy; masks must not reveal unseen robot occupancy |
| Scaling | N=5/10/20 at fixed area and area per robot; report chargers/bays per robot | Throughput/robot, congestion, energy, inference cost; train fleet-specific mixers |
| Distribution shift | Held-out layout, outage plus burst plus fault, capacity variation plus degraded sensing | Worst-family results and infeasible-case labels; no universal preparedness claim |

Existing named executable skills cover portions of these rows, not every
cross-product. COO06 loaded failure and ROB05 outage/burst compositions are kept
outside the named training mix. EVA05 requires repeated completed charge cycles.
Passing the shorter CHARGE_CYCLE case alone does not validate long-shift charging.

## How to decide what to fix next

1. **Check feasibility and traces before increasing training.** Reproduce a failed
   seed with the privileged planner, inspect service/ownership and timing, and
   preserve the failing manifest. The newly fixed joint-plan WAIT-at-service bug
   is an example of an environment/baseline defect that more RL could not fix.
2. **Separate exploration from representation.** Compare initial, cloned and
   post-TD development outcomes. If demonstrations succeed and TD destroys them,
   investigate forgetting, Q scale, update drift and replay balance. If cloning
   itself fails, examine demonstration coverage, recurrent history and aliased
   observations before interpreting a QMIX loss curve.
3. **Separate policy failure from task assignment failure.** The dispatcher can
   create poor assignments or charger contention. Measure task rejection,
   reassignment and lease behavior independently of local steering. Learned
   high-level allocation would be a new experiment, with an explicit interface
   and rule-based allocation control.
4. **Extend budgets when the evidence warrants it.** Improving held-out curves
   at a cap justify a separately recorded continuation. Flat scores, oscillatory
   navigation or poor expert generalization justify a targeted change first.
   Preserve failed runs and match extra steps/expert/update costs across methods.
5. **Change the algorithm only after these controls.** QMIX's monotonic mixer
   restricts joint value representations; it does not explain a solo-navigation
   failure by itself. IQL is a useful control for the same actor/task interface.
   VDN, stronger recurrent baselines, longer-memory/sequence ablations or richer
   value factorization should be labelled new conditions with equal tuning cost.

The [original QMIX paper](https://arxiv.org/abs/1803.11485) establishes its
monotonic factorization and decentralized greedy consistency, not convergence on
our battery/warehouse extension. The [official PyMARL configuration](https://github.com/oxwhirl/pymarl/blob/master/src/config/algs/qmix.yaml)
and [EPyMARL implementation](https://github.com/uoe-agents/epymarl) are reference
implementations for comparison, not grounds to copy settings without measuring.

## Reporting rules

Report joint environment steps, active agent transitions, optimizer updates,
expert steps, BC/DAgger costs and wall time separately. C0 has one active worker
despite five physical slots, so multiplying every joint step by five exaggerates
active experience. Do not rank algorithms by unmatched stopping budgets alone.

Report task completion both as a fraction of released orders and scenario
success. The 95%-completion scenario oracle and Wilson lower bound used for
promotion are stricter than the plan's 80%-delivery target. At 20 scenarios the
current lower-bound rule effectively needs 20/20 successes; state this openly.
These are development gates, not a proof of 80% reliability under arbitrary
warehouse conditions.

Treat repeated evaluation scenarios as paired measurements, not independent
training seeds. One hundred evaluations of one trained actor do not replace
multiple training runs. Final tests should be consumed only after freezing the
policy/selection rule. Record null or censored metrics when no delivery occurs;
do not turn undefined energy per delivery or unfinished-order latency into zero.

Remain explicit about what is unfinished: final/OOD generalization, robust-stage
training, larger fleets, hyperparameter/reward search, calibrated sensing/energy,
long-shift charger fairness, 3D demonstration and deployment monitoring. None is
completed merely because an implementation, catalog entry or design document exists.
