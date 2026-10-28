# Charging and curriculum corrections — 2026-10-26

The executable contract is now `warehouse_v4_guarded` with observation version
`egocentric_range_v2_action_feedback`. Earlier V3 checkpoint scores and the
browser's C5 replay remain historical results, not measurements of this revision.
Existing weights/manifests fail version checks; they are not silently reused.

## What the pasted diagnosis establishes, and what it does not

The old voluntary `REQUEST_CHARGE` guard allowed requests below 90% SOC and could
release an unpicked order. A request changes the charging state machine, rather
than simply moving one cell. This is a plausible exploration and control-design
problem. The displayed rollout used greedy evaluation, however, so its actions
were not epsilon-random at playback time. High final charge does not establish
charge at the moment of the request: robots may have charged earlier.

No controlled learning ablation yet establishes that this explains the entire
zero-delivery result. The actual reward has delivery credit and energy, backlog,
conflict, lateness and depletion costs; it has no approach-to-charger shaping
bonus or charging bonus. At nominal empty movement cost, losing ten energy units
takes fifty forward moves, not just a few steps.

Tracing the first recorded C5 trial confirms a concrete failure: its third joint
action requests charging for robots 0, 1, 2 and 4 at approximately 64%, 48%, 78%
and 66% SOC. All four held pickup assignments immediately before requesting;
all four release them. The new guard rejects each request. This is an observed
state-machine failure, not a measured post-fix improvement in learned throughput.

The [QMIX paper](https://proceedings.mlr.press/v80/rashid18a.html) establishes the
monotonic construction's centralized/decentralized greedy consistency; it does
not guarantee neural TD convergence under the two conditions alleged in the
pasted diagnosis. Randomizing sensor configuration between episodes is a harder
inference problem, not by itself proof of nonstationary transition laws or pure
noise in every TD target. Recurrent history already existed. Explicit feedback
and configuration improve available information but do not guarantee recovery
of hidden observations, other robots' actions, or occluded state.

## Changes

### Charging admission and leases

`RuntimeConfig.voluntary_charge_threshold` defaults to 0.45. An actor request is
legal only below that threshold, while active, unassigned, empty, outside a
charging reservation, and past its retry cooldown. The mask and transition share
the same own-state guard. Sending raw action 6 cannot bypass the restriction.

Automatic dispatcher requests still handle low SOC (default 30%) and insufficient
energy for an available mission. Loaded cargo stays on its carrier. The dispatcher
has a separate explicit `automatic=True` entry point; action 6 cannot set that flag.
Automatic requests and actor requests are distinguished in event logs.

Charger ownership starts a new approach lease upon admission or promotion from
the queue. Genuine approach progress renews its timer. Unrelated old task progress
no longer instantly cancels a new owner. Queue membership can still be a long
commitment; this revision does not add a learned cancellation action.

### Assignment

The dispatcher constructs all feasible robot–order pairs and greedily selects
them by order priority, deadline, and travel energy before assigning. Each robot
and order is used at most once. Exact ties rotate reproducibly with scenario seed
and simulation tick. A nearby higher-ID robot can receive work before a distant
lower-ID robot. Repeated decision preparation at the same tick preserves decisions.
This is a hand-written greedy matcher, not globally optimal matching or learned
task allocation.

### Observations

The default observation expands from 328 to 346 features. Four appended values
report front/rear FOV, configured dropout probability, and configured sensor
latency. Fourteen appended values encode two actual executed actions, latest first.
Initial history is all zeros. A blocked move or rejected request reports WAIT;
an accepted request retains its request marker. No remote robot/queue state is
added to the actor. Configured dropout is known; the particular dropped cells and
future randomness are not disclosed. The recurrent actor retains its existing
previous-issued-action input as well.

### Curriculum

| Stage | Ordinary training episode |
|---|---|
| C0 | One worker, simplified shelf/pick/drop setup, one order |
| C0.5 | One worker on the full rack map, two orders |
| C0.75 | Two workers on the full rack map, four orders |
| C1 | Full fleet, full rack map, initially full batteries |
| C2 | Add initial SOC/capacity/consumption variation |
| C3 | Add fault/repair events |
| C4 | Add later demand and sensor variation |
| C5 | Same ordinary generator as C4; broader named training mixture |

Worker identities and headings rotate in the bridges. C1–C3 have zero sensor
latency/dropout. C4/C5 retain randomized sensing and now disclose its configuration.
The passing-bay fixture rotates the physical participants and starting headings.
A new `HND_TRANSFER` prerequisite explicitly authorizes receiver service at T1,
requires physical cargo transfer, and then requires delivery. This is a handover
execution prerequisite, not learned discovery of when to authorize a handover.

Stage transitions resume the preceding actor, mixer, optimizer and replay; earlier
experience retains its configured replay allocation. Exploration decay starts
again relative to each new stage's step origin, while same-stage resume retains
that origin. No BC is repeated after the first stage by the curriculum script.

### Promotion and retention

Every periodic and final curriculum evaluation checks all preceding generic
stages and the union of their named prerequisites. Generic checks retain the
95%-delivery, zero-executed-collision and zero-depletion rule with a Wilson lower
bound of at least 0.8 across at least 20 development scenarios. Named prerequisites
use their case-specific operational success tests, the same statistical lower
bound, and at least `max(20, 4*N)` cases to cover the scheduled worker/headings.
Parking must actually clear to parking; handover must actually transfer cargo.
At 20 cases a single failure prevents the required Wilson bound.

Partial delivery greater than zero is deliberately insufficient for promotion.
Periodic failures allow training to continue within its declared stage budget;
a failed final stage check stops progression. Passing smoke prerequisites would
still not replace independent training seeds or frozen final/OOD evaluation.

The GPU-capable script supports `--device cuda`, periodic `--eval-every`, explicit
skill-case counts and exploration budgets. It saves research artifacts per run
instead of publishing canonical deployment weights. Its uniform 100k-step default
per stage is an 800k-step pilot cap. The YAML research plan keeps a separate
2M-step total by reallocating budgets to the two bridge stages.

## Known limits

Parking is still required by COO02 while the runtime reward can favour cheaper
waiting. That known expected failure is retained; the new gate must block instead
of falsely certifying parking. Charger outages remain outside the ordinary
training primitive mix. Critic state still omits some physical/queue/timer details.
C5 ordinary scenarios have not been made more difficult than C4. A new observation
does not prove long-horizon navigation, coordination or convergence.

## Validation

Regression coverage includes raw-action charging bypass, emergency loaded/unloaded
handling, nearer-robot assignment, reproducible ties, repeated decision preparation,
queue promotion, actual action feedback after blocked moves, bridge worker/heading
coverage, physical handover, named-skill failure despite generic success, checkpoint
incompatibility, and curriculum warm-start/failure reporting. Measured validation
artifacts are saved separately under `outputs/guardrail_validation_20261026/`.

Measured checks: 185 CPU-suite tests passed, three CUDA-only tests skipped there,
and one known parking expected failure; 20 selected tests passed in the CUDA
environment, including actual IQL/VDN/QMIX GPU updates. The planner completed all
orders without depletion or executed collisions in 20/20 cases for each bridge
stage, covering the scheduled worker/headings. Those are planner results.

A separate imitation-assisted tiny N=1 GPU smoke ran 400 steps, resumed the same
stage to 600 steps, and exercised a stage-change resume to 800 total steps / 193
TD updates. Each final five-case development check was 100%. The stage-change
smoke deliberately retains `tiny=True` to isolate resume mechanics: its C0.5 label
does not mean it trained full-map solo navigation. Stage step origins were 0, 0,
and 600 respectively, and source hashes stayed unchanged. No complete V4
curriculum, full-warehouse learned-policy evaluation, or frozen final test has
been completed. Source and measured artifacts are archived with the validation.
