# Council adjudication: what C5 failure does and does not show

The environment reviewer, learner reviewer and arbiter reviewed the same code and
failed development runs, exchanged findings and challenged proposed remedies. The
arbiter accepted defects supported by executable behavior or code paths, and
rejected changing scientific acceptance criteria merely to reach a later stage.

## Decisions

| Finding | Adjudication | Action |
|---|---|---|
| C5 completion is poor | It is transfer from policies stopped at C0.5. C3–C5 were absent from the old launch configuration. It is not a measurement of converged C5 training. | New explicit C0–C5 protocol, with core stages followed by robust stages; retain failed prerequisites. |
| Training was still reported running | Parent process inspection found no surviving training processes, despite stale `running` metadata. | Archive interrupted attempts; durable hidden launcher, process/heartbeat evidence and periodic replay checkpoints. |
| Named skill exposure is inadequate | Forty demonstration episodes contain only eight named slots; C1 exposes two of twenty roles per named case. DAgger repeats these roles. | Enumerate family × worker × heading, report actual expert counts, preserve representative replay and prior skills. |
| More QMIX steps are needed | The old navigation budget is small and learning improved, but duration alone does not address missing exposure, interrupted execution or forgetting. | Larger matched ceilings and slower exploration decay, with diagnostic evaluations. No convergence guarantee. |
| QMIX mathematics must be broken | Reviewer found no confirmed defect in positive-weight mixing, legal Double-Q targets or terminal/truncation handling. Solo navigation failure cannot establish a monotonic-factorization limitation. | Preserve algorithm math; investigate observed trajectories and assistance/TD transitions. |
| Retention under new stage profiles | Applying the current robust profile to an earlier core C2 evaluation silently changes the retained task. | Record and evaluate each stage with the profile actually used for training. |
| Charging skill starts already docked | Docked charging does not establish request, charger approach and docking competence. | Add off-dock `CHARGE_APPROACH`, rotating roles, both chargers and subsequent delivery. |
| A robot above 30% may have no feasible order | Confirmed design dead zone under the user's action threshold and conservative mission admission. It is not permission to unlock charging early. | Preserve the threshold; report unproductive admission/threshold waiting and test feasibility. |
| C4 and C5 generic mechanisms overlap | Confirmed. Labels do not make equivalent distributions progressively harder. | State overlap explicitly; C5 includes additional composed/long-shift prerequisites. A distinct scenario-distribution change requires its own declared condition. |
| Twenty evaluations at the current Wilson threshold require twenty successes | Correct and stricter than simply 80% delivered orders. | Keep the rule and report failures and delivery fractions separately; do not claim competence by relaxing it. |
| A committed joint route can become stale | An outside robot occupied a planned next cell; the controller kept issuing a forbidden head-on exchange indefinitely. | Invalidate a plan when its next cells are externally occupied or newly blocked; preserve valid physical service. |
| A charging robot depleted despite starting beside its charger | Joint makespan ties sent the low-energy robot eight cells away and back while its charged partner travelled to a distant pickup. This was a baseline planning defect, not evidence that the 30% charge rule caused this failure. | Minimize summed translation energy and waiting cost with a geodesic lower bound, instead of joint makespan. Test charger-side yielding and all previously failing long shifts. Turns remain outside this search cost, so it is an approximation to total execution energy. |

## New experimental contract

`configs/council_full_curriculum.yaml` defines a fresh exploratory,
imitation-assisted and potential-shaped comparison. Its upper limit is 2,730,000
joint RL steps per algorithm/seed, with early advancement or stopping. It is not
a promise that C5 will be reached. C0–C2 use the core profile; C3–C5 use robust
conditions. Both methods use the same role schedule, assistance and stage caps;
actual steps, optimizer updates, expert data and wall time remain separate costs.
Adaptive advancement produces different realized costs and must be reported.

The previous completed V6 seed-11 results and interrupted seed-23 attempts remain
historical evidence. They must not be pooled as repetitions of the new protocol.
Final/OOD evaluations remain outside development selection. No assertion that
every robot is prepared for every situation follows from passing this curriculum.

## Prelaunch and recovery checks

Verify all twenty off-dock charging roles with the planner; complete family/role
coverage; per-stage profile retention; sequence learner and existing simulator
regressions; atomic report/checkpoint writes; preserved optimizer/replay/scenario
cursor; and restart without repeating same-stage imitation. Test a deliberately
interrupted owned run before trusting a long run. Protocol gate streaks restart
conservatively after interruption; this is disclosed rather than called exact
gate continuation. Confirm current process identity and increasing heartbeat or
training counters after launch. A stale matrix is never sufficient evidence.

If the new policy still fails full-map navigation, inspect failed geometries,
role coverage, loops and loss of cloned behavior before changing rewards or
adding more stages. If navigation passes but charging fails, distinguish charge
request, approach, queue, service and return-to-work failures. If isolated skills
pass but composed C5 fails, vary one disruption at a time before combining them.

## Cross-review and execution evidence

The environment and learner reviewers exchanged findings directly and the arbiter
challenged their causal explanations. Cross-review also corrected new-code
mistakes: repeated same-stage BC on restart, loss of recorded assistance costs,
missing skill families after replay subsampling, changed prior-stage profiles,
stale scenario hashes and a duplicate-supervisor restart race. The changes are
not based on a vote that more steps must improve QMIX.

The complete CPU test suite passed **267 tests**, with three CUDA-only skips and
six upstream warnings. A separate CUDA/recovery/coverage selection passed
**12 tests**. A detached CUDA end-to-end smoke job completed 80 RL steps and
20 optimizer updates, wrote recoverable checkpoints and exited with an honest
failed competence gate. Its purpose was process/checkpoint verification, not
learning performance. The launching terminal had already returned while the
hidden supervisor and training child continued.

The full planner feasibility sweep has 440 episodes: 160 generic stage cases
and 280 named skill/role cases. The original failed sweep is preserved separately
from the recheck after planner repairs. Do not infer a zero-order bay-clearance
failure from delivery fraction zero; use its case-specific operational oracle.

The fresh study includes three main training seeds per algorithm plus one seed
each for unshaped and pure-RL controls. It is a new exploratory condition and
does not retroactively complete the interrupted earlier matrix. Small controls
do not support strong significance claims. Check supervisor identity, heartbeat,
child logs and saved learner counters rather than trusting a static status label.

Launch verification: the final planner sweep passed **440/440**, compared with **433/440** before the causal baseline repairs. Training source SHA256 is `4b6199edf51b3f7693549e6f8eac6ba9250c0ca0e891d956c8c9eb919d691ffc`. The new finite detached study is under `outputs/council_training_20261026/`; `status.md` refreshes from the supervisor. Source/configuration remain frozen while its jobs run. The initial jobs start at C0; no new C5 training outcome is claimed at launch.
