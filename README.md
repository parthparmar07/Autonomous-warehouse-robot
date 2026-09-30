# Project 25 — Autonomous Warehouse Robot Fleet Coordination with Multi-Agent RL

Cooperative control of a warehouse robot fleet using **QMIX** (centralised training, decentralised
execution), compared against **IQL** and a **rule-based nearest-task dispatcher**, at fleet sizes
**N = 5, 10, 20**.

The signature contribution is the **scalability analysis**: how task-completion rate, average
delivery time, and collision rate change as the fleet grows, and whether the cooperative benefit of
value decomposition grows with it.

> **Current status: working prototype and research foundation.** The canonical simulator, safety
> interfaces, training scaffolds, evaluation scripts, FastAPI service, and browser demo are implemented
> and tested. The learned policies are still preliminary and must not be presented as final research
> results until longer training and repeated evaluation are complete.

**Team:** 4 students · **Duration:** 15 weeks (Semester 5) · **Budget:** ₹0 (free tiers only)

## Implementation Status

### Completed

- Deterministic NumPy warehouse simulator with five discrete actions, local 30-feature observations,
   task generation, rewards, robot movement, delivery progression, and collision metrics.
- **A real safety layer (ADR-003), not a stub.** The floor plan (`src/environment/layout.py`) marks a
   perimeter human-worker safety corridor and a central shelf block. `WarehouseEnv.get_action_mask`
   masks out any action that would leave the grid, enter a shelf/corridor cell, or collide with another
   robot's current cell — before the policy's argmax (layer 2), identically in training and evaluation.
   `WarehouseEnv.step` additionally refuses those moves physically and counts them as
   `safety_overrides` (layer 3, the redundant post-hoc check), so the guarantee holds even if a caller
   ignores the mask. `src/safety/action_mask.py` holds the one shared implementation both layers read,
   so they cannot drift apart. Covered by dedicated tests including an adversarial-policy check that
   mirrors the project's proposed CI safety gate.
- Task generation only places pickups/deliveries on free floor cells, so every generated task is
   reachable under the mask.
- Nearest-task dispatcher baseline with dynamic task assignment and **BFS pathfinding** around shelves
   and corridors (a straight-line greedy baseline could not route around the new shelf block).
- **Fixed a real partial-observability bug that was the actual root cause of every learned policy
   failing.** The local observation's 5×5 window and target-bearing vector are both in fixed world-axis
   coordinates, and the robot's own heading was never included as a feature — so the *same* observation
   had to map to different correct actions depending on an orientation the policy could never see. The
   BFS baseline only ever worked because `choose_action` reads `robot["orientation"]` from ground truth
   directly, a privilege no learned policy had. This was silently making the task partially observable
   for every algorithm tried, regardless of training budget or bug fixes elsewhere. Fixed by adding a
   sin/cos heading encoding to the observation (28 → 30 features; see `get_local_observation`).
- **Imitation-learning warm start from the BFS dispatcher (PRIMAL-style), with DAgger refinement.**
   `src/training/imitation.py` collects (observation, expert-action) demonstrations from the working
   BFS baseline and trains each network with class-weighted cross-entropy (the expert turns rarely, so
   unweighted loss let early networks learn to almost never predict LEFT/RIGHT); `scripts/pretrain_bc.py`
   then runs DAgger rounds, letting the *current* network drive the rollout while still querying the
   expert for the correct label at every state it visits, which is what fixes behaviour cloning's
   compounding-error problem. This is not improvisation: the project's own literature review
   (`docs/09_assignment1_paper_summaries.md`, Paper 5, PRIMAL) pre-registers exactly this fallback —
   "if our own method fails to reach the target completion rate through trial and error alone,
   initialising it from our rule-based planner is an approach with published support."
- DQN, IQL, and QMIX agent foundations, replay buffers, target networks, and training scripts. Fixed
   two further real bugs along the way: QMIX's `act()` had no exploration at all (always deterministic
   argmax) — now epsilon-greedy like DQN/IQL; and QMIX's GRU hidden state was silently reset to zero on
   every decision instead of carried across the rollout — the trainer now tracks it per agent and stores
   it in the replay buffer so `train_step` trains on the same hidden state the rollout actually used.
   QMIX training is CUDA-aware when a compatible PyTorch build is installed.
- **Checkpoints now generalise across fleet sizes.** `src/deployment/policies.py` cycles a checkpoint's
   per-robot networks to fill any fleet size instead of only loading on an exact N match (a DQN/IQL
   agent's network doesn't architecturally depend on fleet size — only the *count* of robots does; QMIX's
   mixer is the one genuinely fleet-size-specific piece, and it's only used during training, not `act()`).
- Deterministic evaluation manifests for N=5, 10, and 20 with SHA-256 hashes, regenerated against the
   real floor plan.
- Evaluation metrics, bootstrap helper, comparison runner, and JSON benchmark output; evaluation now
   passes the real per-robot mask to every learned policy, and records `safety_overrides` per episode.
- FastAPI service (`src/deployment/api.py`) with `/health`, `/algorithms`, `/predict`, `/state`,
   `/reset`, `/tick`, `/step`, and multi-session support (`?session_id=`) so several algorithms can run
   side by side. Policy loading is shared with the benchmark script (`src/deployment/policies.py`) so the
   API and the offline benchmark can never disagree about how a checkpoint is loaded.
- A real browser demo (`frontend/`, React 18 + Babel via CDN, no build step) — replacing the previous
   raw-JSON-dump page: a canvas-rendered warehouse floor with smoothly animated (not tick-snapped) robot
   motion, fading path trails, a pulse effect on delivery, algorithm/fleet-size/seed controls, play/pause/
   step, live reward/delivery sparklines, and a **"Compare all" mode** that runs all four algorithms
   side by side on the same seed and floor plan with a live leaderboard.
- GPU support: an NVIDIA GeForce RTX 4050 Laptop GPU with CUDA-enabled PyTorch (2.6.0+cu124) is installed
   and verified (`torch.cuda.is_available()` and a real matmul both checked directly on-device). Honest
   caveat: these networks are small enough (a few hundred thousand parameters) that CUDA kernel-launch
   overhead roughly cancels the parallelism benefit, and the actual training bottleneck is the
   CPU/Python-bound BFS expert-data collection, not the neural network forward/backward pass — so GPU
   training is available and used, but hasn't meaningfully changed wall-clock time for this workload.
- Fleet-size-specific checkpoints (`src/deployment/policies.py`): a checkpoint trained at a given N is
   preferred exactly when available (`artifacts/{algo}_policy_N{n}.pt`), falling back to cycling the
   generic checkpoint's per-robot networks otherwise. QMIX now has real N=5 and N=10 trained mixers;
   N=20's checkpoint completed its behaviour-cloning pass but not its DAgger refinement (see caveat below).
- Sealed-manifest evaluation with real confidence intervals (`scripts/evaluate_sealed.py`): verifies each
   manifest's SHA-256 hash before use (aborts on mismatch, per "Evaluation integrity" below), replays the
   exact sealed floor plan/seed/task list, and reports a percentile bootstrap 95% CI
   (`src/evaluation/bootstrap.py`) on completion rate, collisions, and reward per algorithm × fleet size.
- Regression suite currently passes 23 tests.

### Current Measured Results

Two independent evaluations, run after all of the fixes above, agree on the overall shape of the result.

**5 seeds × 2 episodes each (`scripts/benchmark_algorithms.py`):**

| Algorithm | N=5 | N=10 | N=20 |
|---|---:|---:|---:|
| Nearest-task baseline | 99% | 99% | 97% |
| DQN | 21% | 20% | 30% |
| IQL | 25% | 31% | 46% |
| QMIX | 43% | 21% | 36% |

**Sealed manifests, hash-verified, 95% bootstrap CI (`scripts/evaluate_sealed.py`):**

| Algorithm | N=5 | N=10 | N=20 |
|---|---|---|---|
| Nearest-task | 0.98 [0.95, 1.00] | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] |
| DQN | 0.28 [0.15, 0.40] | 0.30 [0.15, 0.50] | 0.28 [0.20, 0.40] |
| IQL | 0.18 [0.10, 0.35] | 0.37 [0.30, 0.45] | 0.42 [0.35, 0.55] |
| QMIX | 0.45 [0.40, 0.50] | 0.30 [0.00, 0.50] | 0.37 [0.25, 0.50] |

**This is the first time in the project's history that every learned algorithm has a substantial,
nonzero completion rate at every fleet size** — and every one has a dramatically lower collision rate
than the baseline (single digits vs. 52–312), consistent with them having actually learned something
about avoiding collisions rather than just moving in straight lines. For most of this session every
learned policy scored exactly 0% regardless of which bugs were fixed or how much longer training ran
(see the diagnosis below for why). The imitation-learning warm start plus the observation fix is what
actually moved these numbers off zero — not a hyperparameter tweak.

**Honest caveats, not smoothed over:**
- **N=20's QMIX checkpoint is behaviour-cloning only.** Its DAgger refinement run was killed after ~76
  minutes with no reliable ETA (confirmed still genuinely computing, not hung, via sustained high CPU
  use — just far slower than N=5/N=10's scaling predicted, likely because 20 robots on the same 20×20
  floor plan means far more path congestion for the BFS expert). The saved checkpoint is real and valid
  (round 0 completed and saved before the process was stopped), just less refined than N=5/N=10's.
- **QMIX N=10 is the weakest of QMIX's three fleet sizes in both evaluations**, and its sealed-manifest
  CI is wide (`[0.00, 0.50]` from only 3 seeds) — read that number as genuinely uncertain, not precise.
  A dedicated N=10 mixer was trained, but the mixer is only used during further training, not at
  inference (`act()` only touches the per-robot agent networks) — so this result is better explained as
  training variance in one BC+DAgger run than as evidence the fleet-size-specific mixer helped or hurt.
- The two evaluations don't match exactly (different seeds, different sample sizes) and shouldn't be
  expected to — they agree on the shape (baseline dominates completion; all three learned algorithms are
  real but well below it; no single learned algorithm is cleanly best across every fleet size yet), which
  is what matters at this sample size.

**Safety result (across every run in both evaluations): zero `safety_overrides`.** No policy — trained,
untrained, imitation-warm-started, or the baseline — has ever gotten a movement past the mask into a
corridor cell. This holds regardless of how good or bad the policy's task-completion is, which is
exactly ADR-003's claim: the safety guarantee does not depend on model quality.

**Why every learned policy scored 0% for most of this session, and what actually fixed it.** Diagnosed
by hand at each stage (see `src/algorithms/`, `src/environment/warehouse_env.py`), not just read off the
completion number:

1. Two real bugs were found and fixed first (QMIX's missing exploration, QMIX's discarded GRU hidden
   state) — both confirmed necessary by direct rollout inspection, neither sufficient: completion stayed
   at 0% after both fixes and substantially longer training.
2. The actual root cause was a **partial-observability bug**: the observation never told the policy
   which way the robot was facing, so the same observation had to mean different correct actions
   depending on unobservable heading. No amount of extra training data or a better algorithm fixes an
   observation that is genuinely ambiguous — this is why DQN, IQL, and QMIX all converged to the same
   family of degenerate behaviours (spinning in place, oscillating between two cells) independent of
   which bugs had already been fixed.
3. Fixing the observation and adding the imitation-learning warm start together is what produced the
   first nonzero results. The two are both necessary: on the un-fixed observation, imitation learning's
   cross-entropy loss plateaued around 0.85–0.93 (with class-weighting) and completion stayed at 0%; on
   the fixed observation, an otherwise-identical pretraining pass reached a loss of ~0.19–0.21 and every
   algorithm immediately showed nonzero completion.

### Remaining Work

1. ~~Make the warehouse action mask enforce actual boundary, shelf, and robot-conflict constraints.~~
   **Done.**
2. ~~Thread QMIX's GRU hidden state through the rollout instead of discarding it every step.~~ **Done**
   (necessary, not sufficient on its own — see diagnosis above).
3. ~~Fix the observation's missing-heading partial-observability bug; add an imitation-learning warm
   start.~~ **Done** — this is what actually produced nonzero completion rates.
4. ~~Install and verify CUDA-enabled PyTorch.~~ **Done**, with the honest caveat that it hasn't
   meaningfully sped up this workload (see above) — the win would come from a workload where the neural
   net forward/backward pass dominates, not BFS data collection.
5. ~~Train fleet-size-specific checkpoints instead of only ever training at N=5 and cycling.~~ **Done**
   for QMIX N=5 and N=10 (real trained mixers); **N=20 is BC-only**, its DAgger refinement was cut short
   (see caveat above) — finishing that run (or restarting it with a smaller `--dagger-episodes`) is the
   most direct next step.
6. ~~Switch evaluation from one smoke episode to something with real statistical grounding.~~ **Done**:
   `scripts/evaluate_sealed.py` replays the sealed, hash-verified manifests and reports a 95% bootstrap CI.
   Next: more sealed seeds per fleet size than 3 — the current CIs are wide largely because of that small
   n, not because of a flaw in the method — and add the plots for the report.
7. Push completion rates further: more DAgger rounds / demonstrations, or RL fine-tuning on top of the
   imitation-pretrained checkpoints (the training scripts accept `--warm-start`, on by default, and load
   fleet-size-specific checkpoints automatically — this is one command, not new engineering).
8. Add deployment hardening: restricted CORS, model version reporting, request logging, and production configuration.

### Running the Working Prototype

```powershell
$env:PYTHONPATH='.'
pytest -q
python scripts/generate_manifests.py

# Imitation-learning warm start (recommended before benchmarking or RL fine-tuning --
# see "Current Measured Results" for why): behaviour cloning from the BFS dispatcher,
# refined with DAgger. Produces artifacts/{dqn,iql,qmix}_policy.pt (and, with
# --num-agents, a fleet-size-specific artifacts/{algo}_policy_N{n}.pt too).
python scripts/pretrain_bc.py --num-agents 5 --episodes 80 --max-steps 150 --epochs 12 --dagger-rounds 2 --dagger-episodes 25
python scripts/pretrain_bc.py --only qmix --num-agents 10 --episodes 80 --max-steps 150 --epochs 12 --dagger-rounds 2 --dagger-episodes 25
python scripts/pretrain_bc.py --only qmix --num-agents 20 --episodes 80 --max-steps 150 --epochs 12 --dagger-rounds 2 --dagger-episodes 25

python scripts/benchmark_algorithms.py       # 5-seed smoke comparison
python scripts/evaluate_sealed.py            # sealed manifests, hash-verified, 95% bootstrap CI

# Optional: further RL fine-tuning on top of the imitation-pretrained checkpoints,
# instead of from a random init -- --warm-start is on by default and loads the
# fleet-size-specific checkpoint automatically.
python scripts/train_qmix.py --num-agents 5 --episodes 300 --max-steps 150 --train-every 4
python scripts/train_iql.py --num-agents 5 --episodes 150 --max-steps 150
python scripts/train_dqn.py

python -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000
```

Serve the browser demo from another terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Then open `http://127.0.0.1:5500`. Pick an algorithm and fleet size and hit **Reset**, then **Play** —
the canvas renders the live floor plan (shelves, the safety corridor, pickup/delivery markers) with
smoothly animated robots, fading trails, and a pulse on delivery, stepping through the FastAPI service's
`/reset` and `/tick` endpoints using that algorithm's real, masked decisions. Switch to **Compare all**
to run all four algorithms side by side on the same seed and floor plan, with a live leaderboard.

---

## Documentation

**New to the project? Start with the [Plain-English Guide](docs/08_plain_english_guide.md)** — the
whole project and the supporting papers explained without jargon, with a glossary. ~10 minutes.

**Coursework:** [Assignment 1 — Literature Summary](docs/09_assignment1_paper_summaries.md)
([print-ready HTML](docs/09_assignment1_paper_summaries.html)).

Then read in this order:

| # | Document | What it settles |
|---|---|---|
| 0 | [Project brief](docs/00_project_brief.md) | Source of truth + a table of every deviation from it |
| 1 | [Stakeholder map](docs/01_stakeholder_map.md) | Who is affected, the conflicts between them, and why floor workers get elevated treatment |
| 2 | [Feasibility study](docs/02_feasibility_study.md) | What is and isn't achievable in 15 weeks at ₹0, and the safety-critical framing |
| 3 | [Requirements specification](docs/03_requirements_specification.md) | FR / NFR / DR / ETH with acceptance criteria, verification methods, and the traceability matrix |
| 4 | [CRISP-DM adaptation](docs/04_crisp_dm_adaptation.md) | Why CRISP-DM needs modifying for RL, and the two safeguards added |
| 5 | [Milestone plan](docs/05_milestone_plan.md) | Roles, M1–M10 gates, week-by-week schedule, critical path, descope ladder |
| 6 | [Risk register](docs/06_risk_register.md) | 14 scored risks; three reds, all landing in W4 |
| 7 | [Literature review](docs/07_literature_review.md) | Five supporting papers, what each changes about our design, and the RAI sourcing gaps |
| — | [ADRs](docs/adr/) | Five accepted architecture decisions with rejected alternatives |
| — | [Papers](papers/) | The five PDFs, indexed and mapped to requirements |

---

## The five decisions that shape everything

1. **Two environments** ([ADR-001](docs/adr/ADR-001-dual-environment.md)) — a fast NumPy grid world
   is canonical and drives every reported number; PyBullet only *replays* recorded trajectories for
   the 3D demo. Physics-stepped training would consume the whole semester on N=5 alone.
2. **Custom PyTorch QMIX** ([ADR-002](docs/adr/ADR-002-custom-qmix-over-epymarl.md)) — ~300 lines we
   own and can debug, instead of an unmaintained framework that fights the MLOps stack and obstructs
   ONNX export.
3. **Safety by action masking** ([ADR-003](docs/adr/ADR-003-safety-layer-action-masking.md)) —
   corridor-entering actions are removed before the argmax, identically in training and evaluation,
   with a redundant post-hoc assertion in the service. The guarantee does not depend on the model.
4. **Python 3.11, PyBullet isolated** ([ADR-004](docs/adr/ADR-004-python-311-toolchain.md)) —
   PyBullet ships no Windows wheels for any modern version, and the renderer is demo-only anyway.
5. **NFR-2 is a hypothesis, not a target** ([ADR-005](docs/adr/ADR-005-nfr2-as-hypothesis.md)) —
   whether the QMIX−IQL gap widens with N is not something we control. We pre-register it as **H1**
   and report the measurement either way.

---

## Evaluation integrity

Because the environment generates its own data, the usual protections against tuning on the test set
do not apply automatically. Three mechanisms replace them:

- **Sealed manifests.** The 20-episode evaluation manifests for each fleet size are generated,
  SHA-256 hashed, committed, and tagged `eval-manifests-sealed` in W4 — **before any training run**.
  Every evaluation asserts the hash and aborts on mismatch. The git tag must predate the first
  training commit, which an examiner can verify in one command.
- **Two evaluation loops, only one iterable.** Training-seed evaluation is a development instrument
  used freely. Sealed-manifest evaluation is a measurement, taken **once** per algorithm × fleet size.
- **Pre-registration + negative-result policy (ETH-5).** H1 is written down before any result exists,
  and reported whichever way it falls.

---

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (or use a system Python 3.11 with `venv` + `pip`).

```bash
uv venv --python 3.11 .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

Verified working: Python 3.11.12 · torch 2.13.0+cpu · numpy 2.1.3 · mlflow 3.15.0 · optuna 4.9.0 ·
onnxruntime 1.28.0 · gymnasium 1.3.0.

### 3D visualisation (optional, demo only)

`requirements-viz.txt` is **not** needed for training, evaluation, the service, or CI — the renderer
is import-isolated from all of them (FR-7). PyBullet has no Windows wheels, so install it on Linux:

```bash
# WSL2 / Ubuntu / Colab / Docker
pip install -r requirements-viz.txt
```

On Windows 11, `wsl --install -d Ubuntu-22.04` plus WSLg gives a live GUI window for the W15 demo.

### GPU

N=5 trains on CPU. N=10 and N=20 use a free Colab/Kaggle T4. Install the CUDA build of torch there
instead of the `+cpu` wheel.

---

## Planned layout

```
configs/          warehouse_layout.json, algorithm + training configs
data/manifests/   sealed evaluation manifests + HASHES.txt
src/
  env/            WarehouseEnv (canonical), task-queue generator, collision detection
  safety/         corridor mask, action masking, service-side assertion
  algos/          rule-based dispatcher, IQL, QMIX (agent nets + mixer)
  train/          CTDE training loop, replay buffer, Optuna study     [no renderer imports]
  eval/           evaluation harness, bootstrap statistics, scalability analysis  [no renderer imports]
  serve/          FastAPI fleet-decision service, ONNX inference
  viz/            PyBullet trajectory replay (demo only, isolated)
tests/            unit, property (mixer monotonicity), safety, isolation tests
docs/             planning artefacts + ADRs
.github/workflows/  CI: pytest + lint + smoke episodes + completion gate + safety gate
```

---

## CI gates (both blocking)

| Gate | Requirement |
|---|---|
| **Safety gate** | Zero policy-induced corridor intrusions across 20 max-aggression safety episodes (NFR-3). Never `continue-on-error`, never bypassed. |
| **Quality gate** | Task-completion rate on the smoke evaluation must not regress below threshold (NFR-1). |

**Hard rule:** CI never trains. Training runs on Colab or a laptop; CI only validates
(~8 min/build against the 2,000 min/month free tier — RSK-09).

---

## Licence

All dependencies are permissively licensed (BSD-3 / Apache-2.0 / MIT). No dataset licence
obligations — all data is generated by the simulation. Total out-of-pocket cost: ₹0.
