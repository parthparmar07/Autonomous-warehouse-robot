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

- Deterministic NumPy warehouse simulator with five discrete actions, local 28-feature observations,
   task generation, rewards, robot movement, delivery progression, and collision metrics.
- Safety/action-mask interfaces used by policy inference and the API.
- Nearest-task dispatcher baseline with dynamic task assignment.
- DQN and IQL agent foundations, replay buffers, target networks, and training scripts.
- QMIX agent networks, monotonic mixer, replay buffer, target networks, centralized loss, and optimizer
   update path. QMIX training is CUDA-aware when a compatible PyTorch build is installed.
- Deterministic evaluation manifests for N=5, 10, and 20 with SHA-256 hashes.
- Evaluation metrics, bootstrap helper, comparison runner, and JSON benchmark output.
- FastAPI service with `/health`, `/predict`, `/state`, and `/step` endpoints.
- Optional DQN checkpoint loading for API inference with a deterministic fallback when no checkpoint exists.
- Browser demo connected to the live state, prediction, and simulation-step endpoints.
- GPU support verified on an NVIDIA GeForce RTX 4050 Laptop GPU with CUDA-enabled PyTorch.
- Regression suite currently passes 14 tests.

### Current Measured Results

The current benchmark is a preliminary smoke comparison using one seed and one episode per algorithm
and fleet size. It is useful for checking the pipeline, not for final claims.

| Algorithm | N=5 completion | N=20 completion | Interpretation |
|---|---:|---:|---|
| Nearest-task baseline | 95% | 100% | Strong completion, but collision counts rise sharply |
| IQL checkpoint | 0% | 0% | Briefly trained; not yet effective |
| QMIX checkpoint | 0% | 0% | Smoke-trained only; not a final result |
| DQN checkpoint | 0% | 0% | Briefly trained; not yet effective |

The baseline recorded 1 collision at N=5 and 140 at N=20 in the current smoke run. This makes the
collision metric important: completion alone is not sufficient evidence of a safe policy.

### Remaining Work

1. Make the warehouse action mask enforce actual boundary, shelf, and robot-conflict constraints.
2. Train QMIX for longer on N=5 using the CUDA device, with checkpoint metadata and repeatable seeds.
3. Train separate N=20 policies; QMIX mixer weights are fleet-size-specific.
4. Evaluate each algorithm over repeated episodes and seeds using sealed manifests rather than one smoke episode.
5. Add confidence intervals, final plots, and a results table for the report.
6. Improve the browser demo from JSON output to a visual warehouse grid with robot/task status.
7. Add deployment hardening: restricted CORS, model version reporting, request logging, and production configuration.

### Running the Working Prototype

```powershell
$env:PYTHONPATH='.'
pytest -q
python scripts/generate_manifests.py
python scripts/benchmark_algorithms.py
python scripts/train_qmix.py --episodes 3 --max-steps 30 --train-every 5
python -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000
```

Serve the browser demo from another terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Then open `http://127.0.0.1:5500`.

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
