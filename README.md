# Autonomous Warehouse Robot Fleet Coordination (`warehouse_v6_research`)

Structured multi-agent warehouse simulation and CTDE reinforcement learning (`QMIX`, `VDN`, `IQL`) with fixed-aisle shelves, pickup/drop bays, transfer bays, charging docks, human-worker safety corridors, egocentric local observations (`OBS_DIM = 348`), and a FastAPI + HTML5 dashboard.

## Architecture & Domain Highlights

- **Structured Warehouse (`warehouse_v6_research`)**:
  - $20 \times 20$ structured grid with shelf rows (`1`), human-worker safety corridors (`2`), pickup bays (`3`), drop bays (`4`), charger docks (`5`), transfer bays (`6`), and staging/parking cells (`7`).
  - Supports both outbound fulfillment and stock-backed mixed inbound/outbound workflows (`mixed_stock_v1` in [`src/environment/mixed_logistics.py`](src/environment/mixed_logistics.py)).
  - Hard safety layer ([`src/safety/action_mask.py`](src/safety/action_mask.py) and [`src/environment/physics.py`](src/environment/physics.py)) masks and physically blocks any entry into human-worker safety corridors.
  - Realistic battery dynamics: movement, loaded-movement, turning, waiting, and 3-tick bay service energy costs; voluntary charging gated at $\text{SoC} \le 30\%$ when unloaded and unassigned.
- **Egocentric Observations (`egocentric_range_v3_pose_feedback`, `OBS_DIM = 348`)**:
  - $5 \times 5 \times 12$ local FOV/occlusion grid (with configurable sensor dropout and latency), proprioception, mode one-hot encoding, sensor configuration, and 2-step executed action feedback.
- **Matched Recurrent CTDE Learners ([`src/training/learner.py`](src/training/learner.py))**:
  - Shared recurrent GRU actor (`SharedAgent`), legal masked Double-Q bootstrapping, burn-in sequence replay (`SequenceReplay`), and algorithm-specific value factorization (`MonotonicMixer` for `QMIX`, additive sum for `VDN`, independent targets for `IQL`).
  - Pre-calibrated mixer TD targets during `BC + DAgger` warm-start, separate actor/mixer gradient clipping, and demonstration cross-entropy regularization to preserve action logit margins under partial observability.

---

## Held-Out Sealed Benchmark (`N=5`, `C5` Mixed-Stock, Seeds `200000..200004`)

Full episode metrics are saved in [`outputs/evaluation/sealed_mixed_stock_comparison.json`](outputs/evaluation/sealed_mixed_stock_comparison.json):

| Policy | Training Kind | Dev Completion (`300000..300004`) | Sealed Completion (`200000..200004`) | Mean Throughput (tasks/step) | Mean Delivery Time (ticks) | Mean Energy Used (EU) | Avoidable Depletions | Executed Collisions | Completed Charges (Total) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Planner** (`NearestTaskDispatcher`) | Privileged map baseline | `100.0%` | **`100.0%`** | `0.0667` | `37.21` | `343.48` | `0` | `0` | `27` |
| **IQL** (`iql_policy_N5.pt`) | `bc_regularized_rl` (`750` updates) | `100.0%` | **`100.0%`** | `0.0667` | `50.99` | `354.52` | `0` | `0` | `26` |
| **VDN** (`vdn_policy_N5.pt`) | `bc_regularized_rl` (`750` updates) | `100.0%` | **`100.0%`** | `0.0667` | `50.36` | `351.82` | `0` | `0` | `29` |
| **QMIX** (`qmix_policy_N5.pt`) | `bc_regularized_rl` (`750` updates) | `100.0%` | **`99.0%`** | `0.0660` | **`46.51`** | **`351.12`** | `0` | `0` | `27` |

- All three decentralized policies execute strictly from local observations (`OBS_DIM = 348`, 5% sensor dropout, 1-tick latency) with **0 executed collisions** and **0 avoidable battery depletions**.
- Among the learned policies, **QMIX** achieves the fastest mean delivery time (**`46.51` ticks**) and lowest mean energy consumption (**`351.12 EU`**).

---

## Tests & Smoke Verification

```powershell
.\.venv-cuda\Scripts\python.exe -m pytest -q
.\.venv-cuda\Scripts\python.exe -m scripts.learning_smoke
```

## GPU Training

```powershell
.\.venv-cuda\Scripts\python.exe -m src.training.runner --algorithm qmix --workload mixed --stage C5 --num-agents 5 --steps 6000 --horizon 600 --bc-episodes 60 --bc-epochs 10 --dagger-rounds 2 --bc-aux-weight 0.25 --potential-weight 2.0 --epsilon-start 0.30 --epsilon-steps 30000 --train-every 8 --eval-episodes 5 --eval-every 3000 --device cuda
```

Run configurations, learning curves, development outcomes, and resumable checkpoints are saved under `outputs/runs/`. Canonical checkpoints with SHA-256 digests are published under `artifacts/warehouse_v6_research/`.

## Dashboard

```powershell
.\.venv-cuda\Scripts\python.exe -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000 --workers 1
.\.venv-cuda\Scripts\python.exe -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:8080`. Missing or incompatible learned policies fail closed and are reported explicitly.
