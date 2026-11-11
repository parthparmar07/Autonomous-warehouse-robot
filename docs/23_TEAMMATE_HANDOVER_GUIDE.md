# Project 25 — Teammate Handover Guide

**Repository:** https://github.com/parthparmar07/Autonomous-warehouse-robot · branch `handover-completion`
**Contracts:** environment `warehouse_v6_research`, observation `egocentric_range_v3_pose_feedback`
(348 features, 7 actions), workload `mixed_stock_v1`.

This guide replaces an earlier draft that circulated outside the repo. That draft had several
errors; they are corrected here and listed in §5.

---

## 1. Set up and check that everything works

```powershell
git clone https://github.com/parthparmar07/Autonomous-warehouse-robot.git
cd Autonomous-warehouse-robot
git switch handover-completion

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
.\.venv\Scripts\python.exe -m pip install codecarbon evidently scipy pandas

.\.venv\Scripts\python.exe -m pytest -q                  # expect 229 passed
.\.venv\Scripts\python.exe -m scripts.learning_smoke     # tiny N=1 BC+QMIX check; completion 0.0 -> 1.0
```

`learning_smoke` rewrites the tracked `artifacts/warehouse_v6_research/tiny/` checkpoint. Run
`git restore artifacts/warehouse_v6_research/tiny outputs/evaluation/learning_smoke.json` afterwards
unless you mean to update it.

**Live dashboard** (two terminals):

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000 --workers 1
.\.venv\Scripts\python.exe -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8080. If Windows refuses port 8080, serve the frontend on another port and set
`WAREHOUSE_CORS_ORIGINS` to include it before starting the API.

## 2. Where each deliverable lives

| Requirement | Deliverable | Location |
|---|---|---|
| OBJ-1 / NFR-1 | QMIX, VDN, IQL at N=5, ≥ 80% sealed completion | `artifacts/warehouse_v6_research/mixed_stock_v1/*_N5.pt`; [MODEL_CARD.md](MODEL_CARD.md) §4 |
| OBJ-2 / OBJ-3 / NFR-2/4/5 | 9-cell scalability benchmark (planner/IQL/QMIX × N=5/10/20), 20 sealed seeds, bootstrap CIs | [scripts/evaluate_scalability.py](../scripts/evaluate_scalability.py) → `outputs/evaluation/scalability_N5_N10_N20.json` |
| OBJ-4 / FR-3 / NFR-3 / ETH-3 | Non-learned safety layer; 20 sealed adversarial episodes per fleet size | Mask: [physics.py:249](../src/environment/physics.py#L249); refusal: `WarehouseEnv.step`; tests: [tests/test_safety_layer.py](../tests/test_safety_layer.py); CI: [.github/workflows/verify.yml](../.github/workflows/verify.yml) |
| OBJ-5 / NFR-4 | Reward-shaping Optuna study, completion vs collision Pareto frontier | [scripts/tune.py](../scripts/tune.py) → `outputs/tuning/*.json` |
| FR-5 / EO-3 | FastAPI service, Docker, ONNX actors with parity checks | [src/deployment/api.py](../src/deployment/api.py), [Dockerfile](../Dockerfile), `artifacts/onnx/warehouse_v6_research/` |
| FR-6 | MLflow tracking | `--mlflow` flag on `src.training.runner` |
| EO-5 | Evidently PSI task-queue drift monitor (retrain if PSI > 0.25) | [scripts/monitor_drift.py](../scripts/monitor_drift.py) → `outputs/monitoring/drift_report.json` |
| ETH-4 | CodeCarbon footprint of every training/eval run | [scripts/track_emissions.py](../scripts/track_emissions.py) → `outputs/emissions/emissions.csv` |
| EO-4 | 3D PyBullet demo | [scripts/record_demo.py](../scripts/record_demo.py) + [scripts/demo_pybullet.py](../scripts/demo_pybullet.py) → `outputs/demo/qmix_N5_3d.gif` |
| ETH-1..4 | Responsible AI & sustainability report | [24_RESPONSIBLE_AI_AND_SUSTAINABILITY_REPORT.md](24_RESPONSIBLE_AI_AND_SUSTAINABILITY_REPORT.md) |
| EO-6 | Model card, data card | [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md) |

## 3. How to reproduce the results

### Training (GPU)

Every fleet size uses the published N=5 recipe. **Pass these flags explicitly**:
`configs/runtime_training.yaml` has different defaults (train-every 4, ε from 1.0), and a run left on
those defaults is not comparable with the N=5 checkpoints.

```powershell
# one lane per algorithm; runs inside a lane are sequential, both lanes share the GPU
powershell -File scripts\run_scalability_queue.ps1 -Runs "qmix:10,qmix:20"
powershell -File scripts\run_scalability_queue.ps1 -Runs "iql:10,iql:20"
powershell -File scripts\run_scalability_queue.ps1 -Runs none -WaitFor "<pids of both lanes>" -Evaluate
```

A lane skips any checkpoint that already exists. Plan on these costs:

- **Memory:** each run holds 1.5–3 GB of RAM, and RAM, not the GPU, decides how many lanes fit.
- **GPU use stays low:** the network is small and the warehouse simulation runs on the CPU, so
  GPU utilisation of a few percent is expected.
- **Time:** on an RTX 4050 laptop, N=10 took about 25 min.

### Benchmark, drift, tuning, export

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_scalability            # needs N=10/20 checkpoints
.\.venv\Scripts\python.exe -m scripts.monitor_drift                   # ~2 min
.\.venv\Scripts\python.exe -m scripts.tune --trials 16 --steps 3000   # ~5 min per trial
.\.venv\Scripts\python.exe -m scripts.export_onnx --checkpoint artifacts/warehouse_v6_research/mixed_stock_v1/qmix_policy_N5.pt --output artifacts/onnx/warehouse_v6_research/mixed_stock_v1/qmix_policy_N5.onnx
```

### 3D demo (Windows)

PyBullet has no pip wheel for Windows, but conda-forge has one. No Docker is needed.

```powershell
# one-time: standalone micromamba (no admin)
curl.exe -L -o $env:USERPROFILE\micromamba\micromamba.exe https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
& $env:USERPROFILE\micromamba\micromamba.exe create -y -n pybullet -c conda-forge python=3.11 pybullet=3.25 numpy pillow imageio

.\.venv\Scripts\python.exe -m scripts.record_demo --output outputs/demo/trajectory_qmix_N5.json
& $env:USERPROFILE\micromamba\envs\pybullet\python.exe scripts/demo_pybullet.py outputs/demo/trajectory_qmix_N5.json --gif outputs/demo/qmix_N5_3d.gif
# add --gui for an interactive window
```

## 4. Status and what is left

See the [model card](MODEL_CARD.md) and the RAI report for the final numbers.

| Item | Status |
|---|---|
| CUDA bug: IQL/VDN crashed on GPU (`learner.py`, mixer-norm tensor created on the CPU) | Fixed |
| ONNX export of the six N=5 policies | Done, parity verified |
| Drift monitor | Done: control 0.04 max PSI (no trigger); 3× demand triggers (queue length 2.4 → 44) |
| 3D demo | Done |
| N=10 / N=20 training, sealed scalability benchmark, Optuna study | Running at handover; results go into the model card and RAI report ETH-4 |
| Multiple training seeds per cell | **Not done.** CIs cover scenario variance only. Training 2+ seeds per cell would allow `paired_hierarchical_interval` to report training-run variance. |
| Out-of-distribution layouts, sim-to-real | Not in scope this semester |

## 5. Corrections to the earlier draft

1. **Safety mask location.** `src/safety/action_mask.py` is *not* the enforced mask. It holds the
   pre-v6 five-action mask, still exercised by old tests; only its `orientation_delta` helper is
   used by the environment. The enforced mask is `get_action_mask` in `src/environment/physics.py`.
2. **Test count.** On `main` (a597539), 226 of 228 tests passed. The 2 failures were the CUDA bug
   above, and that bug would have crashed the draft's own IQL `--device cuda` commands.
3. **Training commands.** The draft used `--horizon 400` (sealed evaluation uses 600) and left the
   YAML defaults in place, so results would not have matched N=5. Use §3.
4. **`tune.py`.** The draft said it tunes reward weights. It only tuned learning rate and gamma.
   It now tunes conflict cost, delivery reward, potential weight and learning rate against two
   objectives.
5. **"Outbound" N=5 checkpoints.** These are the same actor weights as the mixed-stock checkpoints,
   repackaged with an outbound-sized mixer, not separately trained policies.
6. **3D demo.** The draft's `pip install pybullet` does not work on Windows; see §3.
