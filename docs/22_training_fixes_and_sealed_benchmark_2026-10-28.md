# Training Defect Root-Cause Fixes & Sealed Benchmark (`2026-10-28`)

## 1. Root Causes Identified and Resolved

### 1.1 Interrupted 3-Tick Service Mode-Lock (`src/environment/logistics.py`, `src/environment/physics.py`)
- **Symptom**: During online RL exploration ($\epsilon > 0$), robots entering a 3-tick `PICKING` or `DROPPING` service at a pickup or drop bay occasionally took an exploratory movement action on tick 2 or 3, moving off the bay.
- **Root Cause**: `physics.py` reset `service_remaining=0, service_key=None` when a robot did not choose `SERVICE` (`5`), but left `robot['mode']` set to `'PICKING'` or `'DROPPING'`. Once off the bay, `SERVICE` was masked out, permanently locking the robot in `'PICKING'` or `'DROPPING'` mode for the remainder of the episode.
- **Fix**: Added `_clear_interrupted_service(robot)` in `logistics.py` (called from `physics.py`) to revert `'PICKING'` $\to$ `'TO_PICKUP'` (or `'IDLE'`) and `'DROPPING'` $\to$ `'TO_DROP'` (or `'IDLE'`), and enforced the invariant in `core.py:assert_invariants()`.

### 1.2 Low-SoC Charger Rejection Loop (`src/environment/logistics.py`)
- **Symptom**: Unloaded robots with $\text{SoC} \le 0.30$ repeatedly selected `REQUEST_CHARGE` (`6`) up to 90+ times in a single episode without moving.
- **Root Cause**: `voluntary_charge_allowed()` exposed `action_mask[6] = True` whenever an unloaded, unassigned robot had $\text{SoC} \le 0.30$, but `request_charge()` rejected the request if the pessimistic turn-padded `route_energy` bound exceeded `energy_eu` or if all chargers were offline/full.
- **Fix**: Allowed unloaded low-SoC robots (`energy_eu > 0`) to fall back to the nearest `online_open` charger when the pessimistic turn-padded bound is tight, and added a 5-tick `charge_retry_after` backoff when all chargers are offline or full.

### 1.3 Transfer Bay Local Sensor Visibility (`src/environment/sensors.py`)
- **Root Cause**: `transfer_bays` (`grid == 6`) were not mapped into any channel of the $5 \times 5 \times 12$ egocentric sensor grid (`channels[5]` only checked `code == 4`).
- **Fix**: Mapped `code in (4, 6)` to `channels[5] = 1`, preserving `OBS_DIM == 348`.

### 1.4 QMIX Actor/Mixer Gradient Starvation & Post-BC TD Margin Collapse (`src/training/learner.py`, `src/training/runner.py`)
- **Root Causes**:
  1. Uncalibrated `MonotonicMixer` at the end of `behavior_clone()` produced large initial TD gradients during RL; clipping `actor + mixer` parameters jointly (`clip_grad_norm_(self.parameters, 10.0)`) shrank actor gradients by up to $10\times$.
  2. In `train_step()`, 1-step Bellman bootstrapping with $\gamma = 0.99$ pulled suboptimal exploratory action values ($Q \approx -8.0$) up toward $+6.5$, compressing the actor's action-logit margin below the `sensor_dropout = 0.05` observation noise floor.
  3. Starting post-BC online RL at `epsilon_start = 0.30` caused 5 robots to take random actions every 3.3 ticks, interrupting 3-tick services and clogging narrow 1-cell aisles.
- **Fixes**:
  1. Clipped `self.actor.parameters()` and `self.mixer.parameters()` independently in both `behavior_clone()` and `train_step()`.
  2. Added a frozen-actor mixer TD calibration pass at the end of `behavior_clone()` so `mixer` and `target_mixer` are aligned with the cloned actor before online RL starts.
  3. Balanced unweighted demonstration cross-entropy regularization against 1-step TD loss in `train_step()` (`0.2 * td_loss + (bc_aux_weight + 1.0) * imitation`) to preserve action margins while calibrating value estimates.
  4. Capped post-BC online exploration at `0.10` when demonstrations are present and added best-RL checkpoint restoration (`_eval_rank`) across development evaluations.
  5. Cached joint pair plans in `src/baselines/pair_planner.py`, accelerating DAgger collection by $>10\times$.

---

## 2. Empirical Results (`N=5`, `C5` Mixed-Stock)

### 2.1 Training & Development Gate (`Seeds 300000..300004`)

| Algorithm | Run ID | RL Steps / Updates | Dev Eval @ 3,000 Steps | Dev Eval @ 6,000 Steps | Final Dev Completion | Avoidable Depletions | Completed Charges (per seed) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IQL** | `iql_N5_seed11_1793128930179731100` | `6,000` / `750` | `100.0%` | `100.0%` | **`100.0%`** | `0` | `[6, 6, 5, 5, 8]` |
| **VDN** | `vdn_N5_seed11_1793129302304910400` | `6,000` / `750` | `100.0%` | `100.0%` | **`100.0%`** | `0` | `[6, 6, 5, 5, 7]` |
| **QMIX** | `qmix_N5_seed11_1793129635091427600` | `6,000` / `750` | `100.0%` | `99.5%` | **`100.0%`** | `0` | `[6, 6, 5, 5, 6]` |

### 2.2 Held-Out Sealed Benchmark (`Seeds 200000..200004`)

Saved in `outputs/evaluation/sealed_mixed_stock_comparison.json`:

| Policy | Sealed Completion Mean | Per-Seed Completion | Mean Throughput (tasks/step) | Mean Delivery Time (ticks) | Mean Energy Used (EU) | Conflicts / Robot-Tick | Avoidable Depletions | Executed Collisions | Completed Charges (Total) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Planner** (`NearestTaskDispatcher`) | **`100.0%`** | `[1.0, 1.0, 1.0, 1.0, 1.0]` | `0.0667` | `37.21` | `343.48` | `0.0001` | `0` | `0` | `27` |
| **IQL** (`iql_policy_N5.pt`) | **`100.0%`** | `[1.0, 1.0, 1.0, 1.0, 1.0]` | `0.0667` | `50.99` | `354.52` | `0.0091` | `0` | `0` | `26` |
| **VDN** (`vdn_policy_N5.pt`) | **`100.0%`** | `[1.0, 1.0, 1.0, 1.0, 1.0]` | `0.0667` | `50.36` | `351.82` | `0.0096` | `0` | `0` | `29` |
| **QMIX** (`qmix_policy_N5.pt`) | **`99.0%`** | `[1.0, 0.95, 1.0, 1.0, 1.0]` | `0.0660` | **`46.51`** | **`351.12`** | `0.0096` | `0` | `0` | `27` |
