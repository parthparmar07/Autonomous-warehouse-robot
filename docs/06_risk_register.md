# Risk Register — Project 25

**Phase:** Initiation (W1–W3) · **Status:** Live document, reviewed every Monday stand-up · **Owner:** R3

**Scoring:** Likelihood (L) and Impact (I) on 1–5. Score = L × I. **≥ 15 = red** (active mitigation
required now), **8–14 = amber** (mitigation planned, monitored), **≤ 7 = green** (accept and watch).

---

## 1. Active risks

| ID | Risk | L | I | Score | Mitigation | Trigger / early warning | Owner |
|---|---|---|---|---|---|---|---|
| **RSK-01** | **QMIX fails to converge at N=20** — non-stationarity plus sparse reward plus a harder credit-assignment problem | 4 | 3 | **12** | Train N=5 fully first and validate; curriculum (5→10→20) reusing earlier weights where shapes permit; checkpoint every 50k steps; **pre-agreed descope: report N=20 as attempted-not-converged with the learning curve as evidence** — the brief's own risk register permits this | Episode return flat after 3 M steps at N=20 | R2 |
| **RSK-02** | **Environment too slow / W4 overruns** — W4 is the zero-float critical path | 3 | 5 | **15** 🔴 | Dual-environment split from day 1 (ADR-001); throughput benchmark is an explicit M2 exit criterion (≥ 10⁴ steps/s); vectorise with NumPy, no per-agent Python loops in the step function; profile before optimising | Benchmark below 10⁴ steps/s at mid-W4 | R1 |
| **RSK-03** | **Corridor deadlock makes 80% unreachable** — a floor-plan property no algorithm can fix | 3 | 5 | **15** 🔴 | Measure random-policy deadlock frequency during W4 EDA *before* training; ensure task-routing aisles are ≥ 2 cells wide or add a yield/swap primitive; treat as W4 design issue #1 | Random-policy deadlock rate > 20% of episodes | R1 |
| **RSK-04** | **Tuning on the evaluation set** — with self-generated data, iterating against sealed manifests is easy to do accidentally and destroys the comparison's credibility | 3 | 5 | **15** 🔴 | Seal manifests in W4 with SHA-256 + git tag; eval runs assert the hash; **separate training-seed evaluation (free) from sealed-manifest evaluation (once per configuration)**; ETH-5 negative-result policy | Any sealed-manifest run appearing more than once per algorithm × N in MLflow | R4 |
| **RSK-05** | **QMIX does not beat IQL / H1 is contradicted** | 3 | 3 | 9 | Reframed as a research finding, not a failure (ADR-005, ETH-5). Sanity path if QMIX loses at N=5: that indicates an implementation bug — cross-check the mixer against epymarl as an oracle before reporting | QMIX ≤ IQL on training seeds at N=5 after tuning | R2 |
| **RSK-06** | **Safety overrides disrupt coordination** — masking distorts the policy's action distribution | 3 | 3 | 9 | Design the safety layer *before* training (W5) and apply the mask identically in training and evaluation (ADR-003) so the policy learns within the feasible set; add a corridor-entry penalty to reduce mask firings; log per-episode override count as a training diagnostic | Override count not declining over training | R1/R4 |
| **RSK-07** | **PyBullet install / rendering issues on Windows** | 2 | 3 | 6 | Python 3.11 toolchain chosen specifically for wheel availability (ADR-004); toolchain smoke test in W1; renderer is isolated (FR-7) so a rendering failure cannot block training or evaluation; fallback is a matplotlib 2D animation for the demo | Import failure or GUI crash in W1 smoke test | R3 |
| **RSK-08** | **Colab session pre-emption loses long runs** | 4 | 2 | 8 | Checkpoint every 50 k steps; `--resume` flag; MLflow logs streamed, not written only at the end; keep single runs under ~8 h | Any run lost to disconnect | R2 |
| **RSK-09** | **GitHub Actions free-tier exhaustion** | 2 | 3 | 6 | **Hard rule: never run training in CI.** CI runs pytest + lint + 5-episode smoke + gates only (~8 min/build). Budget ~240 min/month against the 2,000 min quota | Monthly usage > 1,000 min | R3 |
| **RSK-10** | **Reproducibility claim (NFR-6) fails at the end** — non-determinism from torch, thread scheduling, or unpinned versions surfaces in W14 when there is no time to fix it | 3 | 4 | **12** | Pin all versions; seed Python/NumPy/torch; set deterministic torch flags; record env-config hash + git commit + seed on every run (FR-8); **do a trial clean-clone reproduction in W9, not W14** | Two runs with identical seed producing different results | R3 |
| **RSK-11** | **Team member unavailable** (illness, exams, other coursework) | 3 | 3 | 9 | Clean interfaces between the four verticals; all work on branches with PRs so state is visible; every role has a documented handover section in its module README; safety layer deliberately co-owned | Any member missing two consecutive stand-ups | all |
| **RSK-12** | **Scope creep from the portfolio breadth** — 20+ named deliverables invites shallow work everywhere | 4 | 3 | **12** | Descope ladder pre-agreed (milestone plan §5); "never descoped" list protects the integrity core; Friday gate checks force explicit decisions rather than silent carry-over | Two consecutive missed milestone gates | R3 |
| **RSK-13** | **Overclaiming in the Responsible AI report** — presenting a passing CI gate as a safety case, or simulation results as ISO evidence | 2 | 4 | 8 | ETH-2 mandates an explicit non-certification statement; ETH-3 documents the guarantee's exact boundary (the policy never *commands* corridor entry ≠ a robot never *enters* one); model card scope section reviewed in W13 | Any draft sentence claiming compliance or certification | R4 |
| **RSK-14** | **Energy-benefit claim (ETH-4) is net negative** — training footprint exceeds the routing saving | 3 | 2 | 6 | Report it honestly as a two-sided ledger (NFR-8); the amortisation framing (training is one-off, routing saving is per-delivery) is itself the interesting analysis, and a negative short-run result is a legitimate finding | CodeCarbon total large relative to the estimated per-delivery saving | R4 |

---

## 2. Risk heat map

```
      I=5 |            RSK-02  RSK-03
          |            RSK-04
      I=4 |            RSK-10        RSK-13(I=4,L=2)
      I=3 | RSK-07     RSK-05  RSK-01
          | RSK-09     RSK-06  RSK-12
          |            RSK-11
      I=2 |            RSK-14  RSK-08
          +--------------------------------
             L=2        L=3     L=4
```

Three red risks (RSK-02, RSK-03, RSK-04), and all three land in **W4**. That is the correct reading
of this project: the environment week is where it is won or lost, and it deserves disproportionate
team effort and the strictest exit gate.

---

## 3. Risks retired from the brief's original register

| Original risk | Status | Reason |
|---|---|---|
| epymarl compatibility / installation complexity | **Retired** | Eliminated by decision, not mitigated — custom PyTorch QMIX is primary (ADR-002). epymarl is now an optional cross-check oracle whose absence blocks nothing. |
| PyBullet warehouse env too slow for MARL training | **Restructured** → RSK-02 | No longer a contingency; the dual-environment split is the baseline design (ADR-001). The residual risk is that even the NumPy env is too slow. |
| QMIX training unstable at N=20 | **Retained** → RSK-01 | Genuine and likely. Downgraded from red to amber only because the descope path is pre-agreed. |
| Safety overrides disrupt coordination | **Retained** → RSK-06 | Genuine; mitigation strengthened by ADR-003's train/eval-identical masking. |

---

## 4. Review protocol

- Reviewed at every Monday stand-up; scores updated; new risks added with an ID.
- Any risk reaching score ≥ 15 is raised at that stand-up and gets a named owner and a dated action.
- Realised risks are moved to a **Realised** section with what actually happened and what was done —
  this feeds directly into the W15 retrospective and is worth more than a clean register.
