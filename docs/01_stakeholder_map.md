# Stakeholder Map — Project 25

**Phase:** Initiation (W1–W3) · **Status:** Draft for review · **Owner:** R4 (Evaluation & Responsible AI Lead)

---

## 1. Purpose

Warehouse fleet automation is not a technically-isolated optimisation problem. The system's outputs
(robot actions) are executed in a physical space that is *co-occupied by human workers*, and its
success metric (throughput) is directly coupled to a labour market. Identifying who is affected —
including those who have no say in whether the system is deployed — is a precondition for the
Responsible AI analysis in W13 (ETH-1 … ETH-4) and for the safety requirements in NFR-3.

This map answers three questions per stakeholder: **what they want**, **what they risk**, and
**which project artefact addresses them**.

---

## 2. Stakeholder register

| # | Stakeholder | Type | Primary interest | Primary risk / concern | Addressed by |
|---|---|---|---|---|---|
| S1 | **Warehouse floor workers** (pickers, packers, loaders) | Affected, non-decisional | Continued employment; physical safety; humane pace of work | Displacement; wage pressure; injury from robot proximity; algorithmic work-rate intensification | Hard safety layer (FR-3, NFR-3); ETH-1 displacement analysis; ETH-2 ISO 3691-4 framing |
| S2 | **Operations managers** | Decisional, internal | Orders/hour; on-time dispatch; predictable throughput | Fleet deadlock during peak; unexplainable policy behaviour; downtime | NFR-1 completion rate; NFR-5 scalability curves; Evidently drift dashboard; model card |
| S3 | **Safety officers / EHS** | Decisional, gatekeeping | Zero personnel incidents; auditable safety case | An ML policy that "learned" safety and can therefore unlearn it | ETH-3 safety-layer independence; safety-layer blocking CI gate; override logging (FR-3) |
| S4 | **Robot vendors / integrators** | Supplier | Interoperable control interface; portable policy artefacts | Vendor lock-in to a bespoke stack; policies that assume unavailable state | ONNX export (EO-3); decentralised-execution design (§9); documented observation contract |
| S5 | **Regulators / standards bodies** (ISO 3691-4 conformity assessors) | External, gatekeeping | Demonstrable conformity to personnel-protection-zone requirements | Overclaiming: simulation results presented as certification evidence | ETH-2 explicit non-certification statement; model card scope section |
| S6 | **E-commerce platform / end customer** | Downstream beneficiary | Fast, reliable delivery | Service degradation under demand spikes | High-demand injected drift scenario (W12); documented degradation limitation |
| S7 | **Maintenance & IT operations** | Internal, operational | Deployable, monitorable, retrainable system | Undiagnosable failures; no retraining runbook | Dockerised service; PSI-based retraining trigger; runbook in README |
| S8 | **Project team (4 students)** | Delivery | Learning outcomes; deliverable completeness within 15 weeks | Scope overrun; unreproducible results near deadline | Milestone plan; sealed manifests; risk register |
| S9 | **Course faculty / examiners** | Assessing | Evidence of full ML lifecycle competence and honest evaluation | Cherry-picked results; unreproducible claims | Sealed pre-registered eval manifests (FR-4, NFR-6); MLflow lineage |
| S10 | **Environment** (non-human stakeholder, PO7) | Affected | Lower energy per delivery | Training compute footprint outweighing routing savings | ETH-4 sustainability estimate; CodeCarbon accounting of *all* runs |

---

## 3. Power / interest grid

```
        HIGH INFLUENCE
              |
   S3 Safety  |  S2 Ops managers
   officers   |  S9 Faculty
   S5 Regs    |  S8 Project team
              |
--------------+--------------------  HIGH INTEREST -->
              |
   S4 Vendors |  S1 Floor workers  <-- the critical cell
   S7 Maint.  |  S6 Customers
              |  S10 Environment
        LOW INFLUENCE
```

**The defining asymmetry of this project:** S1 (floor workers) sit in the *high-interest,
low-influence* quadrant. They bear the greatest concentrated risk — physical proximity to moving
machinery and employment displacement — while having the least say in deployment. Standard
stakeholder practice ("manage closely" only the top-right quadrant) would systematically
under-weight them.

**Project response.** We deliberately elevate S1 to "manage closely" by giving their two concerns
*hard, non-negotiable* status in the requirements rather than soft-goal status:

1. **Proximity safety** becomes a blocking CI gate (NFR-3) — a build that fails it cannot merge.
   No throughput improvement can buy its way past this constraint.
2. **Displacement** becomes a mandatory report section (ETH-1) that must present the evidence on
   employment, wages, and injury rates — including evidence unfavourable to automation — rather
   than a benefits-only business case.

This is the single most important design commitment in the stakeholder analysis and should be
stated explicitly in the final presentation.

---

## 4. Conflicts between stakeholders

Real projects have stakeholder conflicts; naming them is more honest than pretending the interests
align.

| Conflict | Parties | How this project resolves it |
|---|---|---|
| **Throughput vs. worker safety margin** — wider human corridors and conservative right-of-way rules cost delivery time | S2 vs. S1, S3 | Safety is a *constraint*, not a term in the objective. Corridor exclusion is enforced by action masking outside the learned policy; throughput is optimised only within the feasible set. The cost in delivery time is measured and reported, not hidden. |
| **Throughput vs. collision rate** — an aggressive policy completes more tasks and crashes more | S2 vs. S3, S7 | Made explicit as a Pareto frontier via the Optuna reward-shaping study (OBJ-5). We report the frontier rather than a single tuned point, so the trade-off is a management decision, not an engineering default. |
| **Automation benefit vs. displacement cost** — the productivity case for the system is partly a case against S1's jobs | S2, S6 vs. S1 | Not resolvable by engineering. ETH-1 states it plainly and presents both sides with evidence; the project does not claim to have solved it. |
| **Demonstrated safety vs. certified safety** — a passing CI gate looks like a safety case | S5 vs. S8 | ETH-2 mandates an explicit statement that simulation-based demonstration is not ISO 3691-4 certification, and enumerates what production would additionally require (hardware E-stops, proximity sensors, formal conformity assessment). |
| **Research honesty vs. attractive results** — NFR-2 predicts a result we cannot guarantee | S9 vs. S8 | NFR-2 reframed as a research hypothesis with a pre-registered test, not an acceptance criterion (see `docs/adr/ADR-005`). Either outcome is a valid deliverable. |

---

## 5. Engagement plan

Since this is a simulation-based academic project, no live stakeholders are interviewed. Each
stakeholder is instead represented by a **proxy artefact** that must be produced and reviewed —
this keeps the analysis accountable rather than rhetorical.

| Stakeholder | Proxy artefact | Review week |
|---|---|---|
| S1 Floor workers | ETH-1 displacement & proximity-safety section, sourced from occupational-safety and labour-economics literature | W13 |
| S3 Safety officers | Safety-layer test suite + 20 dedicated safety episodes + blocking CI gate | W10, re-verified every build |
| S5 Regulators | ISO 3691-4 mapping table with explicit non-certification statement | W13 |
| S2 Ops managers | Scalability report + drift dashboard + model card limitations | W11–W12 |
| S4 Vendors | ONNX artefacts + documented observation/action contract | W10–W11 |
| S10 Environment | CodeCarbon totals for every training run + energy-per-delivery estimate | W11, W13 |

---

## 6. Traceability

| Stakeholder | Requirements | Objectives |
|---|---|---|
| S1 | FR-3, NFR-3, ETH-1, ETH-3 | OBJ-4, OBJ-6 |
| S2 | FR-5, FR-6, NFR-1, NFR-5 | OBJ-1, OBJ-2, OBJ-3 |
| S3 | FR-3, NFR-3, ETH-3 | OBJ-4 |
| S4 | FR-1, FR-5 | OBJ-6 |
| S5 | ETH-2 | OBJ-6 |
| S6 | NFR-5 + drift scenario | OBJ-6 |
| S7 | FR-5, FR-6 | OBJ-6 |
| S9 | FR-4, NFR-6 | all |
| S10 | ETH-4 | OBJ-6 |
