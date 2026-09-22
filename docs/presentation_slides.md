# Project 25: Autonomous Warehouse Robot Fleet Coordination
## Presentation Slide Deck Notes & Speaker Guide

---

### Slide 1: Title & Group Members
- **Slide Title**: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning
- **Subtitle**: Decentralised Collision Avoidance, Path Finding, and Cooperative Fleet Dispatching under Centralised Training with Decentralised Execution (CTDE)
- **Project Milestone**: Project 25: Unit 1 & Unit 2 Milestone
- **Institution**: ATLAS SkillTech University | uGDX School of Technology
- **Group Members**:
  - **Parth Parmar** (Roll No: 2410244) - MARL & Pipeline Lead
  - **Saket Sarvaiya** (Roll No: 2408838) - Environment & Evaluation Lead
  - **Swarnim Jambhrunkar** (Roll No: 2409789) - Feasibility & Risk Lead
- **Page Number**: Slide 1 of 11

---

### Slide 2: Project Introduction
- **Slide Title**: Domain, Industry Context & Problem Area
- **Domain**: Smart Logistics & Autonomous Mobile Robots (AMRs) in automated e-commerce fulfillment centers.
- **Background & Context**:
  - E-commerce surges require high-density robot fleets (10 to 50 robots) operating in shared grid corridors.
  - Classical centralised pathfinding (CBS, A*) suffers exponential complexity $O(b^N)$, leading to computational bottlenecks and freezing when fleet size grows.
  - Single-agent RL fails due to non-stationarity where moving robots appear as unpredictable dynamic noise.
- **Problem Area**:
  - Inter-robot physical collisions causing hardware damage and downtime.
  - Symmetrical deadlocks and gridlock in narrow warehouse aisles.
  - Suboptimal package pick throughput under static heuristic dispatchers.
- **Page Number**: Slide 2 of 11

---

### Slide 3: Problem Statement and Project Scope
- **Problem Statement**:
  - *How can an autonomous multi-robot fleet coordinate dynamically in real-time to maximize warehouse package pick-and-delivery throughput while achieving zero collision incidents and avoiding deadlocks, using Centralised Training with Decentralised Execution (CTDE) under partial local observability?*
- **Importance of Project**:
  - Guaranteed zero inter-robot collisions via hard action-masking safety controls.
  - High order fulfillment speed (Target: >= 120 completed orders/hour for N=10 robots).
  - Ultra-fast decentralized decision latency (< 20ms per robot).
  - Zero expensive warehouse physical remodeling required.
- **Scope Boundaries**:
  - **In-Scope**: 2D discrete grid environment (20x20 `WarehouseEnv`), fleet sizes N=5, 10, 20, QMIX vs IQL vs Rule-based baseline comparison, hard action-masking safety layer, 3D PyBullet visual trajectory replay demo.
  - **Out-of-Scope**: Physical robot hardware prototyping, battery/thermal physics, continuous SLAM pipelines, dynamic human tracking.
- **Page Number**: Slide 3 of 11

---

### Slide 4: Proposed AI/ML Solution
- **Slide Title**: Multi-Agent RL with QMIX & Safety-Critical Controls
- **1. Uncertainty and Experimentation**:
  - Non-stationarity resolved via Centralised Mixing Network conditioned on global state during training.
  - Monotonicity constraint ($\partial Q_{tot}/\partial Q_i \ge 0$) guarantees Individual-Global-Max (IGM) consistency.
  - Potential-based distance reward shaping mitigates sparse terminal rewards.
- **2. Model Evaluation**:
  - 100 fixed, seed-generated evaluation scenarios across held-out warehouse layouts.
  - Statistical significance testing using Welch's t-test and Wilcoxon rank-sum (p < 0.05).
- **3. Monitoring After Deployment**:
  - Real-time inference latency tracking (< 20ms threshold).
  - Safety intervention telemetry logging whenever hard mask overrides RL policy.
  - Deadlock watchdog flagging stationary robots (> 15 steps).
- **4. Retraining Triggers**:
  - Layout drift (aisle or rack layout modifications).
  - Fleet scaling beyond training distribution (N > 20).
  - Automated shadow deployment before promoting new checkpoints.
- **Page Number**: Slide 4 of 11

---

### Slide 5: Project Life Cycle and ML Pipeline
- **Slide Title**: ML Pipeline, Automated Stages & Human Accountability
- **Life Cycle Flow**:
  1. Problem Definition (targets & KPI thresholds)
  2. Environment & Data Generation (20x20 grid layouts & pick generators)
  3. State Preparation (7x7 local FOV encoding & reward design)
  4. Model Training (PyTorch QMIX CTDE on GPU workers)
  5. Evaluation Gate (100-seed benchmark vs baseline dispatcher)
  6. Deployment (Shadow run & PyBullet trajectory replay)
  7. Monitoring & Retraining Feedback Loop
- **Automated Pipeline Stages**:
  - Vectorized parallel rollout worker episode batching.
  - Automated gradient updates with monotonic hypernetwork constraints.
  - Automated regression testing gate in CI/CD pipeline.
- **Human Judgment & Accountability Gates**:
  - **Gate 1**: Reward function design sign-off to avoid freeze behaviors.
  - **Gate 2**: Hard safety constraint verification and audit.
  - **Gate 3**: Production deployment sign-off following shadow-mode evaluation.
- **Page Number**: Slide 5 of 11

---

### Slide 6: Comparing Project Life Cycle Frameworks
- **Slide Title**: Life Cycle Comparison: CRISP-DM vs TDSP vs Google ML Lifecycle
- **Comparison Matrix**:
  - **Primary Emphasis**: CRISP-DM (Business/Tabular) | TDSP (Agile cloud sprints) | Google ML (Continuous MLOps & RL loops).
  - **Data Work**: CRISP-DM (Static dataset preprocessing) | TDSP (Cloud schemas) | Google ML (Dynamic simulation rollouts & replay buffer).
  - **Operations**: CRISP-DM (Static handoff) | TDSP (Containerized endpoints) | Google ML (Continuous telemetry & automated retraining).
  - **Automation**: CRISP-DM (Manual scripts) | TDSP (Azure DevOps) | Google ML (Automated validation gates & registry).
  - **Collaboration**: CRISP-DM (Analyst reports) | TDSP (Sprint boards) | Google ML (ADRs, artifact hashing & reproducibility logs).
- **Framework Selection**:
  - Google ML Lifecycle selected as best fit because multi-agent RL requires dynamic simulation data streams, rigorous multi-seed safety evaluation gates, and continuous telemetry monitoring.
- **Page Number**: Slide 6 of 11

---

### Slide 7: Feasibility Analysis
- **Slide Title**: Technical, Economic, Operational & Ethical Feasibility
- **Technical Feasibility**: Custom PyTorch QMIX implementation (~300 lines) running on local RTX 4070 / Apple Silicon GPU; risk of N=20 scalability mitigated via parameter sharing and pre-registered hypothesis H1.
- **Economic Feasibility**: Zero external API costs; built on open-source Python, PyTorch, NumPy, PyBullet.
- **Operational Feasibility**: Sub-20ms decentralized inference easily runs on edge warehouse controllers.
- **Ethical Feasibility Matrix**:
  - *Safety in Shared Zones*: Mitigated via deterministic hard action-masking safety layer.
  - *Flawed Reproducibility*: Mitigated via SHA-256 sealed test manifests and 100-seed evaluation.
  - *Worker Pacing Stress*: Mitigated via dynamic human-paced buffer controls.
- **Page Number**: Slide 7 of 11

---

### Slide 8: Requirements Engineering for the Project
- **Slide Title**: Requirements Engineering: Functional, Non-Functional, Data & Ethics
- **Functional Requirements (FR)**:
  - **FR-1**: Decentralized per-robot inference (< 50ms) on 7x7 FOV observation.
  - **FR-2**: Cooperative collision-free pathfinding to dynamically assigned pick stations.
  - **FR-3**: Dynamic automated task allocation upon order fulfillment.
- **Non-Functional Requirements (NFR)**:
  - **NFR-1**: Zero physical inter-robot collisions across 100 evaluation seeds.
  - **NFR-2**: Fleet throughput efficiency > 80% when scaling N=5 to N=10 and N=20.
  - **NFR-3**: Decision latency <= 20ms per agent on single CPU core.
- **Data Requirements (DR)**:
  - **DR-1**: 7x7x3 normalized tensor representing obstacles, agents, and target vectors.
  - **DR-2**: FIFO replay buffer storing 5,000 multi-agent transition episodes.
  - **DR-3**: 100 fixed, seed-generated evaluation manifests locked with SHA-256 hashes.
- **Ethical & Governance Requirements (ETH)**:
  - **ETH-1**: Hard safety layer cannot be bypassed by learned policies.
  - **ETH-2**: Structured telemetry logging for all safety interventions and deadlocks.
  - **ETH-3**: Full code, seed, and environment reproducibility in open-source repository.
- **Page Number**: Slide 8 of 11

---

### Slide 9: Stakeholder Management
- **Slide Title**: Stakeholder Management Matrix & Concern Mapping
- **Academic Evaluators & Faculty**: High Power | Concern: Methodological rigor & reproducibility | Response: 100-seed benchmark, statistical tests & SHA-256 manifests.
- **Warehouse Operations Managers**: High Power | Concern: Costly deadlocks and floor downtime | Response: Deadlock watchdog heuristics & fallback routing.
- **Warehouse Floor Workers**: High Interest | Concern: Unpredictable robot collisions in shared aisles | Response: Hard action-masking safety layer & human-paced dispatch buffer.
- **ML Engineering Team**: High Power | Concern: Scope creep and hyperparameter tuning complexity | Response: Explicit ADRs, modular PyTorch codebase & fast 2D simulator.
- **Page Number**: Slide 9 of 11

---

### Slide 10: Future Work & Next Steps
- **Slide Title**: Future Work, Risk Prioritization & Immediate Next Steps
- **1. Core Gap Identified**:
  - Scalability degradation at N=20 due to centralized mixing capacity bottlenecks.
  - Sim-to-Real gap between 2D discrete grid and continuous robot dynamics (wheel slip, momentum).
- **2. Most Important Risk to Address First**:
  - Non-stationary policy oscillations causing fleet circular blocking at narrow intersections.
  - Immediate mitigation: Potential-based reward shaping and corridor directional priority rules.
- **3. Immediate Next Steps**:
  - Step 1: Complete 50k-step QMIX training runs across fleet sizes N=5, 10, 20.
  - Step 2: Conduct 100-seed statistical benchmark against IQL and classical rule dispatcher.
  - Step 3: Render PyBullet 3D trajectory replay demo for final visual presentation.
- **Page Number**: Slide 10 of 11

---

### Slide 11: Thank You Slide
- **Slide Title**: Thank You & Open Q&A
- **Summary**: Multi-Agent Reinforcement Learning for Autonomous Warehouse Fleet Coordination.
- **Q&A Focus**: CTDE architecture, safety-masking layers, evaluation benchmarks, and MLOps pipeline design.
- **Group Members**: Parth Parmar (2410244), Saket Sarvaiya (2408838), Swarnim Jambhrunkar (2409789).
- **Institution**: ATLAS SkillTech University • uGDX School of Technology
- **Repository**: `parthparmar07/Autonomous-warehouse-robot`
- **Page Number**: Slide 11 of 11
