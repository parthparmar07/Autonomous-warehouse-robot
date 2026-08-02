# Assignment 1 — Literature Summary

**Project 25:** Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning

| | |
|---|---|
| **Group members** | _(name — roll no.)_ · _(name — roll no.)_ · _(name — roll no.)_ · _(name — roll no.)_ |
| **Course / Semester** | _(course code)_ · Semester 5 |
| **Faculty** | Tasmiya Khan |
| **Date** | 2 August 2026 |
| **Papers reviewed** | 6 (minimum required: 5) |

---

## 1. Introduction

Our project builds a fleet of warehouse robots that learn to cooperate. Multiple robots must
complete a queue of pick-and-deliver tasks without blocking each other, colliding, or entering
corridors reserved for human workers. We implement **QMIX**, a cooperative multi-agent
reinforcement learning (MARL) algorithm, and compare it against **IQL** (Independent Q-Learning)
and a **rule-based** dispatcher at fleet sizes of 5, 10, and 20 robots.

The six papers below were selected because each answers a specific question we had to resolve
before writing code. They are not a general survey of the field — every one of them is cited in at
least one of our design documents, and two of them changed decisions we had already made.

### A note on publication venues

In computer science and machine learning, the flagship **conferences** — ICML, NeurIPS, AAMAS — are
peer-reviewed archival venues and are generally ranked *above* most journals in the field, with
acceptance rates typically between 20% and 26%. This differs from disciplines where the journal is
the primary venue.

We have nonetheless prioritised journal publications where one exists. Our selection is:

- **3 journal articles** — *Journal of Machine Learning Research* (P1), *IEEE Robotics and
  Automation Letters* (P5), *Applied Intelligence* (P6)
- **2 top-tier peer-reviewed conference papers** — NeurIPS (P2), AAMAS (P4)
- **1 non-archival paper** — P3, an arXiv preprint later presented on the ICLR Blogposts track.
  We have flagged this openly rather than presenting it as peer-reviewed. It is included because its
  empirical ablation study directly changed two of our design decisions, and we judged that more
  valuable than a safer but less useful citation. It is our sixth paper, so the five-paper minimum
  is met by peer-reviewed work alone.

---

## 2. Summary table

| # | Paper (short) | Year | Venue | Venue type | Question it answers for us |
|---|---|---|---|---|---|
| **P1** | QMIX: Monotonic Value Function Factorisation | 2020 | *J. Machine Learning Research* 21(178) | Journal | How do you share credit fairly among cooperating agents? |
| **P2** | Benchmarking MARL Algorithms in Cooperative Tasks | 2021 | NeurIPS (Datasets & Benchmarks) | Conference | Which algorithms should we compare against, and how do we compare them fairly? |
| **P3** | Rethinking Implementation Tricks and Monotonicity | 2021 | arXiv preprint / ICLR Blogposts 2023 | Preprint | Is the algorithm doing the work, or the implementation details? |
| **P4** | Safe MARL via Shielding | 2021 | AAMAS | Conference | How do we guarantee safety when the policy is still untrained? |
| **P5** | PRIMAL: Pathfinding via RL and Imitation | 2019 | *IEEE Robotics & Automation Letters* 4(3) | Journal | Does decentralised learning work on real warehouse robots? |
| **P6** | A Review of Cooperative Multi-Agent Deep RL | 2023 | *Applied Intelligence* 53(11) | Journal | Where do our chosen methods sit in the wider field? |

---

## 3. Paper summaries

### P1 — Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (QMIX)

**Title explanation.**
A *value function* is a robot's estimate of how good a situation is. *Factorisation* means breaking
the team's single overall value into one piece per robot, so each robot can act on its own share.
*Monotonic* is the constraint that makes this safe: if any one robot's estimate goes up, the team's
estimate can never go down. In short — **splitting a team score into individual scores in a way that
keeps the two consistent.** The algorithm is called QMIX.

**Objectives.**
To train robots that must each act using only their own local observations, while exploiting the
fact that during training in simulation the whole system state is visible. The paper targets a gap
between two existing extremes: VDN, which simply adds agent values together and is too restrictive,
and fully centralised methods, which cannot be deployed on independently-acting agents. QMIX aims
for a richer combination than a sum, while still guaranteeing that each agent greedily choosing its
own best action reproduces the best joint action.

**Methodology.**
Each agent has its own recurrent deep Q-network (DRQN) reading only its local observation history.
A separate **mixing network** combines the agents' individual values into a joint value `Q_tot`.
The mixing network's weights are produced by *hypernetworks* conditioned on the global state and are
forced to be non-negative, which structurally guarantees monotonicity. The whole system is trained
end-to-end by minimising a standard temporal-difference loss on `Q_tot`. This training scheme is
called **CTDE** — Centralised Training with Decentralised Execution. Evaluation is on the StarCraft
Multi-Agent Challenge (SMAC).

**Type of results achieved.**
*In the paper:* QMIX consistently outperforms both IQL and VDN across SMAC scenarios, with ablation
studies isolating the contribution of the mixing network and the global-state conditioning.
*For our project:* this is the algorithm we implement. It provides the exact network architecture
for Week 7, and its monotonicity guarantee becomes a testable property — our automated test perturbs
one agent's value upward and asserts the team value never decreases. We expect it to yield a
task-completion rate of at least 80% at 5 robots, and a measurable advantage over IQL that we
predict will widen as the fleet grows.

**Referenced journal or website.**
Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2020).
*Journal of Machine Learning Research*, 21(178), 1–51.
Available at: https://www.jmlr.org/papers/v21/20-081.html
(Earlier conference version: *Proceedings of ICML 2018*, PMLR 80:4295–4304. Preprint:
[arXiv:1803.11485](https://arxiv.org/abs/1803.11485))

---

### P2 — Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks

**Title explanation.**
*Benchmarking* means running many algorithms on the same problems under the same conditions so the
comparison is fair. *Cooperative tasks* are problems where all agents share one goal, as opposed to
competing. The title is therefore: **a fair, like-for-like contest between MARL algorithms on
problems where agents must work together.**

**Objectives.**
The authors observe that MARL lacks commonly-used evaluation tasks and criteria, which makes
published comparisons unreliable. Their objectives are to systematically evaluate three distinct
classes of algorithm, to provide reference performance figures the community can cite, and to
release open-source tooling and environments so future comparisons are reproducible.

**Methodology.**
They compare three algorithm families — **independent learning** (including IQL), **centralised
multi-agent policy gradient**, and **value decomposition** (including QMIX) — across a diverse range
of cooperative environments. Each algorithm receives its own hyperparameter search per task, so no
method is disadvantaged by poor tuning, and results are reported across multiple random seeds. They
open-source **EPyMARL**, an extension of the PyMARL codebase that allows implementation details such
as parameter sharing to be configured explicitly, plus two new environments designed around
coordination under sparse rewards — one of which, **RWARE**, is a grid-based robotic warehouse.

**Type of results achieved.**
*In the paper:* reference performance tables across all algorithm–task combinations, and the finding
that no single algorithm dominates everywhere. Critically, they show that implementation details
such as parameter sharing materially affect results — meaning comparisons that fail to control for
them are not meaningful.
*For our project:* this validates our experimental design. It confirms QMIX-versus-IQL is the
field's standard comparison rather than an arbitrary choice, and RWARE's existence establishes a
grid warehouse as an accepted research environment. It also imposed a rule on us: QMIX and IQL must
be held identical in network size, parameter sharing, and training budget, so that the mixing
network is the only difference between them. Every such setting is logged for our results to be
defensible.

**Referenced journal or website.**
Papoudakis, G., Christianos, F., Schäfer, L., & Albrecht, S. V. (2021).
*Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021),
Datasets and Benchmarks Track.*
Available at: https://arxiv.org/abs/2006.07869 · Code: https://github.com/uoe-agents/epymarl

---

### P3 — Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL

**Title explanation.**
*Implementation tricks* are small, unglamorous coding choices — how rewards are scaled, how
gradients are clipped, which optimiser is used. The *monotonicity constraint* is QMIX's core rule
from P1. *Rethinking* signals that the paper challenges accepted wisdom. In plain terms: **are
recent improvements over QMIX genuinely better ideas, or just better code?**

**Objectives.**
Several published variants of QMIX claimed better performance by *relaxing* its monotonicity
constraint. The authors set out to test whether the constraint is genuinely the limitation those
papers assume, by separating the effect of the algorithmic change from the effect of code-level
optimisations bundled into the same implementations.

**Methodology.**
A controlled ablation study. The authors examine the released source code of QMIX variants,
identify the code-level optimisations each one contains, and then apply those same optimisations to
standard QMIX. All methods are re-run under matched conditions on SMAC and on Difficulty-Enhanced
Predator-Prey (DEPP). They add a theoretical analysis of why the monotonicity constraint should help
specifically in purely cooperative tasks, and open-source their implementation as **PyMARL2**.

**Type of results achieved.**
*In the paper:* three findings. First, the reported improvements of QMIX variants are significantly
affected by code-level optimisations rather than by relaxing monotonicity. Second, QMIX with
normalised optimisations outperforms the variants designed to beat it. Third — contrary to common
wisdom — the monotonicity constraint can *improve* sample efficiency in purely cooperative tasks.
*For our project:* two consequences, both significant. It warns that our from-scratch QMIX could
underperform for reasons unrelated to the science, so we now maintain an explicit checklist of these
implementation details, each recorded, so any underperformance is diagnosable rather than guesswork.
It also **corrected an argument we had already written down**: we had assumed the monotonicity
constraint would become more limiting as the fleet grows. Since our warehouse task is purely
cooperative — the exact regime this paper examines — that assumption is contested and possibly
backwards. We amended our design record rather than deleting the original claim.

**Referenced journal or website.**
Hu, J., Jiang, S., Harding, S. A., Wu, H., & Liao, S. (2021).
Preprint, [arXiv:2102.03479](https://arxiv.org/abs/2102.03479); later presented on the
**ICLR 2023 Blogposts Track**. Code: https://github.com/hijkzzz/pymarl2

> **Venue note (stated openly):** this is the one item in our selection that is not a peer-reviewed
> journal or conference paper. We cite it for its empirical ablations, not as theoretical authority.
> It is retained because it materially changed our design, and it is our sixth source — the
> five-paper requirement is satisfied by peer-reviewed work without it.

---

### P4 — Safe Multi-Agent Reinforcement Learning via Shielding

**Title explanation.**
A *shield* is a separate, non-learning component placed between an agent and the world; it watches
proposed actions and blocks or corrects unsafe ones. *Safe MARL* means guaranteeing that no unsafe
state is ever entered — including during training, when the policy is still bad at its job. The
title reads: **making multi-robot learning provably safe by adding a guard that does not learn.**

**Objectives.**
The authors note that MARL is increasingly applied in safety-critical settings that require
guaranteed safety during the learning process, and that current MARL methods provide no such
guarantee. Their objective is to supply one, and to determine how it should be structured so that it
remains practical as the number of agents grows.

**Methodology.**
Two shielding architectures are proposed and compared. In **centralised shielding**, a single shield
is synthesised to monitor all agents' joint actions and correct any unsafe action. In **factored
shielding**, multiple shields are synthesised based on a factorisation of the joint state space;
these shields monitor agents concurrently, each responsible for a subset of agents at each step.
Both are evaluated experimentally against unshielded MARL on safety violations, learned-policy
quality, and scalability in agent count.

**Type of results achieved.**
*In the paper:* both approaches guarantee agent safety during learning **without compromising the
quality of the learned policies** — safety is not paid for in performance. Factored shielding is
shown to be more scalable in the number of agents than centralised shielding.
*For our project:* this is the strongest external validation of our safety design. Our human-corridor
block is a factored shield in its simplest form — one shield per robot, whose correctness depends
only on that robot's position and a fixed map of corridor cells. The first finding is direct
evidence against a risk we had logged (that blocking actions might disrupt learned coordination),
which we previously could only argue from first principles. The second finding is why we expect our
safety check to remain fast at 20 robots. Our target is zero corridor intrusions across 20 dedicated
safety-test episodes, enforced automatically on every build.

**Referenced journal or website.**
ElSayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., & Feng, L. (2021).
*Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems
(AAMAS 2021)*, pp. 483–491. IFAAMAS.
Available at: https://arxiv.org/abs/2101.11196

---

### P5 — PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

**Title explanation.**
*Pathfinding* here means **multi-agent path finding (MAPF)** — routing many robots to many
destinations at once without collisions. *Reinforcement learning* is learning by trial and error;
*imitation learning* is learning by copying an expert. PRIMAL combines both: **robots learn to route
themselves by a mixture of their own experience and copying a expert planner.**

**Objectives.**
The authors observe that most state-of-the-art MAPF planners rely on centralised planning and scale
poorly past a few hundred agents, and that such approaches are poorly suited to real deployments
where noise and uncertainty require paths to be recomputed online — impossible when planning takes
seconds to minutes. Their objective is fully decentralised policies in which agents plan reactively
under partial observability while still coordinating implicitly.

**Methodology.**
Reinforcement learning is combined with imitation of an expert centralised MAPF planner, whose
demonstrations are supplied during training. The authors emphasise **careful reward shaping and
environment sampling** as essential to making the approach work. Agents observe only a local window
of the world. Once learned, a single policy can be copied onto any number of agents, so the method
naturally scales to different team sizes and world dimensions. Evaluation covers randomised worlds
with up to 1024 agents, benchmarked against state-of-the-art MAPF planners, plus a hybrid simulation
of a factory mockup involving both real and simulated robots.

**Type of results achieved.**
*In the paper:* success rates competitive with state-of-the-art planners while scaling to 1024
agents, and successful validation on physical hardware in a factory mockup.
*For our project:* three things. It grounds the project in real robotics rather than a toy problem.
Its central argument — centralised planning degrades as agent count grows — is precisely the
mechanism we predict will cause our rule-based baseline to fall behind as we scale from 5 to 20
robots, which is the effect our scalability study is designed to measure. And its emphasis on reward
shaping justifies our planned hyperparameter study treating reward weights as a research variable
rather than a guess. One deliberate difference: PRIMAL trains one policy and copies it to any number
of agents, whereas we train separately per fleet size to isolate how fleet size affects learning.
PRIMAL's approach is better for deployment and is noted in our future work.

**Referenced journal or website.**
Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., & Choset, H. (2019).
*IEEE Robotics and Automation Letters*, 4(3), 2378–2385.
DOI: [10.1109/LRA.2019.2903261](https://doi.org/10.1109/LRA.2019.2903261) ·
Preprint: [arXiv:1809.03531](https://arxiv.org/abs/1809.03531)

---

### P6 — A Review of Cooperative Multi-Agent Deep Reinforcement Learning

**Title explanation.**
A *review* (or survey) paper does not present new experiments; it organises an entire research area,
explains how the main approaches relate, and identifies open problems. *Cooperative* restricts scope
to settings where agents share a goal. The title is therefore: **a structured map of every major way
researchers make multiple agents learn to work together.**

**Objectives.**
To present and organise recent MARL algorithms for cooperative problems, to explain the challenges
each family of methods faces and how the literature has addressed them, to review real-world
applications, to catalogue the simulation environments available for MARL research, and to propose
future research directions.

**Methodology.**
A systematic literature review rather than an experimental study. The authors organise cooperative
MARL into five families: **(I) independent learners**, **(II) fully observable critic**,
**(III) value function factorisation**, **(IV) consensus**, and **(V) learn to communicate**. Each
family is explained in turn along with its characteristic difficulties and the mitigations proposed
in the relevant papers, and connections are drawn between papers within each category. Separate
sections cover emerging research areas, real-world applications, and available environments.

**Type of results achieved.**
*In the paper:* a taxonomy of the field, a comparative account of the strengths and weaknesses of
each family, a catalogue of applications and environments, and a set of proposed research
directions. Being a review, the contribution is organisational rather than empirical.
*For our project:* it locates our work precisely within the field. The two algorithms we compare are
not arbitrary picks from a large space — IQL is the canonical member of family (I) and QMIX of
family (III), so our comparison is between two of the five recognised approaches to cooperative
MARL. This gives our project's framing an authoritative, peer-reviewed, journal-published source,
and its sections on applications and environments informed our decision to build a custom grid
warehouse rather than adopt an existing benchmark.

**Referenced journal or website.**
Oroojlooy, A., & Hajinezhad, D. (2023). *Applied Intelligence*, 53(11), 13677–13722. Springer.
DOI: [10.1007/s10489-022-04105-y](https://doi.org/10.1007/s10489-022-04105-y) ·
Preprint: [arXiv:1908.03963](https://arxiv.org/abs/1908.03963)

---

## 4. How the papers fit together

| The question | Answered by |
|---|---|
| Where does our work sit in the field? | **P6** (journal review) |
| What algorithm do we build, and why does it work? | **P1** |
| What do we compare against, and how do we compare fairly? | **P2** |
| What separates a working implementation from a broken one? | **P3** |
| How do we guarantee safety without relying on learning? | **P4** |
| Why should coordination get harder with more robots? | **P5** |

P1 and P4 are load-bearing: remove either and a core objective of our project loses its foundation.
P3 is the paper that argued against a position we had already taken, and improved the project by
doing so.

## 5. Identified gaps in this literature set

Our project also has a Responsible AI component covering warehouse-worker displacement, the
**ISO 3691-4** safety standard for industrial autonomous mobile robots, and energy consumption per
delivery. **None of the six papers above address any of these.**

This is not an oversight. These are not machine-learning questions and no MARL paper will answer
them; they require labour-economics research, occupational-safety literature, the standard document
itself, and an engineering reference for robot motor power. Sourcing these is scheduled as a
separate activity in Week 5 of our project plan, and recording the gap now is how we ensure it is
not discovered late.

---

## 6. References

1. ElSayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., & Feng, L. (2021). Safe
   Multi-Agent Reinforcement Learning via Shielding. In *Proceedings of the 20th International
   Conference on Autonomous Agents and Multiagent Systems (AAMAS 2021)* (pp. 483–491). IFAAMAS.
   https://arxiv.org/abs/2101.11196

2. Hu, J., Jiang, S., Harding, S. A., Wu, H., & Liao, S. (2021). *Rethinking the Implementation
   Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning.* arXiv
   preprint arXiv:2102.03479. Later presented on the ICLR 2023 Blogposts Track.
   https://arxiv.org/abs/2102.03479

3. Oroojlooy, A., & Hajinezhad, D. (2023). A review of cooperative multi-agent deep reinforcement
   learning. *Applied Intelligence*, 53(11), 13677–13722. https://doi.org/10.1007/s10489-022-04105-y

4. Papoudakis, G., Christianos, F., Schäfer, L., & Albrecht, S. V. (2021). Benchmarking Multi-Agent
   Deep Reinforcement Learning Algorithms in Cooperative Tasks. In *Proceedings of the 35th
   Conference on Neural Information Processing Systems (NeurIPS 2021), Datasets and Benchmarks
   Track.* https://arxiv.org/abs/2006.07869

5. Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S.
   (2020). Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
   *Journal of Machine Learning Research*, 21(178), 1–51. https://www.jmlr.org/papers/v21/20-081.html

6. Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., & Choset, H. (2019).
   PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning. *IEEE Robotics and
   Automation Letters*, 4(3), 2378–2385. https://doi.org/10.1109/LRA.2019.2903261

*References listed alphabetically by first author. All six PDFs are archived in the project
repository under `papers/`.*
