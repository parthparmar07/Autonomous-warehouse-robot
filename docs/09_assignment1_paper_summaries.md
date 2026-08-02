# Assignment 1: Literature Summary

**Project 25: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning**

> **This is the repository copy.** It exists so the text is version-controlled and readable on GitHub.
> **Nothing here needs to be submitted.** For submission, open
> [`09_assignment1_paper_summaries.html`](09_assignment1_paper_summaries.html) in a browser and print
> it to PDF. Both files carry identical text; if you edit one, edit the other.

| Application ID | Name |
|:---:|---|
| 2408838 | Saket Sarvaiya |
| 2409789 | Swarnim Jambhrunkar |
| 2410244 | Parth Parmar |

---

## 1. Introduction

This project develops a fleet of warehouse robots that learn to cooperate. Several robots must
complete a queue of pick-and-deliver tasks without blocking one another, colliding, or entering
corridors reserved for human workers. Coordination is the central difficulty. A robot travelling to a
shelf may obstruct another robot's route, and without joint planning the fleet can deadlock in
congested aisles, which destroys throughput. We implement QMIX, a cooperative multi-agent
reinforcement learning (MARL) algorithm, and evaluate it against Independent Q-Learning (IQL) and a
rule-based dispatcher at fleet sizes of 5, 10 and 20 robots. The principal research question is
whether the benefit of cooperative learning grows as the fleet becomes larger.

The six papers summarised below were selected because each one answers a specific question that had
to be resolved before implementation could begin. They are not intended as a general survey of the
field. Every paper listed here is cited in at least one of the project's design documents, and two of
them caused us to revise decisions that had already been taken.

### 1.1 Note on publication venues

In computer science and machine learning, the leading conferences (ICML, NeurIPS and AAMAS) are
peer-reviewed archival venues and are generally regarded as ranking above most journals in the field,
with acceptance rates typically between 20 and 26 per cent. This differs from disciplines in which the
journal is the primary venue of record.

Where a journal version of a paper exists, we have cited it in preference to the conference version.
The final selection comprises three journal articles (*Journal of Machine Learning Research*, *IEEE
Robotics and Automation Letters*, and *Applied Intelligence*), two peer-reviewed conference papers
(NeurIPS and AAMAS), and one preprint (Paper 3). Paper 3 is identified as a preprint wherever it
appears rather than being presented as peer-reviewed work. It has been retained because its empirical
findings directly changed two of our design decisions. As it is the sixth paper, the five-paper
minimum is satisfied by peer-reviewed work alone.

---

## 2. Summary of Selected Papers

**Table 1: Overview of the six papers reviewed.**

| No. | Paper | Year | Venue | Venue Type |
|:---:|---|:---:|---|---|
| 1 | Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (QMIX) | 2020 | *Journal of Machine Learning Research*, 21(178) | Journal |
| 2 | Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks | 2021 | NeurIPS, Datasets and Benchmarks Track | Conference |
| 3 | Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL | 2021 | arXiv preprint; ICLR Blogposts Track 2023 | Preprint |
| 4 | Safe Multi-Agent Reinforcement Learning via Shielding | 2021 | AAMAS | Conference |
| 5 | PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning | 2019 | *IEEE Robotics and Automation Letters*, 4(3) | Journal |
| 6 | A Review of Cooperative Multi-Agent Deep Reinforcement Learning | 2023 | *Applied Intelligence*, 53(11) | Journal |

---

## 3. Paper Summaries

### 3.1 Paper 1: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (QMIX)

**Title explanation.** A value function is an estimate of how good a situation is, expressed as the
total reward an agent expects to collect from that point onward. Factorisation refers to dividing the
team's single overall value into one component per robot, so that each robot can act on its own share
without consulting the others. Monotonic is the constraint that makes this division safe: if any one
robot's estimate increases, the team's estimate can never decrease. The word deep indicates that the
value functions are represented by neural networks rather than lookup tables, which is what allows the
method to handle the large state spaces found in realistic problems. Taken together, the title
describes a method for splitting a team score into individual scores in a way that keeps the two
consistent. The algorithm is named QMIX.

**Objectives.** To train agents that must each act using only their own local observations, while
taking advantage of the fact that the complete system state is visible during training in simulation.
This setting is formalised as a decentralised partially observable Markov decision process, in which
no agent can see the whole world. The paper addresses a gap between two existing extremes. Value
Decomposition Networks (VDN) simply sum the individual agent values, which is easy to decentralise but
too restrictive to represent the value of situations where agents genuinely depend on one another.
Fully centralised methods can represent any such situation but produce a single joint policy that
cannot be deployed on agents acting independently. QMIX aims to permit a much richer combination than
a sum while still guaranteeing that each agent selecting its own best action reproduces the best joint
action, a property the authors call consistency between the centralised and decentralised policies.

**Methodology.** Each agent is given its own recurrent deep Q-network, built on a gated recurrent
unit, which reads only that agent's local observation history together with its previous action and an
identifier. Recurrence is used so that an agent can compensate for partial observability by
remembering what it has previously seen. A separate mixing network then combines the individual agent
values into a joint value. The weights of this mixing network are not learned directly but are
produced by hypernetworks, which are small networks that take the global state as input and output the
weights of the mixing network. Those weights are passed through an absolute value operation so that
they cannot be negative, which is what structurally guarantees monotonicity. The complete system is
trained end to end by minimising a temporal-difference loss on the joint value, using experience
replay, a target network and epsilon-greedy exploration. This training arrangement is known as
Centralised Training with Decentralised Execution (CTDE). To demonstrate the representational
advantage over VDN, the authors construct a deliberately simple two-step cooperative matrix game for
two agents, in which the first agent's initial choice determines which payoff matrix the pair will
face. The main empirical evaluation is carried out on the StarCraft Multi-Agent Challenge (SMAC), a
benchmark built on StarCraft II in which each unit is controlled by an independent learning agent
acting on local observations, while the opposing units are controlled by the built-in game AI. The
journal version extends the original conference paper with standardised performance metrics, sample
efficiency measurements and an analysis of computational requirements.

**Type of results achieved.** In the paper, QMIX outperforms IQL, VDN and COMA on SMAC, both in
absolute performance and in learning speed, with the largest gains appearing on the harder scenarios
and on those involving heterogeneous agents. On randomly generated matrix games, QMIX learns
significantly more accurate estimates of the maximum joint value, which means it provides better
targets for its own learning updates. The ablation studies establish two things that matter for anyone
reimplementing the method: conditioning the mixing network on the global state is necessary, and so is
using a flexible multi-layer mixing network rather than a linear one. For our project, this is the
algorithm we implement, and the paper supplies the network architecture we build. The ablation
findings are directly actionable, because they tell us that a simplified mixer would underperform for
reasons unrelated to our warehouse problem, so the hypernetwork must be implemented properly rather
than approximated. The monotonicity guarantee also becomes a testable property: our automated test
increases one agent's value and asserts that the team value never decreases, which checks this paper's
central structural claim rather than an invariant of our own devising. We expect to obtain a
task-completion rate of at least 80 per cent with 5 robots, together with a measurable advantage over
IQL that we predict will widen as the fleet grows.

**Referenced journal or website.** Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G.,
Foerster, J., and Whiteson, S. (2020). *Journal of Machine Learning Research*, 21(178), 1-51.
https://www.jmlr.org/papers/v21/20-081.html
Earlier conference version: *Proceedings of ICML 2018*, PMLR 80:4295-4304.

---

### 3.2 Paper 2: Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks

**Title explanation.** Benchmarking means running a number of algorithms on the same problems under
identical conditions so that the comparison between them is fair. Cooperative tasks are problems in
which all agents share a single goal and receive a common reward, as distinct from competing against
one another. The title therefore describes a like-for-like contest between MARL algorithms on problems
that require agents to work together.

**Objectives.** The authors observe that MARL lacks commonly used evaluation tasks, protocols and
criteria, which makes published comparisons between algorithms unreliable and difficult to interpret.
Their objectives are to evaluate three distinct classes of algorithm systematically, to provide
reference performance figures that the research community can cite instead of rerunning baselines, and
to release open-source tools and environments so that future comparisons are reproducible. A further
objective is to isolate the effect of implementation choices, in particular whether agents share
network parameters, which is a decision frequently left unreported in the literature.

**Methodology.** Three algorithm families are compared: independent learning, which includes IQL;
centralised multi-agent policy gradient methods; and value decomposition, which includes QMIX. These
are evaluated on two matrix games and four multi-agent environments, within which the authors define a
total of 25 distinct cooperative learning tasks. The hyperparameters of each algorithm are optimised
separately for each environment using a grid search, so that no method is disadvantaged by poor
tuning, and both the maximum and the average evaluation returns during training are reported. Every
configuration is run both with shared and with non-shared parameters between agents, so the effect of
that choice can be measured rather than assumed. The authors release EPyMARL, an extension of the
widely used PyMARL codebase which adds algorithms and allows implementation details to be configured
explicitly, together with two new environments designed to test coordination under sparse rewards and
partial observability: Level-Based Foraging and Multi-Robot Warehouse (RWARE), the latter being a
grid-based robotic warehouse.

**Type of results achieved.** The paper produces reference performance tables covering all algorithm
and task combinations, and establishes that no single algorithm dominates across every setting, so the
choice of method must be justified against the specific task rather than assumed from published
rankings. It demonstrates that implementation details such as parameter sharing materially affect
measured performance, which means that any comparison failing to control for them is not meaningful.
The environments requiring coordination under sparse rewards prove the most difficult for all
algorithm families. For our project, this validates the experimental design in three ways. It confirms
that comparing QMIX against IQL is the standard comparison in the field rather than an arbitrary
choice, since the two are the leading representatives of two of the three families examined. The
existence of RWARE, a multi-robot warehouse, establishes a grid warehouse as an accepted research
environment and supports our decision to treat a simulated environment as the dataset. Finally, the
paper imposes a methodological requirement on us: QMIX and IQL must be held identical in network size,
parameter sharing and training budget, so that the mixing network is the only difference between them,
and each algorithm must be tuned separately rather than sharing one configuration. Every such setting
is recorded so that our results can be defended.

**Referenced journal or website.** Papoudakis, G., Christianos, F., Schafer, L., and Albrecht, S. V.
(2021). *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021),
Datasets and Benchmarks Track*. https://arxiv.org/abs/2006.07869

---

### 3.3 Paper 3: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL

**Title explanation.** Implementation tricks are small and unremarkable coding choices, such as how
rewards are scaled, how gradients are clipped, which optimiser is selected and how the exploration
rate is reduced over time. None of these appear in the description of an algorithm, yet all of them
affect how well it performs. The monotonicity constraint is the central rule of QMIX described in
Paper 1. The word rethinking signals that the paper challenges an accepted position. In plain terms,
the paper asks whether recent improvements over QMIX represent genuinely better ideas or merely better
code.

**Objectives.** Several published variants of QMIX claimed improved performance by relaxing its
monotonicity constraint, and this had begun to be treated as settled knowledge in the field. The
authors set out to test whether the constraint is in fact the limitation that those papers assume, by
separating the effect of the algorithmic change from the effect of the code-level optimisations bundled
into the same implementations. A secondary objective is to explain, rather than merely observe, why
the constraint behaves as it does in problems where all agents share a single goal.

**Methodology.** The study takes the form of a controlled ablation. The authors examine the released
source code of the QMIX variants, identify the code-level optimisations that each one contains, and
then apply those same optimisations to standard QMIX, so that the only remaining difference between
the methods is the algorithmic change itself. All methods are re-run under matched conditions on two
benchmarks: the StarCraft Multi-Agent Challenge and Difficulty-Enhanced Predator-Prey. Using two
benchmarks rather than one guards against a conclusion that holds only for a single family of tasks.
The empirical work is accompanied by a theoretical analysis of why the monotonicity constraint should
be expected to help specifically in purely cooperative tasks. The authors release their implementation
as PyMARL2 so that the comparison can be reproduced.

**Type of results achieved.** The paper reports three findings. First, the improvements reported for
QMIX variants are significantly affected by code-level optimisations rather than by the relaxation of
monotonicity, which means several published conclusions do not survive a controlled comparison.
Second, QMIX with normalised optimisations outperforms the variants that were designed to surpass it.
Third, and contrary to the prevailing view, the monotonicity constraint can improve sample efficiency
in purely cooperative tasks, so the constraint is in some settings an advantage rather than a
limitation. For our project there are two consequences. The paper warns that a from-scratch QMIX
implementation may underperform for reasons unrelated to the underlying science, so we maintain an
explicit checklist of these implementation details, covering reward normalisation, gradient clipping,
choice of optimiser, exploration schedule, replay buffer size, target network update frequency and
parameter sharing. Each item is recorded for every training run, which makes any underperformance
diagnosable rather than a matter of guesswork, and ensures QMIX and IQL are configured identically.
The paper also corrected an argument we had already committed to writing: we had assumed that the
monotonicity constraint would become increasingly limiting as the fleet grows. Since our warehouse
task is purely cooperative, which is precisely the regime this paper examines, that assumption is
contested and may be the reverse of the truth. We amended our design record and left the original
claim visible alongside the correction, on the grounds that a stated prediction is only credible if
revisions that work against it are also published.

**Referenced journal or website.** Hu, J., Jiang, S., Harding, S. A., Wu, H., and Liao, S. (2021).
arXiv preprint arXiv:2102.03479. Subsequently presented on the ICLR 2023 Blogposts Track.
https://arxiv.org/abs/2102.03479

This is the only source in the selection that is not a peer-reviewed journal or conference paper. It
is cited for its empirical ablation study rather than as theoretical authority, and it is retained
because it materially changed our design decisions.

---

### 3.4 Paper 4: Safe Multi-Agent Reinforcement Learning via Shielding

**Title explanation.** A shield is a separate, non-learning component positioned between an agent and
its environment, which monitors proposed actions and blocks or corrects any that would be unsafe. Safe
MARL means guaranteeing that no unsafe state is ever entered, including during training, when the
policy is still performing poorly and is exploring more or less at random. This is a stronger
requirement than the usual one of producing a safe policy at the end of training. The title therefore
describes a method for making multi-robot learning provably safe by adding a guard that does not
itself learn.

**Objectives.** The authors note that MARL is increasingly applied in safety-critical settings that
require guaranteed safety throughout the learning process, and that current MARL methods provide no
such guarantee. Their objective is to supply one, and to determine how it should be structured so that
it remains practical as the number of agents increases. A further objective is to establish that the
guarantee can be obtained without degrading the policy that is eventually learned, since a safety
mechanism that ruins performance would not be adopted.

**Methodology.** Two shielding architectures are proposed and compared. Under centralised shielding, a
single shield is synthesised to monitor the joint actions of all agents and to correct any unsafe
action. Under factored shielding, multiple shields are synthesised on the basis of a factorisation of
the joint state space; these shields monitor agents concurrently, with each shield responsible for a
subset of agents at each step. Because each individual shield can only monitor a limited number of
agents, as many shields as required can be employed, and together they cover a large population. Both
approaches are evaluated on six benchmark problems drawn from a grid world and a cooperative
navigation environment, measuring safety violations, the quality of the learned policy, and
scalability in the number of agents. Importantly, the experiments are run with two different
underlying MARL algorithms, CQ-learning and MADDPG, in order to demonstrate that the shielding
approaches are not tied to any one learning method.

**Type of results achieved.** The paper demonstrates that both approaches guarantee the safety of
agents during learning without compromising the quality of the learned policies, so that safety is not
paid for in performance. It further shows that factored shielding is more scalable in the number of
agents than centralised shielding, and that both are compatible with structurally different MARL
algorithms. For our project, this provides the strongest external validation of our safety design. Our
human-corridor block is a factored shield in its simplest form, consisting of one shield per robot
whose correctness depends only on that robot's position and a fixed map of corridor cells. The first
finding is direct evidence against a risk we had recorded, namely that blocking actions might disrupt
learned coordination, which we had previously been able to argue against only from first principles.
The second finding is the reason we expect our safety check to remain fast at 20 robots. The third
finding matters for how we test: because shielding is algorithm-independent, the same safety layer
must hold for QMIX, for IQL and for the rule-based baseline alike, which is exactly the assumption our
automated safety check relies on. Our target is zero corridor intrusions across 20 dedicated
safety-test episodes, verified automatically on every build. Adopting the terminology of this paper
also places our design within an established body of research rather than presenting it as something
invented for this project.

**Referenced journal or website.** ElSayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U.,
and Feng, L. (2021). *Proceedings of the 20th International Conference on Autonomous Agents and
Multiagent Systems (AAMAS 2021)*, pp. 483-491. IFAAMAS. https://arxiv.org/abs/2101.11196

---

### 3.5 Paper 5: PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning

**Title explanation.** Pathfinding here refers to multi-agent path finding (MAPF), the problem of
routing many robots to many destinations simultaneously without collisions. Reinforcement learning is
learning through trial and error guided by a reward signal, while imitation learning is learning by
copying the decisions of an expert. PRIMAL combines the two, so that robots learn to route themselves
through a mixture of their own experience and imitation of an expert planner. The combination is used
because trial and error alone rarely discovers good coordinated behaviour in a problem this large,
while imitation alone cannot adapt beyond the situations the expert was shown.

**Objectives.** The authors observe that most state-of-the-art MAPF planners rely on centralised
planning and scale poorly beyond a few hundred agents, and that such approaches are poorly suited to
real deployments, where noise and uncertainty require paths to be recomputed online. This is not
possible when planning times are measured in seconds or minutes. Their objective is to produce fully
decentralised policies in which agents plan reactively under partial observability while still
coordinating implicitly, meaning that coordination emerges from learned behaviour rather than from
explicit communication between robots.

**Methodology.** The work extends the authors' earlier research on distributed learning of
collaborative policies, which was based on the asynchronous advantage actor-critic algorithm and in
which agents treated one another as moving obstacles. Three additions are made. Demonstrations from an
expert centralised MAPF planner are supplied during training, so that agents are shown examples of
good coordinated routing rather than having to discover it unaided. Reward shaping is designed
carefully, which the authors identify as essential to making the approach work. Environment sampling
is also controlled, so that agents encounter a suitable spread of world layouts and congestion levels
during training. Agents observe only a local window of the world and require no explicit communication
with one another. Once learned, a single policy can be copied onto any number of agents, so the method
scales naturally to different team sizes and world dimensions. Evaluation covers randomised worlds
containing up to 1,024 agents, benchmarked against state-of-the-art MAPF planners, together with a
hybrid simulation of a factory mockup involving both real and simulated robots.

**Type of results achieved.** The paper reports success rates competitive with state-of-the-art
planners while scaling to 1,024 agents, far beyond the few hundred at which centralised planners
degrade, and reports successful validation on physical hardware in a factory mockup. For our project
the paper contributes in three ways. It grounds the work in real robotics rather than a purely
theoretical problem, and demonstrates that decentralised learned policies can drive warehouse robots
in practice. Its central argument, that centralised planning degrades as agent count grows, is
precisely the mechanism we predict will cause our rule-based baseline to fall behind as we scale from
5 to 20 robots, and measuring that effect is the purpose of our scalability study. Its emphasis on
reward shaping justifies our planned hyperparameter study, which treats reward weights as a research
variable to be investigated rather than a value to be guessed. The paper also offers a contingency: if
our own method fails to reach the target completion rate through trial and error alone, initialising
it from our rule-based planner is an approach with published support rather than an improvisation. One
deliberate difference should be noted. PRIMAL trains a single policy and copies it to any number of
agents, whereas we train separately for each fleet size in order to isolate how fleet size affects
learning. The PRIMAL approach is better suited to deployment and is recorded in our future work.

**Referenced journal or website.** Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S.,
Koenig, S., and Choset, H. (2019). *IEEE Robotics and Automation Letters*, 4(3), 2378-2385.
https://doi.org/10.1109/LRA.2019.2903261

---

### 3.6 Paper 6: A Review of Cooperative Multi-Agent Deep Reinforcement Learning

**Title explanation.** A review, or survey, paper does not present new experiments. Instead it
organises an entire research area, explains how the principal approaches relate to one another, and
identifies open problems. Its value lies in showing how individual contributions fit together, which
is difficult to see from reading papers in isolation. The word cooperative restricts the scope to
settings in which agents share a goal rather than compete. The title therefore describes a structured
map of the major ways in which researchers make multiple agents learn to work together.

**Objectives.** To present and organise recent MARL algorithms for cooperative problems, to explain
the challenges each family of methods faces and how the literature has addressed them, to review
real-world applications in which MARL has been deployed successfully, to catalogue the simulation
environments available for MARL research, and to propose directions for future research.

**Methodology.** The paper is a systematic literature review rather than an experimental study. The
authors organise cooperative MARL into five families: independent learners, fully observable critic,
value function factorisation, consensus, and learning to communicate. Each family is explained in
turn, together with its characteristic difficulties and the mitigations proposed in the relevant
papers, and connections are drawn between papers within each category so that the development of ideas
over time can be followed. Separate sections address emerging research areas, real-world applications,
and the environments available for research. The review is extensive, running to more than eighty
pages in its preprint form.

**Type of results achieved.** The paper produces a taxonomy of the field, a comparative account of the
strengths and weaknesses of each family of methods, a catalogue of applications and environments, and
a set of proposed research directions. As a review, its contribution is organisational rather than
empirical, and its value is in orientation rather than in any single measurement. For our project it
locates the work precisely within the field. The two algorithms we compare are not arbitrary
selections from a large space: IQL is the canonical member of the independent learners family and QMIX
of the value function factorisation family, so our comparison is between two of the five recognised
approaches to cooperative MARL. This gives the framing of our project an authoritative, peer-reviewed
and journal-published source, which matters because the comparison is the basis of our main research
question. The review also identifies the three families we do not examine, which allows us to state
the boundaries of our study accurately and to justify them rather than leave the omissions
unexplained. Its sections on applications and environments informed our decision to build a custom
grid warehouse rather than adopt an existing benchmark, since no available environment matched the
combination of a task queue, human-worker corridors and variable fleet size that our study requires.

**Referenced journal or website.** Oroojlooy, A., and Hajinezhad, D. (2023). *Applied Intelligence*,
53(11), 13677-13722. Springer. https://doi.org/10.1007/s10489-022-04105-y

---

## 4. Relationship Between the Selected Papers

**Table 2: The question each paper resolves for the project.**

| Question | Paper |
|---|:---:|
| Where does our work sit within the wider field? | 6 |
| Which algorithm do we build, and why does it work? | 1 |
| What do we compare against, and how do we compare fairly? | 2 |
| What separates a working implementation from a broken one? | 3 |
| How do we guarantee safety without relying on learning? | 4 |
| Why should coordination become harder as robot numbers grow? | 5 |

The six papers form a sequence rather than a list. Paper 6 establishes the landscape and shows that
the two algorithms we compare represent two recognised families. Paper 1 supplies the algorithm itself
and the guarantee that makes decentralised execution valid. Paper 2 defines how a fair comparison
between such algorithms must be conducted, and supplies the closest published analogue to our
warehouse. Paper 3 warns that implementation quality, not algorithm choice alone, determines measured
performance. Paper 4 supplies the safety architecture, which is deliberately kept outside the learned
model. Paper 5 connects the whole to physical warehouse robotics and explains why the problem should
become harder as the fleet grows.

Papers 1 and 4 are the load-bearing sources: if either were removed, a core objective of the project
would lose its foundation. Paper 3 is notable as the source that argued against a position we had
already adopted, and the project is stronger for it.

---

## 5. Identified Gaps in the Literature Reviewed

The project also has a Responsible AI component covering the displacement of warehouse workers, the
ISO 3691-4 safety standard for industrial autonomous mobile robots, and energy consumption per
delivery. None of the six papers above addresses any of these subjects.

This is not an oversight. These are not machine learning questions, and no MARL paper will answer
them. They require labour economics research on how automation affects employment and wages,
occupational safety literature on injury rates among workers operating alongside machinery, the text
of the standard itself for the personnel protection requirements our safety layer is intended to
reflect, and an engineering reference for robot motor power in order to convert distance travelled
into energy consumed. Sourcing this material is scheduled as a separate activity in Week 5 of the
project plan. Recording the gap at this stage is how we ensure it is not discovered late, when there
would be insufficient time to address it properly.

---

## 6. References

1. ElSayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., and Feng, L. (2021). Safe
   Multi-Agent Reinforcement Learning via Shielding. In *Proceedings of the 20th International
   Conference on Autonomous Agents and Multiagent Systems (AAMAS 2021)*, pp. 483-491. IFAAMAS.
   Available at: https://arxiv.org/abs/2101.11196

2. Hu, J., Jiang, S., Harding, S. A., Wu, H., and Liao, S. (2021). *Rethinking the Implementation
   Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning*. arXiv
   preprint arXiv:2102.03479. Available at: https://arxiv.org/abs/2102.03479

3. Oroojlooy, A., and Hajinezhad, D. (2023). A review of cooperative multi-agent deep reinforcement
   learning. *Applied Intelligence*, 53(11), 13677-13722. Available at:
   https://doi.org/10.1007/s10489-022-04105-y

4. Papoudakis, G., Christianos, F., Schafer, L., and Albrecht, S. V. (2021). Benchmarking Multi-Agent
   Deep Reinforcement Learning Algorithms in Cooperative Tasks. In *Proceedings of the 35th Conference
   on Neural Information Processing Systems (NeurIPS 2021), Datasets and Benchmarks Track*. Available
   at: https://arxiv.org/abs/2006.07869

5. Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., and Whiteson, S.
   (2020). Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
   *Journal of Machine Learning Research*, 21(178), 1-51. Available at:
   https://www.jmlr.org/papers/v21/20-081.html

6. Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., and Choset, H. (2019).
   PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning. *IEEE Robotics and
   Automation Letters*, 4(3), 2378-2385. Available at: https://doi.org/10.1109/LRA.2019.2903261

References are listed alphabetically by first author. Copies of all six papers are archived in the
project repository under `papers/`.
