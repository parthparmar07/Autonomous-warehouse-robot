# Plain-English Guide to Project 25

**For:** anyone on the team meeting this project for the first time · **Read time:** ~10 minutes

This explains the project and the five supporting papers without jargon. Every technical term is
defined the first time it appears. The formal versions live in
[`03_requirements_specification.md`](03_requirements_specification.md) and
[`07_literature_review.md`](07_literature_review.md) — read this first, those second.

---

## Part 1 — The project in one page

### The setting

Picture a large e-commerce warehouse. Shelves in rows, packing stations along one wall, and a fleet
of small wheeled robots driving around. An order comes in: *fetch item X from shelf Y, deliver it to
station Z*. The robots do the fetching. There might be 5 of them, or 20.

### The problem

**Robots get in each other's way.**

- Two robots meet in a narrow aisle heading opposite directions. Neither can pass. Both stop
  forever. This is called **deadlock**.
- Robots collide — bad for throughput, worse for the hardware.
- Five robots all charge at the same nearby task while a task on the far side sits ignored.

None of this is about one robot being bad at driving. Each individual robot is doing something
sensible. The *team* behaviour is what fails.

### Why not just write rules?

You could hand-code it: *"if a robot is ahead of you, wait two seconds"*, *"always take the nearest
task"*, *"give way to robots on your right"*. People do build systems this way, and it works until
it doesn't. Every rule you write handles the situation you imagined. The warehouse then produces a
situation you didn't imagine, and the fleet jams.

We build exactly this as our **rule-based baseline** — the thing to beat. It matters that we build
it honestly and well, because "our fancy method beat a deliberately weak baseline" is worth nothing.

### The alternative: let them learn

**Reinforcement learning (RL)** is learning by trial and error with a score attached. The robot
tries something, the world responds, and it gets points or loses them:

- Deliver an item → **+ points**
- Crash into another robot → **− points**
- Waste time going nowhere → **small − points**

Run this millions of times in simulation and the robot gradually works out which behaviours earn
points. Nobody tells it *how* to drive. It figures out driving from the score.

**Multi-agent** reinforcement learning (**MARL**) is the same idea with many robots learning at
once. Which brings the hard part.

### The hard part: who gets the credit?

The fleet delivered 50 packages this hour. **Which robot deserves the credit?**

This is the **credit assignment problem**, and it is the entire reason this project exists.

Here's the trap. Suppose robot 3 politely stops at a junction so robot 7 can pass. Robot 7 then
completes a delivery. Robot 3 did something genuinely useful — it kept traffic flowing — but robot 3
personally delivered nothing while waiting.

If you score each robot only on its own deliveries, robot 3 learns: *waiting is bad, never wait
again*. Every robot learns that. Now nobody yields, and the warehouse gridlocks.

It's the group-project problem. Everyone gets the same grade, so how do you tell who actually
contributed?

### QMIX: the method we're building

**QMIX** solves credit assignment with a neat split between *learning* and *doing*.

**While training** (in simulation, where we control everything) there's a god's-eye view. A
component called the **mixing network** sees every robot's position and the whole task queue at
once. Its job is to look at the team's total score and work out a sensible share for each robot — so
robot 3 gets credit for yielding, because the mixer can see that yielding is what let robot 7
through.

**While running** (the real thing) each robot uses only its own sensors. No god's-eye view, no
constant chatter between robots.

This split has a name: **CTDE — Centralised Training, Decentralised Execution.** Train together with
full information; act alone with partial information. It's practical, because in a real warehouse
you can't rely on perfect real-time communication, but you *can* afford a powerful simulator during
development.

> **Analogy.** A football team trains with a coach who watches the whole pitch on video and tells
> each player what their run actually achieved. On match day there's no coach — each player sees
> only their patch of grass. But they play well, because the training taught them what good team
> play feels like from their own point of view.

**One rule QMIX enforces: if any single robot gets better at its job, the team score can't go down.**
This is the **monotonicity** constraint. It sounds obvious, and it's the thing that makes the whole
scheme work — it guarantees that "every robot greedily picks its own best move" gives the same
answer as "someone with full information picks the best team move." Without it, training and
execution disagree and the method collapses.

### What we compare against

| Approach | How it works | Expected weakness |
|---|---|---|
| **Rule-based** | Take the nearest task; use A* (a classic pathfinding algorithm) to route there | No learning, no cooperation. Should choke as traffic grows. |
| **IQL** (Independent Q-Learning) | Every robot learns selfishly, treating other robots as random scenery | No credit sharing, so no reason to cooperate. Simple, and surprisingly hard to beat. |
| **QMIX** | Learns with shared credit via the mixing network | More complex; more that can go wrong |

### The actual research question

> **Does cooperation matter more as the fleet gets bigger?**

Intuition: with 5 robots the warehouse is roomy and selfish behaviour is fine. With 20 robots it's
rush hour, and yielding, spacing out, and not all chasing the same task should pay off much more.

So we train each method at **5, 10, and 20 robots** and measure whether QMIX's lead over IQL
**grows** with fleet size. That gap-versus-fleet-size curve is the point of the whole project.

**Important:** we do not know the answer, and we are not allowed to force it. We wrote the
prediction down *before* running anything and committed to reporting whatever we find. If the gap
shrinks, that's a real result and we publish it — see [ADR-005](adr/ADR-005-nfr2-as-hypothesis.md).
This is called **pre-registration**, and it's what makes the result trustworthy.

### Safety: the part that can't be learned

Some warehouse corridors have **human workers** walking in them. Robots must never enter. Ever.

We could give a huge negative score for entering and hope the robot learns to stay out. **We
deliberately don't do that**, because "the robot learned to avoid it" is a statistical tendency, not
a guarantee. A big enough reward somewhere else can outvote any penalty.

Instead we make it **structurally impossible**. Before the robot chooses a move, we delete the
illegal moves from its menu. It can't choose what isn't there.

> **Analogy.** Instead of putting a sign on a door saying "do not enter", we remove the door handle.

This is **action masking**, and it's plain `if` statements — no machine learning involved. It works
identically whether the robot's brain is well-trained, badly-trained, or completely random. Our
automated tests deliberately attack it with a robot that *only* tries to enter corridors, and the
build fails if a single intrusion gets through.

### What we're actually measuring

Three numbers, for each of the 3 methods at each of the 3 fleet sizes (9 combinations):

1. **Task completion rate** — what fraction of orders got delivered
2. **Average delivery time** — how long an order takes
3. **Collision rate** — how often robots crash per episode

Plus: zero corridor intrusions, always.

### One more thing: PyBullet

PyBullet is a 3D physics simulator. It looks great — you see actual robots rolling around in 3D,
which is perfect for the final presentation.

It's also **far too slow to learn in**. Training needs millions of practice runs. Realistic physics
gives us maybe a few hundred steps per second; a simple grid-based simulation gives tens of
thousands. Same learning, days versus minutes.

So we use **two** things:

- A **fast grid simulation** (just numbers in a NumPy array) — this does all the training and all
  the measuring. Every result comes from here.
- **PyBullet** — plays back a *recording* of what already happened, in 3D, for the demo.

PyBullet never trains anything and never produces a number we report. It's a video player, not a
simulator. See [ADR-001](adr/ADR-001-dual-environment.md).

---

## Part 2 — The five papers, in plain English

We didn't pick these for being famous. Each one answers a question we actually had.

### P1 — QMIX (ICML 2018) · *the instruction manual*

**The question it answers:** *How do you share credit fairly among cooperating agents?*

This is the paper that invented the method we're building. It introduces the mixing network, the
train-together/act-alone split (CTDE), and the monotonicity rule described above. It also shows how
to enforce monotonicity in practice — a small trick where the mixing weights are forced positive, so
one robot improving can never drag the team score down.

**Why we picked it:** we are literally implementing this paper. It's the blueprint for R2's Week 7
work, and it tells us what to test — our automated monotonicity test checks this paper's central
claim, not something we made up.

---

### P2 — Benchmarking MARL in Cooperative Tasks (NeurIPS 2021) · *the referee's scorecard*

**The question it answers:** *Which of these methods actually wins, and where?*

Someone ran the big fair comparison so nobody has to argue from intuition: many cooperative methods,
including IQL and QMIX, across many tasks, under matched conditions. It's also the paper that
released **EPyMARL** — the library your project brief mentions — and a test warehouse environment
called **RWARE**, where robots fetch and deliver items on a grid. Very close to ours.

Its sharpest warning: comparisons in this field are constantly ruined by small implementation
differences. If you don't report those details, your comparison is meaningless.

**Why we picked it:** three jobs.
1. It proves "QMIX vs. IQL" is the standard, recognised comparison — not a matchup we invented.
2. RWARE existing means a grid warehouse is an accepted research environment, backing our brief's
   claim that "the simulation is the dataset."
3. Its warning became a rule for us: we must log every setting, and keep QMIX and IQL **identical
   in every way except the mixing network**. Otherwise we'd be measuring our own inconsistency.

---

### P3 — Rethinking Implementation Tricks and Monotonicity (ICLR Blogposts 2023) · *the myth-buster*

**The question it answers:** *Is the clever idea doing the work, or is it the boring code details?*

Several papers claimed you could beat QMIX by loosening its monotonicity rule. These authors read
the actual code and found the improvements mostly came from **unrelated small optimisations** the
authors had bundled in — better gradient handling, reward scaling, tuned schedules. Apply those same
optimisations to plain QMIX, and plain QMIX wins.

They also found monotonicity **helps** rather than hurts in purely cooperative tasks — which is
exactly our kind of task.

**Why we picked it:** two reasons, and this is the paper that changed our plans most.

First, it's a warning aimed straight at us. We chose to write QMIX ourselves rather than use a
library. P3 says the gap between a good QMIX and a bad one is mostly boring details. So we now have
an explicit Week 7 checklist of those details, all recorded, so that if our QMIX underperforms we
can *diagnose* it instead of guessing.

Second — and this is worth knowing about your own project — **it proved us wrong.** We had already
written down an argument that QMIX's monotonicity rule would hurt more as the fleet grows. P3 shows
that argument is contested and possibly backwards for cooperative tasks like ours. We crossed the
claim out in the document but left it visible with the correction attached, rather than quietly
deleting it. If you're going to pre-register a prediction, you have to be equally willing to publish
the corrections that don't flatter you.

---

### P4 — Safe MARL via Shielding (AAMAS 2021) · *the safety proof*

**The question it answers:** *How do you guarantee an agent never does something dangerous — including
while it's still learning and still bad?*

They add a **shield**: a separate, non-learning component that sits between the agent and the world
and blocks unsafe actions. Two versions — one shield watching everybody (**centralised**), or many
small shields each watching a few agents (**factored**). Two findings we care about:

1. Shielding during training does **not** make the final policy worse. You don't pay for safety in
   performance.
2. Factored shields **scale better** as you add agents.

**Why we picked it:** it's independent confirmation that our safety design is right, and it upgrades
our justification from "this seems sensible" to "here's evidence."

Our corridor block is a factored shield in the simplest possible form — one tiny shield per robot,
which only needs to know where that robot is and where the corridors are. Having the proper name for
it means we can point to established research instead of appearing to have improvised.

Their first finding directly answers a worry in our risk register: *does blocking actions mess up
the robots' coordination?* We could previously only argue "probably not." Now we can cite a paper
that tested it. Their second finding is why we expect the safety check to stay fast at 20 robots.

---

### P5 — PRIMAL (IEEE RA-L 2019) · *the real-world reality check*

**The question it answers:** *Does any of this work on actual warehouse robots?*

PRIMAL tackles **multi-agent path finding** — many robots, many destinations, no collisions — which
is the real problem underneath warehouse automation. Their argument: traditional systems compute all
routes centrally, and that breaks down past a few hundred robots and can't recalculate fast enough
when something unexpected happens.

Their solution is decentralised learned policies, scaled to 1024 agents, and tested on a real
factory mockup with a mix of physical and simulated robots. They flag **reward shaping** — carefully
choosing what earns and loses points — as essential to making it work at all.

**Why we picked it:** it grounds the project in reality — this isn't a toy problem, real robotics
research works this way.

It also explains *why* we expect our rule-based baseline to fade as the fleet grows: PRIMAL's whole
motivation is centralised planning choking under scale. That's the effect our scalability study is
built to measure.

And it justifies spending real effort on reward design (our Optuna study). If a serious robotics
paper says reward shaping was make-or-break, then treating point values as something to study rather
than guess is well-founded.

One useful difference: PRIMAL trains **one** policy and copies it to any number of robots. We train
**separately** for 5, 10, and 20. Ours is right for our question — we want to isolate what changes
when the fleet grows, and a shared policy would blur that. But theirs is the better approach for
actually shipping, and that goes in our future-work section.

---

## Part 3 — How they fit together

| The question | The paper |
|---|---|
| What's the method, and why does train-together/act-alone work? | **P1** |
| Who do we compare against, and is our warehouse a legitimate test? | **P2** |
| What separates a good implementation from a broken one? | **P3** |
| How do we guarantee safety without relying on learning? | **P4** |
| Why should coordination get harder with more robots? | **P5** |

**P1 and P4 are load-bearing** — remove either and a core deliverable loses its foundation.
**P3 is the one that argued back**, and it made the project better.

### What these papers don't cover

None of the five say anything about:

- **Warehouse workers losing jobs** to automation, or getting injured working near robots
- **ISO 3691-4**, the safety standard for industrial mobile robots
- **Energy use** per delivery

That's not an oversight — these aren't machine-learning questions, and no AI paper will answer them.
They need labour economics, occupational-safety research, the actual standard document, and an
engineering reference for how much power a robot motor draws. **That's a Week 5 job**, and we've
written it down now so it doesn't ambush us in Week 13 when the report is due.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Agent** | One robot (from the AI side's point of view) |
| **Policy** | The robot's brain — the thing that turns "what I see" into "what I do" |
| **Reinforcement learning (RL)** | Learning by trial and error, guided by points |
| **MARL** | Reinforcement learning with several agents learning at the same time |
| **Credit assignment** | Working out which robot deserves credit for a team result |
| **CTDE** | Centralised Training, Decentralised Execution — learn together, act alone |
| **QMIX** | Our main method: shares credit using a mixing network that sees everything during training |
| **IQL** | Simpler comparison method: every robot learns selfishly |
| **Mixing network** | The part of QMIX that combines individual robot scores into a team score |
| **Monotonicity** | The rule that one robot improving can never lower the team score |
| **Q-value** | The robot's estimate of "how good is this move, long-term?" |
| **Action masking** | Deleting illegal moves from the menu before the robot chooses |
| **Shield** | A non-learning safety component that blocks unsafe actions |
| **Deadlock** | Two or more robots permanently blocking each other |
| **Episode** | One complete practice run, start to finish |
| **A\*** | A classic (non-learning) shortest-path algorithm |
| **Baseline** | A simpler method you compare against, to prove your method is actually worth it |
| **Pre-registration** | Writing your prediction down before running the experiment, so you can't fudge it later |
| **Reward shaping** | Deciding what earns and loses points — surprisingly, one of the most important design choices |
| **Sealed manifest** | The fixed list of test scenarios, locked before training so nobody can tune against it |
| **PyBullet** | 3D physics simulator; for us, only a video player for the demo |
| **MLflow** | Tool that records every experiment so results can be traced and reproduced |
| **ADR** | Architecture Decision Record — a short document explaining why we chose something |
