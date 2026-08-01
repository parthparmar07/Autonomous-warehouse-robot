# Supporting Papers

Five papers underpinning Project 25. Full analysis, including what each one changes about our design
decisions, is in [`docs/07_literature_review.md`](../docs/07_literature_review.md).

| # | Paper | Venue | arXiv | Supports |
|---|---|---|---|---|
| P1 | [QMIX: Monotonic Value Function Factorisation for Deep MARL](01-qmix-rashid-2018.pdf) | ICML 2018 | [1803.11485](https://arxiv.org/abs/1803.11485) | OBJ-1, FR-1, FR-2, ADR-002 |
| P2 | [Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks](02-benchmarking-marl-papoudakis-2021.pdf) | NeurIPS 2021 D&B | [2006.07869](https://arxiv.org/abs/2006.07869) | OBJ-2, FR-8, RWARE domain |
| P3 | [Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL](03-monotonicity-tricks-hu-2021.pdf) | ICLR Blogposts 2023 | [2102.03479](https://arxiv.org/abs/2102.03479) | ADR-002, ADR-005, RSK-05 |
| P4 | [Safe Multi-Agent Reinforcement Learning via Shielding](04-safe-marl-shielding-elsayedaly-2021.pdf) | AAMAS 2021 | [2101.11196](https://arxiv.org/abs/2101.11196) | OBJ-4, FR-3, NFR-3, ETH-3, ADR-003 |
| P5 | [PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning](05-primal-mapf-sartoretti-2019.pdf) | IEEE RA-L 2019 | [1809.03531](https://arxiv.org/abs/1809.03531) | OBJ-3, OBJ-5, warehouse MAPF |

All five are open-access preprints downloaded from arXiv, retained here for study under normal
academic use. P5 is also published in IEEE RA-L
([10.1109/LRA.2019.2903261](https://doi.org/10.1109/LRA.2019.2903261)); the arXiv preprint is the
copy stored here.

## Known gaps

These five cover the technical core. They do **not** cover three Responsible AI requirements, which
need non-ML sources and are a **W5 (milestone M3)** task:

- **ETH-1** — warehouse-worker displacement, wages, injury rates → labour-economics and
  occupational-safety literature.
- **ETH-2** — ISO 3691-4 personnel-protection-zone requirements → the standard text itself.
- **ETH-4** — energy per delivery → an engineering reference for AMR motor power.
