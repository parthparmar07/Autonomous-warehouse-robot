# Supporting Papers

Six papers underpinning Project 25. Full analysis, including what each one changes about our design
decisions, is in [`docs/07_literature_review.md`](../docs/07_literature_review.md). The
coursework-facing summaries (Assignment 1 format) are in
[`docs/09_assignment1_paper_summaries.md`](../docs/09_assignment1_paper_summaries.md).

| # | Paper | Venue | Type | Preprint | Supports |
|---|---|---|---|---|---|
| P1 | [Monotonic Value Function Factorisation for Deep MARL (QMIX)](01b-qmix-jmlr-rashid-2020.pdf) | **JMLR** 21(178):1–51, 2020 | Journal | [1803.11485](https://arxiv.org/abs/1803.11485) | OBJ-1, FR-1, FR-2, ADR-002 |
| P2 | [Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks](02-benchmarking-marl-papoudakis-2021.pdf) | NeurIPS 2021 D&B | Conference | [2006.07869](https://arxiv.org/abs/2006.07869) | OBJ-2, FR-8, RWARE domain |
| P3 | [Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative MARL](03-monotonicity-tricks-hu-2021.pdf) | arXiv; ICLR Blogposts 2023 | ⚠️ Preprint | [2102.03479](https://arxiv.org/abs/2102.03479) | ADR-002, ADR-005, RSK-05 |
| P4 | [Safe Multi-Agent Reinforcement Learning via Shielding](04-safe-marl-shielding-elsayedaly-2021.pdf) | AAMAS 2021, pp. 483–491 | Conference | [2101.11196](https://arxiv.org/abs/2101.11196) | OBJ-4, FR-3, NFR-3, ETH-3, ADR-003 |
| P5 | [PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning](05-primal-mapf-sartoretti-2019.pdf) | **IEEE RA-L** 4(3):2378–2385, 2019 | Journal | [1809.03531](https://arxiv.org/abs/1809.03531) | OBJ-3, OBJ-5, warehouse MAPF |
| P6 | [A Review of Cooperative Multi-Agent Deep Reinforcement Learning](06-cooperative-marl-review-oroojlooy-2023.pdf) | **Applied Intelligence** 53(11):13677–13722, 2023 | Journal | [1908.03963](https://arxiv.org/abs/1908.03963) | OBJ-2 framing, field taxonomy |

**Venue mix:** 3 journals · 2 top-tier peer-reviewed conferences · 1 preprint.

## Copies stored here

| File | What it is |
|---|---|
| `01b-qmix-jmlr-rashid-2020.pdf` | **The version we cite** — JMLR journal article, 51 pp. |
| `01-qmix-rashid-2018.pdf` | Earlier ICML 2018 conference version, kept for reference |
| `02`–`05` | arXiv preprints (P5's version of record is IEEE RA-L, [10.1109/LRA.2019.2903261](https://doi.org/10.1109/LRA.2019.2903261)) |
| `06` | arXiv v4 preprint (version of record is *Applied Intelligence*, [10.1007/s10489-022-04105-y](https://doi.org/10.1007/s10489-022-04105-y)) |

All are open-access, retained for study under normal academic use.

## Note on P3

P3 is the one source that is **not peer-reviewed** — an arXiv preprint later presented on the ICLR
Blogposts track. It is cited for its empirical ablation study, not as theoretical authority, and its
status is stated openly wherever it appears. It is retained because it materially changed two design
decisions ([ADR-002](../docs/adr/ADR-002-custom-qmix-over-epymarl.md),
[ADR-005](../docs/adr/ADR-005-nfr2-as-hypothesis.md)). With six sources, the five-paper coursework
minimum is satisfied by peer-reviewed work alone.

## Known gaps

These six cover the technical core. They do **not** cover three Responsible AI requirements, which
need non-ML sources and are a **W5 (milestone M3)** task:

- **ETH-1** — warehouse-worker displacement, wages, injury rates → labour-economics and
  occupational-safety literature.
- **ETH-2** — ISO 3691-4 personnel-protection-zone requirements → the standard text itself.
- **ETH-4** — energy per delivery → an engineering reference for AMR motor power.

A warehouse-operations source would also strengthen W4 environment design. The strongest candidate,
Azadeh, De Koster & Roy (2019), *Transportation Science* 53(4):917–945, is paywalled with no
open-access copy, so it is not stored here.
