# Project Brief (source of truth)

> **⚠️ ACTION REQUIRED — paste the original brief below this line, replacing this block.**
>
> Every other document in `docs/` cites this brief by section (§3.2, §9, FR-1, NFR-2, OBJ-4, …), so
> it needs to live in the repo verbatim. It is deliberately **not** transcribed by hand: a
> paraphrased or mis-copied requirement is worse than no copy at all, and the original text is the
> authority when a downstream document and the brief disagree.
>
> Paste the full text of *"Project 25: Autonomous Warehouse Robot Fleet Coordination Using
> Multi-Agent Reinforcement Learning"* — sections 1 through 13 — exactly as issued.

---

## Deviations from the brief

Where the project departs from the brief as written, the deviation is recorded here and in the
requirements change log (`03_requirements_specification.md` §7). Nothing is changed silently.

| Brief says | Project does | Reason | Record |
|---|---|---|---|
| epymarl **or** custom PyTorch QMIX (epymarl listed first; custom listed as contingency) | Custom PyTorch QMIX is primary; epymarl is a debugging oracle only | epymarl is unmaintained and conflicts with the required MLOps stack; obstructs ONNX export | ADR-002 |
| PyBullet + grid environment (reads as one environment) | Two artefacts: canonical NumPy grid env for all training/eval; PyBullet as a trajectory **replay** device for the demo | Physics-stepped training is infeasible within the semester and breaks NFR-6 | ADR-001 |
| Safety layer described as a pre-action check (§4) **and** as post-processing on the action vector (§9) | Both, layered: action masking at selection (train *and* eval, identically) **plus** a redundant post-hoc assertion in the service | The two readings have different training dynamics; masking preserves the learning signal and the FR-3 diagnostic | ADR-003 |
| NFR-2 stated as an acceptance criterion (QMIX−IQL gap maintained or increased with N) | Pre-registered hypothesis **H1**; the deliverable is the measured gap with CIs, either direction | Not an engineering-controllable outcome; as a criterion it incentivises tuning on the sealed evaluation set | ADR-005 |
| Python 3.10+; PyBullet in the main stack | Python 3.11; PyBullet moved to `requirements-viz.txt`, demo runs on WSL2/Docker/Colab | PyBullet ships no Windows wheels for any modern version and no wheels at all for cp312/cp313 | ADR-004 |

Two requirements were **added** rather than changed (FR-7, FR-8, NFR-7, NFR-8, ETH-5) — see the
change log for why each exists.
