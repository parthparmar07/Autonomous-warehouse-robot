# Architecture Decision Records

Each ADR records one decision that shapes the project: the context that forced it, what was decided,
what it costs us, and what was rejected. They exist so that in W12 nobody has to reconstruct why the
environment is split in two, and so the viva has a defensible answer for every non-obvious choice.

**Status values:** Proposed → Accepted → (Superseded by ADR-NNN). An accepted ADR is amended, never
silently rewritten.

| ADR | Decision | Status | Drives |
|---|---|---|---|
| [ADR-001](ADR-001-dual-environment.md) | NumPy grid world is the canonical training/eval environment; PyBullet is a trajectory **replay** device for the demo only | Accepted | FR-7, NFR-6, EO-4 |
| [ADR-002](ADR-002-custom-qmix-over-epymarl.md) | Custom PyTorch QMIX is primary; epymarl retained only as a cross-check oracle | Accepted | OBJ-1, EO-3, RSK-05 |
| [ADR-003](ADR-003-safety-layer-action-masking.md) | Safety = action masking at selection, identical in training and evaluation, plus a redundant post-hoc service assertion | Accepted | FR-3, NFR-3, ETH-3, RSK-06 |
| [ADR-004](ADR-004-python-311-toolchain.md) | Python 3.11 venv; PyBullet isolated into `requirements-viz.txt`; demo runs on WSL2/Docker/Colab | Accepted | NFR-6, RSK-07, RSK-09 |
| [ADR-005](ADR-005-nfr2-as-hypothesis.md) | NFR-2 reclassified from acceptance criterion to pre-registered hypothesis H1 | Accepted | NFR-2, ETH-5, RSK-04 |

## Pending decisions

| Topic | Due | Owner |
|---|---|---|
| Reward function shape (sparse delivery bonus vs. shaped distance reward) | W4, before the env freeze | R1 + R2 |
| Observation window size and whether agents observe other agents' task assignments | W4, before the env freeze | R1 |
| Live-GUI (WSLg) vs. recorded-MP4 (Docker/Colab) for the 3D demo | W11 | R3 |
| Curriculum strategy for N=20 (fresh init vs. warm-start from N=10) | W9 | R2 |
