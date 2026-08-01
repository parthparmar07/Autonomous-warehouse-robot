# ADR-004: Python 3.11 toolchain; PyBullet isolated as an optional visualisation dependency

**Status:** Accepted (W1–W3) · **Deciders:** R3 lead · **Verified empirically on 2026-08-01**

---

## Context

The development machine is Windows 11 Home with a system Python 3.13.2. The brief specifies
"Python 3.10+". Two questions had to be settled before any code was written: which interpreter, and
how to handle PyBullet on Windows.

**Empirical findings (verified against the PyPI API and by attempting the build, not assumed):**

1. **PyBullet's newest release (3.2.7) publishes wheels for cp36–cp311 only.** There is no cp312 or
   cp313 wheel on *any* platform. On Python 3.13 it would build from source everywhere.
2. **PyBullet publishes no Windows wheels for any modern version.** The last `win_amd64` wheel was
   `pybullet-2.6.9-cp27-cp27m-win_amd64.whl`. Every version from 3.x onward compiles from source on
   Windows regardless of Python version.
3. **The source build fails on this machine.** MSVC is present, but there is no Windows SDK
   (`C:\Program Files (x86)\Windows Kits\10\Include` does not exist), so the compile dies at
   `fatal error C1083: Cannot open include file: 'string.h'`. Fixing this means a multi-gigabyte
   Visual Studio Build Tools + Windows SDK install.
4. **Docker Desktop is installed**; WSL is present but hosts only the `docker-desktop` distro — no
   Ubuntu distro yet.

Finding 2 is the important correction to our initial reasoning: the obstacle is the *platform*, not
the Python version. Finding 1 still makes 3.11 the right choice, but for a different reason —
3.11 is the newest interpreter for which a prebuilt PyBullet exists at all, so the Linux path
(WSL2 or Docker) installs instantly on 3.11 and would not on 3.13.

## Decision

**1. Python 3.11 for the project venv**, provisioned with `uv` (`uv venv --python 3.11`), which
downloads a managed CPython rather than requiring a system install. Rationale:
- It is the newest interpreter with a prebuilt PyBullet wheel (Linux), which keeps the demo path cheap.
- The whole scientific stack has mature 3.11 wheels; 3.13 support is still uneven across
  Evidently/CodeCarbon's transitive dependencies.
- Satisfies the brief's "3.10+" and leaves the system 3.13 untouched.

**2. PyBullet moves out of `requirements.txt` into `requirements-viz.txt`.** It is not installed on
the Windows dev machine and is not needed for training, evaluation, the service, or CI. This is not
a workaround — it is the packaging expression of ADR-001 and FR-7: the renderer is a replay device
isolated from every code path that produces a reported number.

**3. The 3D demo runs on Linux.** Decision deferred to W11 (it is off the critical path); two viable
zero-cost options, both already available:
- **WSL2 + Ubuntu 22.04** — `pip install pybullet` gets the cp311 manylinux wheel instantly, and
  WSLg on Windows 11 provides a live GUI window. Best fit for the W15 *live* demo.
- **Docker (already installed) or Google Colab** — headless render to MP4 for the recorded demo.

**4. Fallback if both fail:** a matplotlib 2D animation of the same recorded trajectory. Degrades
the demo's polish, degrades no result.

## Consequences

**Positive**
- Core stack installs cleanly and is verified working: Python 3.11.12, torch 2.13.0+cpu,
  numpy 2.1.3, mlflow 3.15.0, optuna 4.9.0, onnxruntime 1.28.0, gymnasium 1.3.0.
- Zero build-tool dependency for the critical path. A new team member clones and runs in minutes.
- CI images stay small and fast — no compiler toolchain, no PyBullet build. Directly protects the
  Actions free-tier budget (RSK-09).
- RSK-07 (PyBullet install issues) is largely retired: it can now only affect a deliverable that is
  isolated by design and has two fallbacks.

**Negative / accepted**
- The 3D demo cannot be developed natively on Windows without WSL2/Docker/Colab. Accepted — it is
  off the critical path and the alternatives are free and already installed.
- Two requirements files instead of one. Mitigated by both carrying a header explaining why.
- Python 3.11 rather than the newest interpreter. No practical cost for this stack.

**Follow-up actions**
- W1: add a `wsl --install -d Ubuntu-22.04` step to the demo-environment setup notes (not blocking).
- W11: choose live-GUI (WSLg) vs. recorded-MP4 (Docker/Colab) for the demo.
- Pin exact versions in a lockfile before the W9 trial reproduction run (RSK-10).

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| System Python 3.13 | No PyBullet wheel exists for cp312/cp313 on any platform, so the Linux demo path would also need a source build; uneven 3.13 support in transitive deps. |
| Install Visual Studio Build Tools + Windows SDK | Multi-gigabyte download, likely needs admin rights, and buys only a locally-built copy of a demo-only dependency. Poor effort-to-value ratio. |
| Keep PyBullet in `requirements.txt` and let it fail | Breaks `pip install -r requirements.txt` for every team member and every CI run, for a dependency none of them need. |
| Develop the whole project inside WSL2 | Viable and cleaner in some ways, but adds a filesystem/tooling boundary for all four members to manage all semester, to solve a demo-only problem. |
| Drop the 3D demo | Abandons a stated deliverable (EO-4) and the best presentation asset, when free alternatives exist. |
