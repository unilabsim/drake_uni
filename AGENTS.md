# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

This repository is **DrakeUni** (PyPI project `drake-uni-runtime`, version `0.1.0`), an
experimental batch simulation runtime for the UniLab Drake backend. It keeps the
MJCF-facing contract used by UniLab in Python and delegates batched physics
stepping to an optional C++/pybind11 extension linked against a local Drake
installation. It is not a complete Drake backend and must not absorb task,
reward, rollout, or training orchestration logic from UniLab.

The layers are intentionally separated:

- **MJCF contract/materializer** (`drake_uni.runtime.mjcf_model_parser`): parses
  UniLab scenes, expands inherited defaults, and creates a temporary
  Drake-compatible copy without modifying caller-owned assets.
- **Python runtime** (`drake_uni.runtime`): validates configuration, owns model
  metadata and state/sensor arrays, and presents the stable integration API.
- **Native executor** (`drake_uni.compiled._drake_env_pool`, C++20): owns Drake
  diagrams/simulators and worker state, then executes batched reset/step/query
  operations. The extension is optional until it is built locally.

The intended call path is:

```text
UniLab DrakeBackend
  -> drake_uni.runtime
      -> drake_uni.compiled.DrakeEnvPool
          -> Drake C++ simulation
```

## Repository Layout

```text
src/drake_uni/
  __init__.py                 # lazy top-level exports
  batch_env.py                # stable native-pool import boundary
  runtime/
    __init__.py               # runtime factory and diagnostics
    batch.py                  # validation, state/sensor API, lifecycle
    mjcf_model_parser.py      # MJCF contracts and Drake materialization
    types.py                  # public dataclasses
  compiled/
    __init__.py               # optional extension loader and diagnostics
    drake_env_pool.cc         # C++20 pybind11 batch executor
scripts/
  build_drake_batch.py        # compile the extension against DRAKE_HOME
tests/
  test_mjcf_model_parser.py   # parser/materialization and actuator contracts
Makefile                      # sync/install/build/lint/test/package/clean
.github/workflows/ci.yml     # pull-request and branch checks
.github/workflows/release.yml # sdist checks and tag-based PyPI publishing
```

The generated extension (`src/drake_uni/compiled/_drake_env_pool*.so`, or the
platform equivalent) is local build output and is ignored by Git.

## Build and Test Commands

Use **uv** for environment management and invoke project tools through `uv run`.
The normal development flow is:

```bash
make sync
make install
make lint
make test
make check                    # lint + test
```

Building the native executor requires a Drake C++ install prefix. The helper
expects `include/drake/`, `include/pybind11/`, and (on Linux) `lib/libdrake.so`:

```bash
make build DRAKE_HOME=/path/to/drake/install
make build-dry-run DRAKE_HOME=/path/to/drake/install
make test-no-sync             # preserve the just-built extension
```

Equivalent raw commands:

```bash
uv sync
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
uv run ruff check .
uv run pytest -q
uv build --sdist
```

`make install` installs the Python package only; it does not guess a Drake
installation or build native code. Rebuild after changing Drake, Python, the
compiler, or the active virtual environment. `build_drake_batch.py` uses C++20,
links directly to the selected Drake prefix, and writes an extension named with
the active Python ABI suffix. Do not commit that generated file.

## Testing Instructions

- `make test` runs the complete pytest suite.
- Parser tests that reference UniLab robot assets use a fixed local path and
  skip themselves when that checkout is unavailable; this is expected on a
  clean CI runner.
- The current suite does not claim native numerical parity. When changing the
  C++ executor, add a focused test or a reproducible Drake-backed smoke test
  before changing public behavior.
- Run `make check` before committing. If a native build is present, use
  `make test-no-sync` after switching or rebuilding Drake so uv does not replace
  the active environment.

## Code Style Guidelines

- Ruff is configured in `pyproject.toml`: line length 100, Python 3.10 syntax,
  source roots `src` and `tests`, and rules `E`, `F`, `I`, `N`, `W` (with the
  documented naming exceptions).
- Python code uses four-space indentation, `from __future__ import annotations`,
  and modern standard-library typing.
- Keep comments and docstrings in English; user-facing Chinese documentation
  may be added alongside the English README.
- Preserve the lazy import boundary. Importing `drake_uni` should not eagerly
  import `pydrake` or require the optional native extension.
- Keep asset/XML parsing and materialization on cold paths (construction,
  initialization, or cache creation). `step` and `reset` must operate on the
  already prepared numeric contracts and arrays.

## Public API and Scope Boundaries

The stable application entry point is `drake_uni.runtime`:

```python
from drake_uni.runtime import DrakeBatchConfig, create_runtime
```

`drake_uni.batch_env.DrakeEnvPool` is a lower-level native boundary. Preserve
the exported names (`DrakeEnvPool`, `batch_available`, and
`batch_import_error`) when changing the loader.

In scope: MJCF contract compatibility, Drake materialization, batched executor
correctness, diagnostics, and build tooling. Out of scope: Drake source forks,
solver/contact/integrator changes, UniLab task YAMLs, reward/observation policy,
rollout scheduling, distributed execution, and silently downloading a private
Drake toolchain in CI.

## CI and Release

- `.github/workflows/ci.yml` runs Ruff and pytest on Ubuntu, macOS, and Windows
  for Python 3.10 and 3.13, then builds an sdist.
- `.github/workflows/release.yml` performs the same artifact installation/test
  check on tags matching `v*` (or manual dispatch). Only the source distribution
  is published, using PyPI trusted publishing (OIDC); no API token belongs in
  the repository.
- CI intentionally does not build the native extension because a Drake C++
  prefix is machine-specific. Native builds are validated in environments that
  explicitly provide `DRAKE_HOME`.

## GitHub CLI (`gh`)

Remote: `github.com/unilabsim/drake_uni`.

```bash
gh issue list
gh issue view <number>
gh pr list
gh pr view <number>
gh run list
gh run list --workflow=ci.yml
gh run view <run-id>
gh run list --status=failure
```

Before creating or updating a PR, finish the commit, ensure
`git status --short --branch` is clean, and record the `make check` result in the
PR body. If the PR targets `main`, inspect the workflow result for the final
head SHA before reporting it complete.
