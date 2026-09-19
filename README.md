# DrakeUni

[简体中文](README_zh.md)

Experimental Drake batch simulation runtime for UniLab's Drake backend adapter.

Current scope:

- Compiled C++/pybind batch pool driven by parsed MJCF model contracts.
- `nbatch` and `nthread` worker control.
- Python API modeled after MuJoCoUni's `BatchEnvPool` direction.
- Intended to be consumed by UniLab's Drake backend adapter.

This is not a full Drake backend yet. The current runtime expects a model file,
environment count, simulation step size, and worker count. Task semantics such
as base-body meaning, named sensor views, rewards, and observations belong to
the consuming UniLab backend/task layer.

The input model is the original UniLab MJCF scene, for example
`scene_flat.xml`. DrakeUni materializes a temporary Drake-compatible copy at
runtime by expanding MJCF defaults and converting unsupported mesh formats when
needed. Callers should not commit or pass pre-generated files such as
`*_drake.xml` or converted OBJ asset directories.

## Directory Roles

`src/drake_uni/runtime/` is the Python-facing runtime interface. It owns the
public config/data contracts, MJCF contract parsing, Drake-compatible MJCF
materialization, runtime construction, and the reset/step/sensor API used by
UniLab.

`src/drake_uni/compiled/` is the native extension layer. It contains the C++
Drake batch pool source and the built pybind extension that performs batched
physics stepping.

The intended call path is:

```text
UniLab DrakeBackend
  -> drake_uni.runtime
      -> drake_uni.compiled.DrakeEnvPool
          -> Drake C++ simulation
```

## Build Batch Extension

```bash
uv run python scripts/build_drake_batch.py --drake-home /path/to/drake/install
```

The extension is written to `src/drake_uni/compiled/_drake_env_pool*.so`.
The Drake prefix must provide `include/drake`, `include/pybind11`, and the
Drake shared library. The generated extension is local build output and is not
committed.

## Development

The repository uses [uv](https://docs.astral.sh/uv/) for reproducible local
development:

```bash
make sync
make install
make check
make build DRAKE_HOME=/path/to/drake/install
```

`make check` runs Ruff and pytest. Parser tests that depend on a sibling UniLab
checkout skip themselves when its assets are unavailable. See `AGENTS.md` for
the native build contract, scope boundaries, and release workflow.

## Install Editable

From a consuming project:

```bash
uv pip install -e /path/to/drake_uni
```

## Runtime API

```python
from drake_uni.runtime import DrakeBatchConfig, create_runtime

runtime = create_runtime(
    DrakeBatchConfig(
        model_file="/path/to/scene_flat.xml",
        num_envs=32,
        sim_dt=0.002,
        nthread=8,
    )
)
```

The preferred integration point is `drake_uni.runtime`. `DrakeEnvPool` and
the compiled extension are lower-level implementation details.

## Native model property readback

`DrakeBatchRuntime.native_model_properties()` returns a cold-path, versioned
snapshot of the materialized Drake model without exposing Drake objects. The
current property contract version is 1.

```python
properties = runtime.native_model_properties()
assert properties.contract_version == 1
```

The snapshot contains:

- Drake body order, names, default masses, COM vectors, and full 3×3 rotational
  inertias. Inertia is about the body origin and expressed in the body frame.
  The welded world body is reported as a massless static body with zero COM and
  inertia.
- SceneGraph geometries in body-index/name order, with normalized names, owner
  body indices, primitive shape types, native parameters, and proximity-role
  flags. Numeric arrays are detached and read-only.

Primitive parameters are packed as three native values per geometry: sphere is
`(radius, 0, 0)`; box and ellipsoid use three full lengths/axes; capsule and
cylinder use `(radius, length, 0)`; half-space uses zeros. Meshes, convex
meshes, and other shapes without an exact scalar identity fail closed instead of
returning misleading parameters.
