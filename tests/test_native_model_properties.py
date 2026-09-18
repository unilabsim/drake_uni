from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drake_uni.batch_env import batch_available
from drake_uni.runtime import (
    DrakeBatchConfig,
    DrakeBatchRuntime,
    DrakeNativeModelProperties,
    create_runtime,
)
from drake_uni.runtime.types import NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION

requires_native = pytest.mark.skipif(
    not batch_available(), reason="DrakeEnvPool extension is unavailable"
)


def _properties(**overrides: Any) -> DrakeNativeModelProperties:
    values: dict[str, Any] = {
        "contract_version": NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION,
        "body_names": ("world", "base"),
        "body_masses": np.asarray([0.0, 1.25]),
        "body_coms": np.zeros((2, 3)),
        "body_inertias": np.zeros((2, 3, 3)),
        "geometry_names": ("ball",),
        "geometry_body_indices": np.asarray([1], dtype=np.int32),
        "geometry_types": ("sphere",),
        "geometry_collision": (True,),
        "geometry_parameters": np.asarray([[0.07, 0.0, 0.0]]),
    }
    values.update(overrides)
    return DrakeNativeModelProperties(**values)


def test_native_model_properties_contract_returns_detached_read_only_values() -> None:
    source_masses = np.asarray([0.0, 1.25], dtype=np.float32)
    source_indices = np.asarray([1], dtype=np.int64)
    properties = _properties(body_masses=source_masses, geometry_body_indices=source_indices)

    source_masses[0] = 9.0
    source_indices[0] = 0

    assert properties.body_names == ("world", "base")
    assert properties.geometry_names == ("ball",)
    assert properties.geometry_types == ("sphere",)
    assert properties.geometry_collision == (True,)
    assert properties.body_masses.dtype == np.dtype(np.float64)
    assert properties.body_coms.dtype == np.dtype(np.float64)
    assert properties.body_inertias.dtype == np.dtype(np.float64)
    assert properties.geometry_parameters.dtype == np.dtype(np.float64)
    assert properties.geometry_body_indices.dtype == np.dtype(np.int32)
    np.testing.assert_allclose(properties.body_masses, [0.0, 1.25])
    assert properties.geometry_body_indices.tolist() == [1]
    for array in (
        properties.body_masses,
        properties.body_coms,
        properties.body_inertias,
        properties.geometry_body_indices,
        properties.geometry_parameters,
    ):
        assert not array.flags.writeable


def test_native_model_properties_contract_rejects_coercible_invalid_types() -> None:
    with pytest.raises(TypeError):
        _properties(body_names=("world", 1))
    with pytest.raises(TypeError):
        _properties(geometry_types=(1,))
    with pytest.raises(TypeError):
        _properties(geometry_collision=(1,))
    with pytest.raises(ValueError):
        _properties(contract_version=2)
    with pytest.raises(ValueError):
        _properties(geometry_types=("mesh",))


def test_runtime_validates_and_detaches_native_output() -> None:
    source = {
        "body_names": ["world", "base"],
        "body_masses": np.asarray([0.0, 1.25]),
        "body_coms": np.zeros((2, 3)),
        "body_inertias": np.zeros((2, 3, 3)),
        "geometry_names": ["ball"],
        "geometry_body_indices": np.asarray([1]),
        "geometry_types": ["sphere"],
        "geometry_collision": [True],
        "geometry_parameters": np.asarray([[0.07, 0.0, 0.0]]),
    }

    class FakePool:
        def native_model_properties(self) -> dict[str, Any]:
            return source

    runtime = DrakeBatchRuntime.__new__(DrakeBatchRuntime)
    runtime._pool = FakePool()
    runtime._native_model_properties = None
    properties = runtime.native_model_properties()

    source["body_masses"][0] = 9.0
    source["geometry_parameters"][0, 0] = 9.0
    assert properties.body_masses.tolist() == [0.0, 1.25]
    assert properties.geometry_parameters.tolist() == [[0.07, 0.0, 0.0]]


def test_runtime_fails_closed_for_old_native_builds() -> None:
    class OldPool:
        pass

    runtime = DrakeBatchRuntime.__new__(DrakeBatchRuntime)
    runtime._pool = OldPool()
    runtime._native_model_properties = None

    with pytest.raises(RuntimeError, match="does not expose native_model_properties"):
        runtime.native_model_properties()


def test_runtime_rejects_non_native_float_dtypes() -> None:
    class FakePool:
        def native_model_properties(self) -> dict[str, Any]:
            return {
                "body_names": ["world"],
                "body_masses": np.asarray([0.0], dtype=np.float32),
                "body_coms": np.zeros((1, 3), dtype=np.float32),
                "body_inertias": np.zeros((1, 3, 3), dtype=np.float32),
                "geometry_names": [],
                "geometry_body_indices": np.asarray([], dtype=np.int32),
                "geometry_types": [],
                "geometry_collision": [],
                "geometry_parameters": np.asarray([], dtype=np.float32).reshape((0, 3)),
            }

    runtime = DrakeBatchRuntime.__new__(DrakeBatchRuntime)
    runtime._pool = FakePool()
    runtime._native_model_properties = None

    with pytest.raises(RuntimeError, match="body_masses must contain float64"):
        runtime.native_model_properties()


@requires_native
def test_native_model_properties_reads_drake_body_and_geometry_identity(tmp_path: Path) -> None:
    scene = _write_scene(
        tmp_path,
        """<geom name="ball" type="sphere" size=".07" contype="1" conaffinity="1"/>
           <geom name="block" type="box" size=".1 .2 .3" contype="1" conaffinity="1"/>""",
    )
    runtime = create_runtime(
        DrakeBatchConfig(model_file=str(scene), num_envs=1, sim_dt=0.002, nthread=1)
    )
    try:
        properties = runtime.native_model_properties()
    finally:
        runtime.close()

    assert properties.contract_version == NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION
    assert properties.body_names == ("world", "base", "link")
    np.testing.assert_allclose(properties.body_masses, [0.0, 1.25, 0.25])
    np.testing.assert_allclose(
        properties.body_coms,
        [[0.0] * 3, [0.01, 0.02, 0.03], [0.0] * 3],
    )

    # The readback is about the body origin and includes the parallel-axis
    # terms for the 1.25 kg COM offset (.01, .02, .03).
    np.testing.assert_allclose(
        properties.body_inertias[1],
        [
            [0.111625, -0.00025, -0.000375],
            [-0.00025, 0.12125, -0.00075],
            [-0.000375, -0.00075, 0.130625],
        ],
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert properties.geometry_names == ("ball", "block")
    assert properties.geometry_body_indices.tolist() == [1, 1]
    assert properties.geometry_types == ("sphere", "box")
    assert properties.geometry_collision == (True, True)
    np.testing.assert_allclose(
        properties.geometry_parameters,
        [
            [0.07, 0.0, 0.0],
            [0.2, 0.4, 0.6],
        ],
    )


@requires_native
def test_native_model_properties_fails_closed_for_mesh_identity(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.obj"
    mesh.write_text(
        "\n".join(
            [
                "v -1 -1 -1",
                "v 1 -1 -1",
                "v 1 1 -1",
                "v -1 1 -1",
                "v -1 -1 1",
                "v 1 -1 1",
                "v 1 1 1",
                "v -1 1 1",
                "f 1 2 3",
                "f 1 3 4",
                "f 5 6 7",
                "f 5 7 8",
                "f 1 2 6",
                "f 1 6 5",
                "f 2 3 7",
                "f 2 7 6",
                "f 3 4 8",
                "f 3 8 7",
                "f 4 1 5",
                "f 4 5 8",
            ]
        )
        + "\n"
    )
    scene = _write_scene(
        tmp_path,
        '<geom name="mesh_geom" type="mesh" mesh="mesh" contype="1" conaffinity="1"/>',
        assets='<asset><mesh name="mesh" file="mesh.obj"/></asset>',
    )
    runtime = create_runtime(
        DrakeBatchConfig(model_file=str(scene), num_envs=1, sim_dt=0.002, nthread=1)
    )
    try:
        with pytest.raises(RuntimeError, match="native geometry identity is unsupported"):
            runtime.native_model_properties()
    finally:
        runtime.close()


def _write_scene(
    directory: Path,
    geometry_xml: str,
    *,
    assets: str = "",
) -> Path:
    scene = directory / "native_properties.xml"
    scene.write_text(
        f"""<mujoco model="native_properties">
  {assets}
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <inertial pos=".01 .02 .03" mass="1.25" diaginertia=".11 .12 .13"/>
      {geometry_xml}
      <body name="link">
        <joint name="hinge"/>
        <inertial pos="0 0 0" mass=".25" diaginertia=".02 .03 .04"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="drive" joint="hinge"/></actuator>
</mujoco>"""
    )
    return scene
