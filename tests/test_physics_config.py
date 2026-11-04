from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drake_uni.batch_env import batch_available
from drake_uni.runtime import (
    DrakeBatchConfig,
    DrakeBatchRuntime,
    DrakeEffectivePhysics,
    DrakeGeometryMaterial,
    DrakePhysicsConfig,
    create_runtime,
)
from drake_uni.runtime.batch import _build_physics_payload
from drake_uni.runtime.mjcf_model_parser import parse_mjcf_model_contract

requires_native = pytest.mark.skipif(
    not batch_available(), reason="DrakeEnvPool extension is unavailable"
)


def _write_physics_scene(
    directory: Path,
    *,
    gravcomp: str = "",
    frictionloss: str = "",
    geom_attrs: str = "",
    name: str = "physics_scene",
) -> Path:
    scene = directory / f"{name}.xml"
    scene.write_text(
        f"""<mujoco model="{name}">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="base" pos="0 0 1"{gravcomp}>
      <freejoint name="root"/>
      <geom name="ball" type="sphere" size="0.05" mass="1"{geom_attrs}/>
      <body name="link">
        <joint name="hinge"{frictionloss}/>
        <geom name="tip" type="sphere" size="0.01" mass="0.01" pos="0 0 -0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="drive" joint="hinge"/></actuator>
</mujoco>"""
    )
    return scene


def _materials() -> dict[str, DrakeGeometryMaterial]:
    return {
        "floor": DrakeGeometryMaterial("rigid"),
        "ball": DrakeGeometryMaterial(
            "compliant",
            hydroelastic_modulus=1.0e5,
            resolution_hint=0.01,
            hunt_crossley_dissipation=50.0,
        ),
    }


def _report(**overrides: Any) -> DrakeEffectivePhysics:
    values: dict[str, Any] = {
        "contact_model": "point",
        "discrete_contact_approximation": "sap",
        "penetration_allowance": 1.0e-4,
        "geometry_materials": {},
        "field_status": {"contact_model": "applied"},
        "ignored_mjcf_fields": (),
        "drake_version": "1.51.1",
    }
    values.update(overrides)
    return DrakeEffectivePhysics(**values)


def test_batch_config_defaults_preserve_historical_behavior() -> None:
    config = DrakeBatchConfig(model_file="scene.xml", num_envs=2, sim_dt=1 / 240)
    assert config.physics is None
    physics = DrakePhysicsConfig()
    assert physics.contact_model == "point"
    assert physics.discrete_contact_approximation == "sap"
    assert physics.penetration_allowance == 1.0e-4
    assert physics.geometry_materials == {}
    assert physics.frictionloss_regularization_speed is None
    assert physics.preserve_gravity_compensation is False


def test_physics_config_rejects_unknown_enums_and_bad_scalars() -> None:
    with pytest.raises(ValueError, match="contact_model"):
        DrakePhysicsConfig(contact_model="hydroelastic_only")
    with pytest.raises(ValueError, match="discrete_contact_approximation"):
        DrakePhysicsConfig(discrete_contact_approximation="tarski")
    with pytest.raises(ValueError, match="penetration_allowance"):
        DrakePhysicsConfig(penetration_allowance=0.0)
    with pytest.raises(ValueError, match="penetration_allowance"):
        DrakePhysicsConfig(penetration_allowance=np.inf)
    with pytest.raises(TypeError):
        DrakePhysicsConfig(penetration_allowance=True)
    with pytest.raises(ValueError, match="frictionloss_regularization_speed"):
        DrakePhysicsConfig(frictionloss_regularization_speed=-1.0e-6)
    with pytest.raises(TypeError):
        DrakePhysicsConfig(preserve_gravity_compensation=1)
    with pytest.raises(TypeError):
        DrakeBatchConfig(
            model_file="scene.xml", num_envs=1, sim_dt=0.002, physics={"contact_model": "point"}
        )


def test_geometry_material_validation() -> None:
    with pytest.raises(ValueError, match="hydroelastic_type"):
        DrakeGeometryMaterial("soft")
    with pytest.raises(ValueError, match="hydroelastic_modulus"):
        DrakeGeometryMaterial("compliant", resolution_hint=0.01)
    with pytest.raises(ValueError, match="resolution_hint"):
        DrakeGeometryMaterial("compliant", hydroelastic_modulus=1.0e5)
    with pytest.raises(ValueError, match="rigid"):
        DrakeGeometryMaterial("rigid", hydroelastic_modulus=1.0e5)
    with pytest.raises(ValueError, match="hunt_crossley_dissipation"):
        DrakeGeometryMaterial("rigid", hunt_crossley_dissipation=-1.0)
    material = DrakeGeometryMaterial("compliant", hydroelastic_modulus=1, resolution_hint=1)
    assert material.hydroelastic_modulus == 1.0
    with pytest.raises(TypeError):
        DrakePhysicsConfig(
            contact_model="hydroelastic_with_fallback",
            geometry_materials={"ball": {"hydroelastic_type": "rigid"}},
        )


def test_physics_config_rejects_unsupported_combinations() -> None:
    # Materials cannot take effect under point contact: reject instead of
    # silently ignoring them.
    with pytest.raises(ValueError, match="hydroelastic contact model"):
        DrakePhysicsConfig(geometry_materials=_materials())
    # Strict hydroelastic without any material assignment fails at config time.
    with pytest.raises(ValueError, match="strict"):
        DrakePhysicsConfig(contact_model="hydroelastic")


def test_payload_rejects_unknown_and_missing_materials(tmp_path: Path) -> None:
    scene = _write_physics_scene(tmp_path)
    contract = parse_mjcf_model_contract(scene)
    unknown = DrakePhysicsConfig(
        contact_model="hydroelastic_with_fallback",
        geometry_materials={"missing": DrakeGeometryMaterial("rigid")},
    )
    with pytest.raises(ValueError, match="unknown collision geometries"):
        _build_physics_payload(unknown, contract, 0.002)
    # Strict hydroelastic must cover every collision geometry: "tip" is missing.
    with pytest.raises(ValueError, match="missing"):
        _build_physics_payload(
            DrakePhysicsConfig(contact_model="hydroelastic", geometry_materials=_materials()),
            contract,
            0.002,
        )
    full = dict(_materials())
    full["tip"] = DrakeGeometryMaterial("rigid")
    payload = _build_physics_payload(
        DrakePhysicsConfig(contact_model="hydroelastic", geometry_materials=full),
        contract,
        0.002,
    )
    assert payload is not None
    assert payload["contact_model"] == "hydroelastic"
    assert set(payload["geometry_materials"]) == {"floor", "ball", "tip"}


def test_payload_rejects_frictionloss_on_free_joint(tmp_path: Path) -> None:
    scene = _write_physics_scene(tmp_path, frictionloss=' frictionloss="0.5"')
    contract = parse_mjcf_model_contract(scene)
    payload = _build_physics_payload(
        DrakePhysicsConfig(frictionloss_regularization_speed=1.0e-6), contract, 0.002
    )
    assert payload is not None
    assert payload["joint_frictionloss"].tolist() == [0.0, 0.5]
    np.testing.assert_allclose(payload["joint_frictionloss"], contract.joint_frictionloss)

    # A ball joint carries a nonzero frictionloss that the approximation
    # cannot represent: reject instead of silently dropping it.
    free_scene = tmp_path / "free_frictionloss.xml"
    free_scene.write_text(
        """<mujoco model="free_frictionloss">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="ball" type="sphere" size="0.05" mass="1"/>
      <body name="link">
        <joint name="elbow" type="ball" frictionloss="0.5"/>
        <geom name="tip" type="sphere" size="0.01" mass="0.01"/>
        <body name="tip_link">
          <joint name="hinge"/>
          <geom name="tip2" type="sphere" size="0.01" mass="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="drive" joint="hinge"/></actuator>
</mujoco>"""
    )
    free_contract = parse_mjcf_model_contract(free_scene)
    with pytest.raises(ValueError, match="single-dof slide/hinge"):
        _build_physics_payload(
            DrakePhysicsConfig(frictionloss_regularization_speed=1.0e-6),
            free_contract,
            0.002,
        )


def test_parser_reports_ignored_mjcf_physics_fields(tmp_path: Path) -> None:
    scene = _write_physics_scene(
        tmp_path,
        gravcomp=' gravcomp="0.8"',
        frictionloss=' frictionloss="0.5"',
        geom_attrs=' friction="1 0.01 0.002" condim="4"',
    )
    contract = parse_mjcf_model_contract(scene)
    ignored = contract.ignored_physics
    assert ignored.joint_frictionloss == {"hinge": 0.5}
    assert ignored.body_gravcomp == {"base": 0.8}
    assert ignored.torsional_friction_geoms == ("ball",)
    assert ignored.rolling_friction_geoms == ("ball",)
    assert ignored.unsupported_condim_geoms == ("ball",)
    assert ignored.ignored_field_names() == (
        "joint.frictionloss",
        "body.gravcomp",
        "geom.torsional_friction",
        "geom.rolling_friction",
        "geom.condim",
    )
    assert ignored.ignored_field_names(
        frictionloss_approximated=True, gravcomp_preserved=True
    ) == (
        "geom.torsional_friction",
        "geom.rolling_friction",
        "geom.condim",
    )

    clean = parse_mjcf_model_contract(_write_physics_scene(tmp_path, name="clean"))
    assert clean.ignored_physics.ignored_field_names() == ()
    assert clean.ignored_physics.joint_frictionloss == {}
    assert clean.ignored_physics.body_gravcomp == {}


def test_effective_physics_validation() -> None:
    with pytest.raises(ValueError, match="contact_model"):
        _report(contact_model="hydroelastic_only")
    with pytest.raises(ValueError, match="field_status"):
        _report(field_status={"contact_model": "forced"})
    with pytest.raises(ValueError, match="ignored_mjcf_fields"):
        _report(ignored_mjcf_fields=("",))
    with pytest.raises(ValueError, match="drake_version"):
        _report(drake_version="")
    report = _report(drake_version=None)
    assert report.drake_version is None


class _FakePool:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def effective_physics(self) -> dict[str, Any]:
        return self._raw


def _runtime_with_pool(
    scene: Path,
    physics: DrakePhysicsConfig | None,
    pool: Any,
) -> DrakeBatchRuntime:
    runtime = DrakeBatchRuntime.__new__(DrakeBatchRuntime)
    runtime._pool = pool
    runtime._config = DrakeBatchConfig(
        model_file=str(scene), num_envs=1, sim_dt=0.002, physics=physics
    )
    runtime._model_contract = parse_mjcf_model_contract(scene)
    runtime._effective_physics = None
    return runtime


def _raw_native(materials: dict[str, dict[str, Any]] | None = None, **overrides: Any) -> dict:
    raw: dict[str, Any] = {
        "contact_model": "point",
        "discrete_contact_approximation": "sap",
        "penetration_allowance": 1.0e-4,
        "geometry_materials": materials or {},
        "drake_version": "1.51.1",
    }
    raw.update(overrides)
    return raw


def test_effective_physics_composes_native_readback_and_field_status(tmp_path: Path) -> None:
    scene = _write_physics_scene(
        tmp_path,
        gravcomp=' gravcomp="0.8"',
        frictionloss=' frictionloss="0.5"',
        geom_attrs=' friction="1 0.01 0.002" condim="4"',
    )
    physics = DrakePhysicsConfig(
        contact_model="hydroelastic_with_fallback",
        discrete_contact_approximation="lagged",
        geometry_materials=_materials(),
        frictionloss_regularization_speed=1.0e-6,
        preserve_gravity_compensation=True,
    )
    raw = _raw_native(
        {
            "floor": {
                "hydroelastic_type": "rigid",
                "hydroelastic_modulus": None,
                "resolution_hint": None,
                "hunt_crossley_dissipation": None,
            },
            "ball": {
                "hydroelastic_type": "compliant",
                "hydroelastic_modulus": 1.0e5,
                "resolution_hint": 0.01,
                "hunt_crossley_dissipation": 50.0,
            },
        },
        contact_model="hydroelastic_with_fallback",
        discrete_contact_approximation="lagged",
    )
    runtime = _runtime_with_pool(scene, physics, _FakePool(raw))
    report = runtime.effective_physics()

    assert report.contact_model == "hydroelastic_with_fallback"
    assert report.discrete_contact_approximation == "lagged"
    assert report.penetration_allowance == 1.0e-4
    assert report.drake_version == "1.51.1"
    assert report.geometry_materials["ball"].hydroelastic_modulus == 1.0e5
    assert report.geometry_materials["ball"].resolution_hint == 0.01
    assert report.geometry_materials["ball"].hunt_crossley_dissipation == 50.0
    assert report.geometry_materials["floor"].hydroelastic_type == "rigid"
    assert report.field_status == {
        "contact_model": "applied",
        "discrete_contact_approximation": "applied",
        "penetration_allowance": "applied",
        "geometry_materials.floor": "applied",
        "geometry_materials.ball": "applied",
        "joint.frictionloss": "approximated",
        "body.gravcomp": "approximated",
        "geom.torsional_friction": "ignored",
        "geom.rolling_friction": "ignored",
        "geom.condim": "ignored",
    }
    assert report.ignored_mjcf_fields == (
        "geom.torsional_friction",
        "geom.rolling_friction",
        "geom.condim",
    )


def test_effective_physics_reports_ignored_fields_by_default(tmp_path: Path) -> None:
    scene = _write_physics_scene(
        tmp_path, gravcomp=' gravcomp="0.8"', frictionloss=' frictionloss="0.5"'
    )
    runtime = _runtime_with_pool(scene, None, _FakePool(_raw_native()))
    report = runtime.effective_physics()

    assert report.contact_model == "point"
    assert report.discrete_contact_approximation == "sap"
    assert report.field_status["joint.frictionloss"] == "ignored"
    assert report.field_status["body.gravcomp"] == "ignored"
    assert report.ignored_mjcf_fields == ("joint.frictionloss", "body.gravcomp")


def test_effective_physics_fails_loudly_on_native_mismatch(tmp_path: Path) -> None:
    scene = _write_physics_scene(tmp_path)
    physics = DrakePhysicsConfig(
        contact_model="hydroelastic_with_fallback",
        geometry_materials=_materials(),
    )
    # The native side must never silently substitute another contact mode.
    mismatched = _raw_native(
        {
            "floor": {
                "hydroelastic_type": "rigid",
                "hydroelastic_modulus": None,
                "resolution_hint": None,
                "hunt_crossley_dissipation": None,
            },
            "ball": {
                "hydroelastic_type": "compliant",
                "hydroelastic_modulus": 1.0e5,
                "resolution_hint": 0.01,
                "hunt_crossley_dissipation": 50.0,
            },
        },
        contact_model="point",
    )
    runtime = _runtime_with_pool(scene, physics, _FakePool(mismatched))
    with pytest.raises(RuntimeError, match="contact_model"):
        runtime.effective_physics()

    missing = _raw_native({}, contact_model="hydroelastic_with_fallback")
    runtime = _runtime_with_pool(scene, physics, _FakePool(missing))
    with pytest.raises(RuntimeError, match="geometry material"):
        runtime.effective_physics()


def test_effective_physics_fails_closed_for_old_native_builds(tmp_path: Path) -> None:
    class OldPool:
        pass

    runtime = _runtime_with_pool(_write_physics_scene(tmp_path), None, OldPool())
    with pytest.raises(RuntimeError, match="does not expose effective_physics"):
        runtime.effective_physics()


@requires_native
def test_effective_physics_distinguishes_point_sap_from_hydroelastic_lagged(
    tmp_path: Path,
) -> None:
    scene = _write_physics_scene(tmp_path)
    nsteps = 250
    ctrl = np.zeros((1, 1), dtype=np.float64)

    default_runtime = create_runtime(DrakeBatchConfig(str(scene), 1, 0.002, nthread=1))
    try:
        default_report = default_runtime.effective_physics()
        default_state = default_runtime.step(ctrl, nsteps)["state"]
    finally:
        default_runtime.close()

    physics = DrakePhysicsConfig(
        contact_model="hydroelastic_with_fallback",
        discrete_contact_approximation="lagged",
        geometry_materials=_materials(),
    )
    hydro_runtime = create_runtime(
        DrakeBatchConfig(str(scene), 1, 0.002, nthread=1, physics=physics)
    )
    try:
        hydro_report = hydro_runtime.effective_physics()
        hydro_state = hydro_runtime.step(ctrl, nsteps)["state"]
    finally:
        hydro_runtime.close()

    assert default_report.contact_model == "point"
    assert default_report.discrete_contact_approximation == "sap"
    assert default_report.geometry_materials == {}
    assert hydro_report.contact_model == "hydroelastic_with_fallback"
    assert hydro_report.discrete_contact_approximation == "lagged"
    # Native readback of two requested material values.
    assert hydro_report.geometry_materials["ball"].hydroelastic_modulus == 1.0e5
    assert hydro_report.geometry_materials["ball"].resolution_hint == 0.01
    assert hydro_report.geometry_materials["ball"].hunt_crossley_dissipation == 50.0
    assert hydro_report.geometry_materials["floor"].hydroelastic_type == "rigid"
    assert hydro_report.drake_version is None or hydro_report.drake_version

    # The ball starts 0.95 m above the floor and drops onto it; point/SAP and
    # hydroelastic/lagged contact produce distinguishable trajectories.
    ball_z_index = 1 + 2  # time, then free-joint qpos starts at x, y, z.
    assert np.isfinite(default_state).all()
    assert np.isfinite(hydro_state).all()
    assert default_state[0, ball_z_index] > 0.04
    assert hydro_state[0, ball_z_index] > 0.04
    assert not np.allclose(default_state, hydro_state)


@requires_native
def test_effective_physics_gravcomp_and_frictionloss_approximation(tmp_path: Path) -> None:
    scene = _write_physics_scene(
        tmp_path, gravcomp=' gravcomp="1"', frictionloss=' frictionloss="0.5"'
    )
    ctrl = np.full((1, 1), 0.1, dtype=np.float64)

    default_runtime = create_runtime(DrakeBatchConfig(str(scene), 1, 0.002, nthread=1))
    try:
        default_report = default_runtime.effective_physics()
        default_state = default_runtime.step(ctrl, 250)["state"]
    finally:
        default_runtime.close()

    physics = DrakePhysicsConfig(
        frictionloss_regularization_speed=1.0e-6,
        preserve_gravity_compensation=True,
    )
    preserved_runtime = create_runtime(
        DrakeBatchConfig(str(scene), 1, 0.002, nthread=1, physics=physics)
    )
    try:
        preserved_report = preserved_runtime.effective_physics()
        preserved_state = preserved_runtime.step(ctrl, 250)["state"]
    finally:
        preserved_runtime.close()

    # Default behavior keeps ignoring both fields; the opt-in report marks the
    # documented approximations instead.
    assert default_report.field_status["joint.frictionloss"] == "ignored"
    assert default_report.field_status["body.gravcomp"] == "ignored"
    assert default_report.ignored_mjcf_fields == ("joint.frictionloss", "body.gravcomp")
    assert preserved_report.field_status["joint.frictionloss"] == "approximated"
    assert preserved_report.field_status["body.gravcomp"] == "approximated"
    assert preserved_report.ignored_mjcf_fields == ()

    # gravcomp=1 exactly compensates the 1 kg base body; only the 0.01 kg link
    # weight remains, so the base barely sinks compared to free fall.
    base_z_index = 1 + 2
    default_drop = 1.0 - default_state[0, base_z_index]
    preserved_drop = 1.0 - preserved_state[0, base_z_index]
    assert default_drop > 0.1
    assert preserved_drop < 0.1 * default_drop

    # The bounded dissipative brake (bound 0.5 N.m, regularized at 1e-6 rad/s)
    # nearly holds the hinge against a 0.1 N.m motor torque, while the default
    # runtime (frictionloss ignored) spins up freely.
    hinge_vel_index = 1 + 8 + 6
    assert abs(preserved_state[0, hinge_vel_index]) < 1.0e-3
    assert abs(default_state[0, hinge_vel_index]) > 0.1


@requires_native
def test_strict_hydroelastic_missing_material_fails_loudly(tmp_path: Path) -> None:
    scene = _write_physics_scene(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        create_runtime(
            DrakeBatchConfig(
                str(scene),
                1,
                0.002,
                nthread=1,
                physics=DrakePhysicsConfig(
                    contact_model="hydroelastic", geometry_materials=_materials()
                ),
            )
        )
