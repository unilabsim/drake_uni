from __future__ import annotations

import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from drake_uni.batch_env import DrakeEnvPool, batch_available, batch_import_error

from .mjcf_model_parser import (
    JOINT_KIND_HINGE,
    JOINT_KIND_SLIDE,
    ROOT_QPOS_DIM,
    ROOT_QVEL_DIM,
    DrakeCompatibleMjcf,
    DrakeMjcfModelContract,
    materialize_drake_compatible_mjcf,
    parse_mjcf_model_contract,
    read_keyframe_qpos,
    sensor_frames_as_pool_inputs,
)
from .types import (
    NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION,
    PHYSICS_FIELD_STATUS_APPLIED,
    PHYSICS_FIELD_STATUS_APPROXIMATED,
    PHYSICS_FIELD_STATUS_IGNORED,
    DrakeBatchConfig,
    DrakeEffectivePhysics,
    DrakeGeometryMaterial,
    DrakeModelInfo,
    DrakeNativeModelProperties,
    DrakePhysicsConfig,
    DrakeRuntimeDiagnostics,
)


class DrakeBatchRuntime:
    def __init__(self, config: DrakeBatchConfig) -> None:
        if int(config.num_envs) < 1:
            raise ValueError(f"DrakeBatchRuntime requires num_envs >= 1, got {config.num_envs}")
        if DrakeEnvPool is None or not bool(batch_available()):
            detail = batch_import_error()
            message = "DrakeEnvPool batch extension has not been built."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail

        self._config = config
        self._num_envs = int(config.num_envs)
        self._sim_dt = float(config.sim_dt)
        self._model_file = str(Path(config.model_file).expanduser())
        self._drake_model: DrakeCompatibleMjcf = materialize_drake_compatible_mjcf(
            self._model_file
        )
        self._model_contract = parse_mjcf_model_contract(self._drake_model.model_file)
        self._nthread = _resolve_nthread(self._num_envs, int(config.nthread))
        self._physics_payload = _build_physics_payload(
            config.physics, self._model_contract, self._sim_dt
        )
        (
            sensor_frame_body_indices,
            sensor_frame_offsets,
            sensor_frame_ref_body_indices,
            sensor_frame_ref_offsets,
        ) = sensor_frames_as_pool_inputs(self._model_contract)
        self._pool = DrakeEnvPool(
            self._drake_model.model_file,
            self._num_envs,
            self._sim_dt,
            self._model_contract.ctrl_limits,
            self._model_contract.torque_limits,
            self._model_contract.actuator_kind,
            self._model_contract.actuator_gear,
            self._model_contract.actuator_stiffness,
            self._model_contract.actuator_damping,
            self._model_contract.actuator_gainprm,
            self._model_contract.actuator_biasprm,
            self._model_contract.joint_layout_kind,
            self._model_contract.joint_layout_qpos_adr,
            self._model_contract.joint_layout_qvel_adr,
            self._model_contract.joint_layout_qpos_dim,
            self._model_contract.joint_layout_qvel_dim,
            self._model_contract.joint_layout_names,
            self._model_contract.joint_layout_body_names,
            self._model_contract.collision_filter_geom_names1,
            self._model_contract.collision_filter_geom_names2,
            sensor_frame_body_indices,
            sensor_frame_offsets,
            sensor_frame_ref_body_indices,
            sensor_frame_ref_offsets,
            self._model_contract.sensor_type,
            self._model_contract.sensor_index,
            self._model_contract.sensor_adr,
            self._model_contract.sensor_dim,
            self._model_contract.nsensordata,
            self._nthread,
            self._physics_payload,
        )

        nq = int(self._pool.num_positions)
        nv = int(self._pool.num_velocities)
        if nq <= ROOT_QPOS_DIM:
            raise RuntimeError(f"DrakeEnvPool batch runtime returned invalid nq={nq}")
        if nv <= ROOT_QVEL_DIM:
            raise RuntimeError(f"DrakeEnvPool batch runtime returned invalid nv={nv}")
        home_qpos = read_keyframe_qpos(self._model_file, "home")
        if home_qpos is None:
            default_state = np.asarray(self._pool.default_state(), dtype=np.float64).reshape(-1)
            if default_state.shape != (int(self._pool.state_dim),):
                raise RuntimeError(
                    "DrakeEnvPool default_state must have shape "
                    f"({int(self._pool.state_dim)},), got {default_state.shape}"
                )
            self._home_qpos = default_state[1 : 1 + nq].copy()
            self._home_qvel = default_state[1 + nq : 1 + nq + nv].copy()
        else:
            self._home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
            if self._home_qpos.shape != (nq,):
                raise ValueError(
                    f"DrakeBatchRuntime keyframe 'home' qpos must have shape ({nq},), "
                    f"got {self._home_qpos.shape}"
                )
            self._home_qvel = np.zeros(nv, dtype=np.float64)
        self._model_info = DrakeModelInfo(
            nq=nq,
            nv=nv,
            nu=int(self._pool.control_dim),
            home_qpos=self._home_qpos.copy(),
            home_qvel=self._home_qvel.copy(),
            ctrl_limits=self._model_contract.ctrl_limits.copy(),
            torque_limits=self._model_contract.torque_limits.copy(),
            actuator_stiffness=self._model_contract.actuator_stiffness.copy(),
            actuator_damping=self._model_contract.actuator_damping.copy(),
            actuator_qpos_adr=self._model_contract.actuator_qpos_adr.copy(),
            actuator_qvel_adr=self._model_contract.actuator_qvel_adr.copy(),
            joint_ranges=self._model_contract.joint_ranges.copy(),
            num_bodies=int(self._pool.num_bodies),
            site_names=self._model_contract.site_names,
            joint_names=self._model_contract.joint_layout_names,
            joint_qpos_adr=self._model_contract.joint_layout_qpos_adr.copy(),
            joint_qvel_adr=self._model_contract.joint_layout_qvel_adr.copy(),
            joint_qpos_dim=self._model_contract.joint_layout_qpos_dim.copy(),
            joint_qvel_dim=self._model_contract.joint_layout_qvel_dim.copy(),
            sensor_names=self._model_contract.sensor_names,
            sensor_adr=self._model_contract.sensor_adr.copy(),
            sensor_dim=self._model_contract.sensor_dim.copy(),
            nsensordata=self._model_contract.nsensordata,
            actuator_names=self._model_contract.actuator_names,
            joint_body_names=self._model_contract.joint_layout_body_names,
        )
        self._native_model_properties: DrakeNativeModelProperties | None = None
        self._effective_physics: DrakeEffectivePhysics | None = None
        self._physics_state = np.zeros((self._num_envs, int(self._pool.state_dim)), dtype=np.float64)
        self._sensor_data = np.zeros(
            (self._num_envs, self._model_info.nsensordata),
            dtype=np.float64,
        )
        qpos = np.broadcast_to(self._home_qpos, (self._num_envs, self._model_info.nq)).copy()
        qvel = np.zeros((self._num_envs, self._model_info.nv), dtype=np.float64)
        self.reset(np.arange(self._num_envs, dtype=np.int32), qpos, qvel)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nthread(self) -> int:
        return self._nthread

    @property
    def model_file(self) -> str:
        return self._model_file

    def model_info(self) -> DrakeModelInfo:
        info = self._model_info
        return DrakeModelInfo(
            nq=info.nq,
            nv=info.nv,
            nu=info.nu,
            home_qpos=info.home_qpos.copy(),
            home_qvel=info.home_qvel.copy(),
            ctrl_limits=info.ctrl_limits.copy(),
            torque_limits=info.torque_limits.copy(),
            actuator_stiffness=info.actuator_stiffness.copy(),
            actuator_damping=info.actuator_damping.copy(),
            actuator_qpos_adr=info.actuator_qpos_adr.copy(),
            actuator_qvel_adr=info.actuator_qvel_adr.copy(),
            joint_ranges=info.joint_ranges.copy(),
            num_bodies=info.num_bodies,
            site_names=info.site_names,
            joint_names=info.joint_names,
            joint_qpos_adr=info.joint_qpos_adr.copy(),
            joint_qvel_adr=info.joint_qvel_adr.copy(),
            joint_qpos_dim=info.joint_qpos_dim.copy(),
            joint_qvel_dim=info.joint_qvel_dim.copy(),
            sensor_names=info.sensor_names,
            sensor_adr=info.sensor_adr.copy(),
            sensor_dim=info.sensor_dim.copy(),
            nsensordata=info.nsensordata,
            actuator_names=info.actuator_names,
            joint_body_names=info.joint_body_names,
        )

    def native_model_properties(self) -> DrakeNativeModelProperties:
        """Return a validated cold-path native model identity snapshot."""

        if self._native_model_properties is None:
            read_native = getattr(self._pool, "native_model_properties", None)
            if not callable(read_native):
                raise RuntimeError("DrakeEnvPool build does not expose native_model_properties")
            raw = read_native()
            if not isinstance(raw, dict):
                raise RuntimeError("DrakeEnvPool native_model_properties must return a dict")
            body_names = _require_output_strings(raw, "body_names")
            geometry_names = _require_output_strings(raw, "geometry_names")
            geometry_types = _require_output_strings(raw, "geometry_types")
            geometry_collision = tuple(
                _require_output_bool(value)
                for value in _require_output_key(raw, "geometry_collision", len(geometry_names))
            )
            self._native_model_properties = DrakeNativeModelProperties(
                contract_version=NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION,
                body_names=body_names,
                body_masses=_require_output_array(
                    raw,
                    "body_masses",
                    (len(body_names),),
                    "DrakeEnvPool native_model_properties",
                ),
                body_coms=_require_output_array(
                    raw,
                    "body_coms",
                    (len(body_names), 3),
                    "DrakeEnvPool native_model_properties",
                ),
                body_inertias=_require_output_array(
                    raw,
                    "body_inertias",
                    (len(body_names), 3, 3),
                    "DrakeEnvPool native_model_properties",
                ),
                geometry_names=geometry_names,
                geometry_body_indices=_require_output_integer_array(
                    raw,
                    "geometry_body_indices",
                    (len(geometry_names),),
                    "DrakeEnvPool native_model_properties",
                ),
                geometry_types=geometry_types,
                geometry_collision=geometry_collision,
                geometry_parameters=_require_output_array(
                    raw,
                    "geometry_parameters",
                    (len(geometry_names), 3),
                    "DrakeEnvPool native_model_properties",
                ),
            )
        return self._native_model_properties

    def effective_physics(self) -> DrakeEffectivePhysics:
        """Return the validated native effective-physics report.

        The contact model, discrete approximation, penetration allowance and
        per-geometry material values are read back from the finalized Drake
        plant and SceneGraph. The per-field status mapping and the ignored
        MJCF field list are derived from the requested configuration and the
        cold-path MJCF scan; ``drake_version`` comes from the native build.
        """

        if self._effective_physics is None:
            read_native = getattr(self._pool, "effective_physics", None)
            if not callable(read_native):
                raise RuntimeError("DrakeEnvPool build does not expose effective_physics")
            raw = read_native()
            if not isinstance(raw, dict):
                raise RuntimeError("DrakeEnvPool effective_physics must return a dict")
            requested = self._config.physics or DrakePhysicsConfig()
            contact_model = _require_physics_string(raw, "contact_model")
            if contact_model != requested.contact_model:
                raise RuntimeError(
                    f"DrakeEnvPool applied contact_model={contact_model!r} but "
                    f"{requested.contact_model!r} was requested"
                )
            approximation = _require_physics_string(raw, "discrete_contact_approximation")
            if approximation != requested.discrete_contact_approximation:
                raise RuntimeError(
                    "DrakeEnvPool applied "
                    f"discrete_contact_approximation={approximation!r} but "
                    f"{requested.discrete_contact_approximation!r} was requested"
                )
            raw_materials = raw.get("geometry_materials")
            if not isinstance(raw_materials, dict):
                raise RuntimeError("DrakeEnvPool effective_physics geometry_materials must be a dict")
            materials: dict[str, DrakeGeometryMaterial] = {}
            for name in requested.geometry_materials:
                if name not in raw_materials:
                    raise RuntimeError(
                        "DrakeEnvPool effective_physics did not report requested "
                        f"geometry material {name!r}"
                    )
                materials[name] = _material_from_native(name, raw_materials[name])
            version = raw.get("drake_version")
            if version is not None and type(version) is not str:
                raise RuntimeError("DrakeEnvPool effective_physics drake_version must be a string")
            ignored_physics = self._model_contract.ignored_physics
            frictionloss_approximated = (
                requested.frictionloss_regularization_speed is not None
                and bool(ignored_physics.joint_frictionloss)
            )
            gravcomp_preserved = (
                requested.preserve_gravity_compensation and bool(ignored_physics.body_gravcomp)
            )
            field_status: dict[str, str] = {
                "contact_model": PHYSICS_FIELD_STATUS_APPLIED,
                "discrete_contact_approximation": PHYSICS_FIELD_STATUS_APPLIED,
                "penetration_allowance": PHYSICS_FIELD_STATUS_APPLIED,
            }
            for name in materials:
                field_status[f"geometry_materials.{name}"] = PHYSICS_FIELD_STATUS_APPLIED
            field_status["joint.frictionloss"] = _ignored_or_approximated(
                ignored_physics.joint_frictionloss, frictionloss_approximated
            )
            field_status["body.gravcomp"] = _ignored_or_approximated(
                ignored_physics.body_gravcomp, gravcomp_preserved
            )
            field_status["geom.torsional_friction"] = _ignored_or_absent(
                ignored_physics.torsional_friction_geoms
            )
            field_status["geom.rolling_friction"] = _ignored_or_absent(
                ignored_physics.rolling_friction_geoms
            )
            field_status["geom.condim"] = _ignored_or_absent(
                ignored_physics.unsupported_condim_geoms
            )
            self._effective_physics = DrakeEffectivePhysics(
                contact_model=contact_model,
                discrete_contact_approximation=approximation,
                penetration_allowance=_require_physics_number(raw, "penetration_allowance"),
                geometry_materials=materials,
                field_status=field_status,
                ignored_mjcf_fields=ignored_physics.ignored_field_names(
                    frictionloss_approximated=frictionloss_approximated,
                    gravcomp_preserved=gravcomp_preserved,
                ),
                drake_version=version,
            )
        return self._effective_physics

    def keyframe_qpos(self, name: str) -> np.ndarray:
        qpos = read_keyframe_qpos(self._model_file, str(name))
        if qpos is None:
            raise KeyError(f"Drake MJCF model does not contain keyframe {name!r}")
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (self._model_info.nq,):
            raise ValueError(
                f"Drake MJCF keyframe {name!r} qpos must have shape "
                f"({self._model_info.nq},), got {values.shape}"
            )
        return values.copy()

    def reset(self, env_ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, Any]:
        indices = np.asarray(env_ids, dtype=np.int32)
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError(f"env_ids must be one-dimensional, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self._num_envs):
            raise IndexError(f"env_ids must be in [0, {self._num_envs - 1}], got {indices.tolist()}")
        if qpos_rows.shape != (indices.size, self._model_info.nq):
            raise ValueError(f"qpos must have shape ({indices.size}, {self._model_info.nq})")
        if qvel_rows.shape != (indices.size, self._model_info.nv):
            raise ValueError(f"qvel must have shape ({indices.size}, {self._model_info.nv})")
        output = self._pool.reset(indices, self._pack_state_rows(qpos_rows, qvel_rows), True)
        state_rows = _require_output_array(
            output,
            "state",
            (indices.size, int(self._pool.state_dim)),
            "DrakeEnvPool reset",
        )
        sensor_rows = _require_output_array(
            output,
            "sensor_data",
            (indices.size, self._model_info.nsensordata),
            "DrakeEnvPool reset",
        )
        self._physics_state[indices] = state_rows
        self._sensor_data[indices] = sensor_rows
        return {
            "env_ids": indices.copy(),
            "state": state_rows.copy(),
            "sensor_data": sensor_rows.copy(),
        }

    def step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
        body_forces: np.ndarray | None = None,
    ) -> dict[str, Any]:
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self._model_info.nu):
            raise ValueError(
                f"ctrl must have shape ({self._num_envs}, {self._model_info.nu}), got {values.shape}"
            )
        forces = None if body_forces is None else np.asarray(body_forces, dtype=np.float64)
        expected_force_shape = (self._num_envs, self._model_info.num_bodies, 3)
        if forces is not None and forces.shape != expected_force_shape:
            raise ValueError(f"body_forces must have shape {expected_force_shape}, got {forces.shape}")
        output = self._pool.step(self._physics_state, int(nsteps), values, forces, True)
        self._apply_output(output)
        self._apply_sensor_data(output)
        return {
            "state": self.physics_state(),
            "sensor_data": self.sensor_data(),
            "timing": dict(output.get("timing", {})),
        }

    def physics_state(self) -> np.ndarray:
        return self._physics_state.copy()

    def sensor_data(self) -> np.ndarray:
        return self._sensor_data.copy()

    def compute_body_state(
        self,
        body_ids: np.ndarray | list[int] | tuple[int, ...],
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(body_ids, dtype=np.int32)
        output = self._pool.compute_body_state(self._physics_state, ids)
        return {name: np.asarray(value, dtype=np.float64).copy() for name, value in output.items()}

    def body_ids(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self._model_contract.body_index(str(name)) for name in names],
            dtype=np.int32,
        )

    def diagnostics(self) -> DrakeRuntimeDiagnostics:
        detail = batch_import_error()
        return DrakeRuntimeDiagnostics(
            mode="batch",
            available=DrakeEnvPool is not None and bool(batch_available()),
            batch_available=bool(batch_available()),
            batch_import_error=None if detail is None else str(detail),
            pydrake_loaded=_pydrake_loaded(),
            nthread=self._nthread,
            workspace_count=int(getattr(self._pool, "workspace_count", 0)),
            num_filtered_geometries=int(getattr(self._pool, "num_filtered_geometries", 0)),
        )

    def render_capabilities(self) -> dict[str, bool]:
        return {
            "interactive": False,
            "video_capture": False,
            "physics_state_playback": True,
        }

    def close(self) -> None:
        self._drake_model.close()

    def _pack_state_rows(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        state = np.zeros((qpos.shape[0], int(self._pool.state_dim)), dtype=np.float64)
        state[:, 1 : 1 + self._model_info.nq] = qpos
        state[:, 1 + self._model_info.nq :] = qvel
        return state

    def _apply_output(self, output: dict[str, Any]) -> None:
        self._physics_state = np.asarray(output["state"], dtype=np.float64).copy()

    def _apply_sensor_data(self, output: dict[str, Any]) -> None:
        if "sensor_data" not in output:
            raise RuntimeError("DrakeEnvPool output did not include sensor_data")
        sensor_data = np.asarray(output["sensor_data"], dtype=np.float64)
        expected_shape = (self._num_envs, self._model_info.nsensordata)
        if sensor_data.shape != expected_shape:
            raise RuntimeError(
                f"DrakeEnvPool sensor_data must have shape {expected_shape}, got {sensor_data.shape}"
            )
        self._sensor_data = sensor_data.copy()


def _build_physics_payload(
    physics: DrakePhysicsConfig | None,
    contract: DrakeMjcfModelContract,
    sim_dt: float,
) -> dict[str, Any] | None:
    """Validate the physics configuration against the model contract and
    serialize it for the native pool. ``None`` reproduces the historical
    native defaults (point contact, SAP, penetration_allowance=1e-4)."""

    if physics is None:
        return None
    if sim_dt <= 0.0 and physics.discrete_contact_approximation != "sap":
        raise ValueError(
            "discrete_contact_approximation requires a discrete plant (sim_dt > 0); "
            f"got sim_dt={sim_dt}"
        )
    collision_names = {geom.name for geom in contract.collision_geoms}
    unknown = sorted(set(physics.geometry_materials) - collision_names)
    if unknown:
        raise ValueError(
            f"geometry_materials refer to unknown collision geometries: {unknown}"
        )
    if physics.contact_model == "hydroelastic":
        missing = sorted(collision_names - set(physics.geometry_materials))
        if missing:
            raise ValueError(
                "contact_model='hydroelastic' is strict and requires a geometry_materials "
                f"assignment for every collision geometry; missing: {missing}. Use "
                "'hydroelastic_with_fallback' for the explicit point-contact fallback mode."
            )
    frictionloss = np.zeros(len(contract.joint_layouts), dtype=np.float64)
    if physics.frictionloss_regularization_speed is not None:
        unsupported = [
            layout.name or layout.body_name
            for layout in contract.joint_layouts
            if layout.frictionloss != 0.0
            and layout.kind not in {JOINT_KIND_SLIDE, JOINT_KIND_HINGE}
        ]
        if unsupported:
            raise ValueError(
                "the frictionloss approximation only supports single-dof slide/hinge "
                f"joints; nonzero frictionloss on unsupported joints: {unsupported}"
            )
        frictionloss = contract.joint_frictionloss.copy()
    return {
        "contact_model": physics.contact_model,
        "discrete_contact_approximation": physics.discrete_contact_approximation,
        "penetration_allowance": physics.penetration_allowance,
        "geometry_materials": {
            name: {
                "hydroelastic_type": material.hydroelastic_type,
                "hydroelastic_modulus": material.hydroelastic_modulus,
                "resolution_hint": material.resolution_hint,
                "hunt_crossley_dissipation": material.hunt_crossley_dissipation,
            }
            for name, material in physics.geometry_materials.items()
        },
        "frictionloss_regularization_speed": physics.frictionloss_regularization_speed,
        "joint_frictionloss": frictionloss,
        "preserve_gravity_compensation": physics.preserve_gravity_compensation,
        "body_gravcomp": (
            dict(contract.ignored_physics.body_gravcomp)
            if physics.preserve_gravity_compensation
            else {}
        ),
    }


def _require_physics_string(output: dict[str, Any], key: str) -> str:
    value = output.get(key)
    if type(value) is not str:
        raise RuntimeError(f"DrakeEnvPool effective_physics {key} must be a string")
    return value


def _require_physics_number(output: dict[str, Any], key: str) -> float:
    value = output.get(key)
    if type(value) is bool or not isinstance(value, int | float):
        raise RuntimeError(f"DrakeEnvPool effective_physics {key} must be a real number")
    return float(value)


def _material_from_native(name: str, value: Any) -> DrakeGeometryMaterial:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"DrakeEnvPool effective_physics geometry material {name!r} must be a dict"
        )
    try:
        return DrakeGeometryMaterial(
            hydroelastic_type=value["hydroelastic_type"],
            hydroelastic_modulus=value["hydroelastic_modulus"],
            resolution_hint=value["resolution_hint"],
            hunt_crossley_dissipation=value["hunt_crossley_dissipation"],
        )
    except KeyError as exc:
        raise RuntimeError(
            f"DrakeEnvPool effective_physics geometry material {name!r} is missing {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"DrakeEnvPool effective_physics geometry material {name!r} is invalid: {exc}"
        ) from exc


def _ignored_or_approximated(values: Any, approximated: bool) -> str:
    if not values:
        return PHYSICS_FIELD_STATUS_APPLIED
    if approximated:
        return PHYSICS_FIELD_STATUS_APPROXIMATED
    return PHYSICS_FIELD_STATUS_IGNORED


def _ignored_or_absent(values: tuple[str, ...]) -> str:
    return PHYSICS_FIELD_STATUS_IGNORED if values else PHYSICS_FIELD_STATUS_APPLIED


def _resolve_nthread(num_envs: int, requested: int) -> int:
    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _require_output_array(
    output: dict[str, Any],
    key: str,
    shape: tuple[int, ...],
    source: str,
) -> np.ndarray:
    values = np.asarray(output[key])
    if values.dtype != np.dtype(np.float64):
        raise RuntimeError(f"{source} {key} must contain float64 values")
    if values.shape != shape:
        raise RuntimeError(f"{source} {key} must have shape {shape}, got {values.shape}")
    return values.copy()


def _require_output_key(output: dict[str, Any], key: str, length: int | None) -> tuple[Any, ...]:
    try:
        values = tuple(output[key])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"DrakeEnvPool native_model_properties {key} must be a sequence"
        ) from exc
    if length is not None and len(values) != length:
        raise RuntimeError(
            "DrakeEnvPool native_model_properties "
            f"{key} must have length {length}, got {len(values)}"
        )
    return values


def _require_output_strings(output: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _require_output_key(output, key, None)
    if any(not isinstance(value, str) for value in values):
        raise RuntimeError(f"DrakeEnvPool native_model_properties {key} must contain strings")
    return values


def _require_output_bool(value: Any) -> bool:
    if type(value) is not bool:
        raise RuntimeError("DrakeEnvPool native_model_properties collision flags must be bools")
    return value


def _require_output_integer_array(
    output: dict[str, Any], key: str, shape: tuple[int, ...], source: str
) -> np.ndarray:
    values = np.asarray(output[key])
    if values.dtype.kind not in "iu":
        raise RuntimeError(f"{source} {key} must contain integers")
    if values.shape != shape:
        raise RuntimeError(f"{source} {key} must have shape {shape}, got {values.shape}")
    return values.astype(np.int32, copy=True)


def _pydrake_loaded() -> bool:
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)
