from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class DrakeBatchConfig:
    model_file: str
    num_envs: int
    sim_dt: float
    nthread: int = 0


@dataclass(frozen=True)
class DrakeRuntimeDiagnostics:
    mode: str
    available: bool
    batch_available: bool
    batch_import_error: str | None = None
    pydrake_loaded: bool = False
    nthread: int | None = None
    workspace_count: int | None = None
    num_filtered_geometries: int | None = None


@dataclass(frozen=True)
class DrakeNativeModelProperties:
    """Cold-path native model identity exposed without Drake objects."""

    contract_version: int
    body_names: tuple[str, ...]
    body_masses: np.ndarray
    body_coms: np.ndarray
    body_inertias: np.ndarray
    geometry_names: tuple[str, ...]
    geometry_body_indices: np.ndarray
    geometry_types: tuple[str, ...]
    geometry_collision: tuple[bool, ...]
    geometry_parameters: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.contract_version) is not int
            or self.contract_version != NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION
        ):
            raise ValueError(
                "unsupported Drake native model property contract version "
                f"{self.contract_version!r}"
            )
        body_names = tuple(self.body_names)
        geometry_names = tuple(self.geometry_names)
        geometry_types = tuple(self.geometry_types)
        geometry_collision = tuple(self.geometry_collision)
        if any(type(name) is not str for name in (*body_names, *geometry_names)):
            raise TypeError("native model names must be strings")
        if any(type(geometry_type) is not str for geometry_type in geometry_types):
            raise TypeError("native geometry types must be strings")
        if any(type(collision) is not bool for collision in geometry_collision):
            raise TypeError("native geometry collision flags must be bools")
        if not body_names:
            raise ValueError("native model body names must not be empty")
        if any(not name for name in (*body_names, *geometry_names)):
            raise ValueError("native model names must be non-empty strings")
        if len(set(body_names)) != len(body_names):
            raise ValueError("native model body names must be unique")
        if len(set(geometry_names)) != len(geometry_names):
            raise ValueError("native model geometry names must be unique")
        supported_types = {
            "sphere",
            "box",
            "capsule",
            "cylinder",
            "ellipsoid",
            "half_space",
        }
        if any(value not in supported_types for value in geometry_types):
            unsupported = sorted(set(geometry_types) - supported_types)
            raise ValueError(f"unsupported native geometry types: {unsupported}")
        if not (len(geometry_names) == len(geometry_types) == len(geometry_collision)):
            raise ValueError("native geometry identity fields are not aligned")

        body_count = len(body_names)
        geometry_count = len(geometry_names)
        object.__setattr__(self, "body_names", body_names)
        object.__setattr__(self, "geometry_names", geometry_names)
        object.__setattr__(self, "geometry_types", geometry_types)
        object.__setattr__(self, "geometry_collision", geometry_collision)
        object.__setattr__(
            self,
            "body_masses",
            _read_only_finite_array(self.body_masses, (body_count,), np.float64),
        )
        object.__setattr__(
            self,
            "body_coms",
            _read_only_finite_array(self.body_coms, (body_count, 3), np.float64),
        )
        object.__setattr__(
            self,
            "body_inertias",
            _read_only_finite_array(self.body_inertias, (body_count, 3, 3), np.float64),
        )
        object.__setattr__(
            self,
            "geometry_body_indices",
            _read_only_integer_array(self.geometry_body_indices, (geometry_count,), np.int32),
        )
        if np.any(self.geometry_body_indices < 0) or np.any(
            self.geometry_body_indices >= body_count
        ):
            raise ValueError("native geometry body indices are out of range")
        object.__setattr__(
            self,
            "geometry_parameters",
            _read_only_finite_array(self.geometry_parameters, (geometry_count, 3), np.float64),
        )

        for geometry_type, parameters in zip(geometry_types, self.geometry_parameters, strict=True):
            positive = (
                3
                if geometry_type in {"box", "ellipsoid"}
                else 2
                if geometry_type in {"capsule", "cylinder"}
                else 0
                if geometry_type == "half_space"
                else 1
            )
            expected_zeros = 3 - positive
            if expected_zeros:
                if np.any(parameters[positive:] != 0.0):
                    raise ValueError(
                        f"native {geometry_type} parameters have unused non-zero values"
                    )
            if np.any(parameters[:positive] <= 0.0):
                raise ValueError(f"native {geometry_type} parameters must be positive")


def _read_only_array(value: object, shape: tuple[int, ...], dtype: type[np.generic]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"native model array must have shape {shape}, got {array.shape}")
    result = array.copy()
    result.setflags(write=False)
    return result


def _read_only_integer_array(
    value: object, shape: tuple[int, ...], dtype: type[np.generic]
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError("native model integer arrays must contain integers")
    return _read_only_array(raw, shape, dtype)


def _read_only_finite_array(
    value: object, shape: tuple[int, ...], dtype: type[np.generic]
) -> np.ndarray:
    result = _read_only_array(value, shape, dtype)
    if not np.isfinite(result).all():
        raise ValueError("native model numeric values must be finite")
    return result


@dataclass(frozen=True)
class DrakeModelInfo:
    nq: int
    nv: int
    nu: int
    home_qpos: np.ndarray
    home_qvel: np.ndarray
    ctrl_limits: np.ndarray
    torque_limits: np.ndarray
    actuator_stiffness: np.ndarray
    actuator_damping: np.ndarray
    actuator_qpos_adr: np.ndarray
    actuator_qvel_adr: np.ndarray
    joint_ranges: np.ndarray
    num_bodies: int
    site_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_qpos_adr: np.ndarray
    joint_qvel_adr: np.ndarray
    joint_qpos_dim: np.ndarray
    joint_qvel_dim: np.ndarray
    sensor_names: tuple[str, ...]
    sensor_adr: np.ndarray
    sensor_dim: np.ndarray
    nsensordata: int
    # Names are kept optional for compatibility with runtimes created by
    # older DrakeUni builds; current materialization always populates them.
    actuator_names: tuple[str, ...] = ()
    joint_body_names: tuple[str, ...] = ()
