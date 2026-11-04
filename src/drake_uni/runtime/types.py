from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

NATIVE_MODEL_PROPERTIES_CONTRACT_VERSION = 1

CONTACT_MODEL_POINT = "point"
CONTACT_MODEL_HYDROELASTIC = "hydroelastic"
CONTACT_MODEL_HYDROELASTIC_WITH_FALLBACK = "hydroelastic_with_fallback"
CONTACT_MODELS = frozenset(
    {
        CONTACT_MODEL_POINT,
        CONTACT_MODEL_HYDROELASTIC,
        CONTACT_MODEL_HYDROELASTIC_WITH_FALLBACK,
    }
)

DISCRETE_CONTACT_APPROXIMATIONS = frozenset({"sap", "lagged", "similar"})

HYDROELASTIC_TYPES = frozenset({"rigid", "compliant"})

PHYSICS_FIELD_STATUS_APPLIED = "applied"
PHYSICS_FIELD_STATUS_APPROXIMATED = "approximated"
PHYSICS_FIELD_STATUS_IGNORED = "ignored"
PHYSICS_FIELD_STATUS_UNSUPPORTED = "unsupported"
PHYSICS_FIELD_STATUSES = frozenset(
    {
        PHYSICS_FIELD_STATUS_APPLIED,
        PHYSICS_FIELD_STATUS_APPROXIMATED,
        PHYSICS_FIELD_STATUS_IGNORED,
        PHYSICS_FIELD_STATUS_UNSUPPORTED,
    }
)


@dataclass(frozen=True)
class DrakeGeometryMaterial:
    """Per-geometry hydroelastic contact material assignment.

    ``hydroelastic_type`` selects the hydroelastic representation:
    ``"rigid"`` keeps the geometry rigid in hydroelastic contact, while
    ``"compliant"`` requires an explicit ``hydroelastic_modulus`` (Pa) and
    ``resolution_hint`` (m). ``hunt_crossley_dissipation`` (s/m) is optional
    and applies to both representations.
    """

    hydroelastic_type: str
    hydroelastic_modulus: float | None = None
    resolution_hint: float | None = None
    hunt_crossley_dissipation: float | None = None

    def __post_init__(self) -> None:
        if type(self.hydroelastic_type) is not str:
            raise TypeError("hydroelastic_type must be a string")
        if self.hydroelastic_type not in HYDROELASTIC_TYPES:
            raise ValueError(
                f"unsupported hydroelastic_type {self.hydroelastic_type!r}; "
                f"expected one of {sorted(HYDROELASTIC_TYPES)}"
            )
        if self.hydroelastic_type == "compliant":
            object.__setattr__(
                self,
                "hydroelastic_modulus",
                _require_positive_finite(
                    self.hydroelastic_modulus, "hydroelastic_modulus", required=True
                ),
            )
            object.__setattr__(
                self,
                "resolution_hint",
                _require_positive_finite(self.resolution_hint, "resolution_hint", required=True),
            )
        else:
            if self.hydroelastic_modulus is not None or self.resolution_hint is not None:
                raise ValueError(
                    "rigid geometry materials do not accept hydroelastic_modulus "
                    "or resolution_hint"
                )
        object.__setattr__(
            self,
            "hunt_crossley_dissipation",
            _require_nonnegative_finite(
                self.hunt_crossley_dissipation, "hunt_crossley_dissipation"
            ),
        )


@dataclass(frozen=True)
class DrakePhysicsConfig:
    """Engine-level contact and force-model configuration for the batch runtime.

    Defaults reproduce the historical DrakeUni behavior: point contact with the
    SAP discrete approximation and a 1e-4 m penetration allowance, while MJCF
    ``frictionloss`` and ``gravcomp`` remain ignored exactly as Drake's parser
    leaves them.

    ``contact_model="hydroelastic"`` is strict: every collision geometry must
    carry a ``geometry_materials`` assignment, otherwise construction fails
    loudly instead of silently falling back to point contact. Use
    ``"hydroelastic_with_fallback"`` for the explicit opt-in fallback mode
    where unassigned geometries keep point contact.

    ``frictionloss_regularization_speed`` opts into a documented approximation
    for nonzero MJCF joint ``frictionloss``: each affected degree of freedom
    receives a bounded dissipative torque ``-frictionloss * tanh(v / speed)``
    (the joint's dry-friction bound, smoothly regularized at the given speed).
    ``preserve_gravity_compensation`` opts into a documented approximation for
    nonzero MJCF body ``gravcomp``: each affected body receives a constant
    force ``-gravcomp * mass * gravity`` applied at its center of mass.
    """

    contact_model: str = CONTACT_MODEL_POINT
    discrete_contact_approximation: str = "sap"
    penetration_allowance: float = 1.0e-4
    geometry_materials: Mapping[str, DrakeGeometryMaterial] = field(default_factory=dict)
    frictionloss_regularization_speed: float | None = None
    preserve_gravity_compensation: bool = False

    def __post_init__(self) -> None:
        if type(self.contact_model) is not str:
            raise TypeError("contact_model must be a string")
        if self.contact_model not in CONTACT_MODELS:
            raise ValueError(
                f"unsupported contact_model {self.contact_model!r}; "
                f"expected one of {sorted(CONTACT_MODELS)}"
            )
        if type(self.discrete_contact_approximation) is not str:
            raise TypeError("discrete_contact_approximation must be a string")
        if self.discrete_contact_approximation not in DISCRETE_CONTACT_APPROXIMATIONS:
            raise ValueError(
                f"unsupported discrete_contact_approximation "
                f"{self.discrete_contact_approximation!r}; expected one of "
                f"{sorted(DISCRETE_CONTACT_APPROXIMATIONS)}"
            )
        object.__setattr__(
            self,
            "penetration_allowance",
            _require_positive_finite(
                self.penetration_allowance, "penetration_allowance", required=True
            ),
        )
        if not isinstance(self.geometry_materials, Mapping):
            raise TypeError("geometry_materials must be a mapping of name to DrakeGeometryMaterial")
        materials: dict[str, DrakeGeometryMaterial] = {}
        for name, material in self.geometry_materials.items():
            if type(name) is not str or not name:
                raise ValueError("geometry_materials names must be non-empty strings")
            if type(material) is not DrakeGeometryMaterial:
                raise TypeError(
                    f"geometry_materials[{name!r}] must be a DrakeGeometryMaterial, "
                    f"got {type(material).__name__}"
                )
            materials[name] = material
        object.__setattr__(self, "geometry_materials", materials)
        object.__setattr__(
            self,
            "frictionloss_regularization_speed",
            _require_positive_finite(
                self.frictionloss_regularization_speed,
                "frictionloss_regularization_speed",
            ),
        )
        if type(self.preserve_gravity_compensation) is not bool:
            raise TypeError("preserve_gravity_compensation must be a bool")
        if self.contact_model == CONTACT_MODEL_POINT and materials:
            raise ValueError(
                "geometry_materials require a hydroelastic contact model; "
                "they cannot take effect under contact_model='point'"
            )
        if self.contact_model == CONTACT_MODEL_HYDROELASTIC and not materials:
            raise ValueError(
                "contact_model='hydroelastic' is strict: it requires geometry_materials "
                "assignments for every collision geometry. Use "
                "'hydroelastic_with_fallback' for the explicit point-contact fallback mode."
            )


@dataclass(frozen=True)
class DrakeEffectivePhysics:
    """Native effective-physics report returned by the batch runtime.

    ``contact_model``, ``discrete_contact_approximation`` and
    ``penetration_allowance`` are read back from the finalized Drake plant.
    ``geometry_materials`` carries the per-geometry material values read back
    from SceneGraph proximity properties for every requested geometry.
    ``field_status`` marks each physics-relevant field as ``"applied"``,
    ``"approximated"``, ``"ignored"`` or ``"unsupported"``;
    ``ignored_mjcf_fields`` lists the MJCF physics fields Drake ignored in the
    effective configuration. ``drake_version`` is the native Drake version
    string, or None when the local Drake build does not stamp one.
    """

    contact_model: str
    discrete_contact_approximation: str
    penetration_allowance: float
    geometry_materials: Mapping[str, DrakeGeometryMaterial]
    field_status: Mapping[str, str]
    ignored_mjcf_fields: tuple[str, ...]
    drake_version: str | None

    def __post_init__(self) -> None:
        if type(self.contact_model) is not str or self.contact_model not in CONTACT_MODELS:
            raise ValueError(f"effective contact_model must be one of {sorted(CONTACT_MODELS)}")
        if (
            type(self.discrete_contact_approximation) is not str
            or self.discrete_contact_approximation not in DISCRETE_CONTACT_APPROXIMATIONS
        ):
            raise ValueError(
                "effective discrete_contact_approximation must be one of "
                f"{sorted(DISCRETE_CONTACT_APPROXIMATIONS)}"
            )
        object.__setattr__(
            self,
            "penetration_allowance",
            _require_positive_finite(
                self.penetration_allowance, "penetration_allowance", required=True
            ),
        )
        if not isinstance(self.geometry_materials, Mapping):
            raise TypeError("effective geometry_materials must be a mapping")
        materials: dict[str, DrakeGeometryMaterial] = {}
        for name, material in self.geometry_materials.items():
            if type(name) is not str or not name:
                raise ValueError("effective geometry_materials names must be non-empty strings")
            if type(material) is not DrakeGeometryMaterial:
                raise TypeError(
                    f"effective geometry_materials[{name!r}] must be a DrakeGeometryMaterial"
                )
            materials[name] = material
        object.__setattr__(self, "geometry_materials", materials)
        if not isinstance(self.field_status, Mapping):
            raise TypeError("field_status must be a mapping of field name to status")
        status: dict[str, str] = {}
        for name, value in self.field_status.items():
            if type(name) is not str or not name:
                raise ValueError("field_status names must be non-empty strings")
            if type(value) is not str or value not in PHYSICS_FIELD_STATUSES:
                raise ValueError(
                    f"field_status[{name!r}] must be one of {sorted(PHYSICS_FIELD_STATUSES)}"
                )
            status[name] = value
        object.__setattr__(self, "field_status", status)
        ignored = tuple(self.ignored_mjcf_fields)
        if any(type(field_name) is not str or not field_name for field_name in ignored):
            raise ValueError("ignored_mjcf_fields must be non-empty strings")
        object.__setattr__(self, "ignored_mjcf_fields", ignored)
        if self.drake_version is not None and (
            type(self.drake_version) is not str or not self.drake_version
        ):
            raise ValueError("drake_version must be None or a non-empty string")


def _require_positive_finite(
    value: float | None, name: str, *, required: bool = False
) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if type(value) is bool or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result


def _require_nonnegative_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if type(value) is bool or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return result


@dataclass(frozen=True)
class DrakeBatchConfig:
    model_file: str
    num_envs: int
    sim_dt: float
    nthread: int = 0
    physics: DrakePhysicsConfig | None = None

    def __post_init__(self) -> None:
        if self.physics is not None and type(self.physics) is not DrakePhysicsConfig:
            raise TypeError(
                "physics must be a DrakePhysicsConfig or None, "
                f"got {type(self.physics).__name__}"
            )


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
