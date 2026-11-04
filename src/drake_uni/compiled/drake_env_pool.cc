#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/geometry/scene_graph.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/geometry_set.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/joint.h"
#include "drake/multibody/tree/joint_actuator.h"
#include "drake/multibody/tree/model_instance.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

#if defined(__has_include)
#if __has_include("drake/version.h")
#include "drake/version.h"
#define DRAKE_UNI_HAS_DRAKE_VERSION_HEADER 1
#endif
#endif

namespace py = pybind11;

namespace {

constexpr int kRootQposDim = 7;
constexpr int kRootQvelDim = 6;

enum CompactJointKind {
  kFreeJoint = 0,
  kSlideJoint = 1,
  kHingeJoint = 2,
  kBallJoint = 3,
};

enum SensorKind {
  kGyro = 0,
  kAccelerometer = 1,
  kVelocimeter = 2,
  kFramePosition = 3,
  kFrameLinvel = 4,
  kFrameAngvel = 5,
  kFrameZAxis = 6,
  kContactForce = 7,
  kContactFound = 8,
  kJointPosition = 9,
  kJointVelocity = 10,
  kJointActuatorForce = 11,
  kFrameQuat = 12,
};

enum ActuatorKind {
  kPositionActuator = 0,
  kVelocityActuator = 1,
  kMotorActuator = 2,
  kDamperActuator = 3,
  kGeneralActuator = 4,
};

using drake::geometry::CollisionFilterDeclaration;
using drake::geometry::Box;
using drake::geometry::Capsule;
using drake::geometry::Cylinder;
using drake::geometry::Ellipsoid;
using drake::geometry::HalfSpace;
using drake::geometry::GeometrySet;
using drake::geometry::Sphere;
using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::multibody::BodyIndex;
using drake::multibody::ContactResults;
using drake::multibody::ContactModel;
using drake::multibody::DiscreteContactApproximation;
using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::Joint;
using drake::multibody::JointActuatorIndex;
using drake::multibody::JointIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::PdControllerGains;
using drake::multibody::RigidBody;
using drake::multibody::SpatialForce;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

py::array_t<double> MakeArray(const std::vector<py::ssize_t>& shape) {
  return py::array_t<double>(shape);
}

bool AllCovered(const std::vector<bool>& covered) {
  return std::all_of(covered.begin(), covered.end(), [](bool value) { return value; });
}

std::string NormalizeGeometryName(const std::string& name) {
  const std::size_t separator = name.rfind("::");
  if (separator == std::string::npos) {
    return name;
  }
  return name.substr(separator + 2);
}

struct FreeJointMapping {
  int compact_qpos{};
  int compact_qvel{};
  int drake_qpos{};
  int drake_qvel{};
};

struct DirectJointMapping {
  int compact_qpos{};
  int compact_qvel{};
  int drake_qpos{};
  int drake_qvel{};
  int qpos_dim{};
  int qvel_dim{};
};

struct StateLayout {
  std::vector<FreeJointMapping> free_joints;
  std::vector<DirectJointMapping> direct_joints;
};

Eigen::VectorXd CompactQposToDrake(const double* qpos, int nq, const StateLayout& layout) {
  Eigen::VectorXd out(nq);
  out.setZero();
  for (const auto& joint : layout.free_joints) {
    out.segment(joint.drake_qpos, 4) =
        Eigen::Map<const Eigen::Vector4d>(qpos + joint.compact_qpos + 3);
    out.segment(joint.drake_qpos + 4, 3) =
        Eigen::Map<const Eigen::Vector3d>(qpos + joint.compact_qpos);
  }
  for (const auto& joint : layout.direct_joints) {
    out.segment(joint.drake_qpos, joint.qpos_dim) =
        Eigen::Map<const Eigen::VectorXd>(qpos + joint.compact_qpos, joint.qpos_dim);
  }
  return out;
}

Eigen::VectorXd CompactQvelToDrake(const double* qvel, int nv, const StateLayout& layout) {
  Eigen::VectorXd out(nv);
  out.setZero();
  for (const auto& joint : layout.free_joints) {
    out.segment(joint.drake_qvel, 3) =
        Eigen::Map<const Eigen::Vector3d>(qvel + joint.compact_qvel + 3);
    out.segment(joint.drake_qvel + 3, 3) =
        Eigen::Map<const Eigen::Vector3d>(qvel + joint.compact_qvel);
  }
  for (const auto& joint : layout.direct_joints) {
    out.segment(joint.drake_qvel, joint.qvel_dim) =
        Eigen::Map<const Eigen::VectorXd>(qvel + joint.compact_qvel, joint.qvel_dim);
  }
  return out;
}

void DrakeQposToCompact(const Eigen::VectorXd& qpos, double* out, const StateLayout& layout) {
  for (const auto& joint : layout.free_joints) {
    Eigen::Map<Eigen::Vector3d> pos(out + joint.compact_qpos);
    Eigen::Map<Eigen::Vector4d> quat(out + joint.compact_qpos + 3);
    pos = qpos.segment(joint.drake_qpos + 4, 3);
    quat = qpos.segment(joint.drake_qpos, 4);
  }
  for (const auto& joint : layout.direct_joints) {
    Eigen::Map<Eigen::VectorXd> compact(out + joint.compact_qpos, joint.qpos_dim);
    compact = qpos.segment(joint.drake_qpos, joint.qpos_dim);
  }
}

void DrakeQvelToCompact(const Eigen::VectorXd& qvel, double* out, const StateLayout& layout) {
  for (const auto& joint : layout.free_joints) {
    Eigen::Map<Eigen::Vector3d> linear(out + joint.compact_qvel);
    Eigen::Map<Eigen::Vector3d> angular(out + joint.compact_qvel + 3);
    linear = qvel.segment(joint.drake_qvel + 3, 3);
    angular = qvel.segment(joint.drake_qvel, 3);
  }
  for (const auto& joint : layout.direct_joints) {
    Eigen::Map<Eigen::VectorXd> compact(out + joint.compact_qvel, joint.qvel_dim);
    compact = qvel.segment(joint.drake_qvel, joint.qvel_dim);
  }
}

void RequireShape(const py::buffer_info& info, const std::vector<py::ssize_t>& shape,
                  const std::string& name) {
  if (info.ndim != static_cast<py::ssize_t>(shape.size())) {
    throw std::invalid_argument(name + " has wrong rank");
  }
  for (int i = 0; i < info.ndim; ++i) {
    if (info.shape[i] != shape[i]) {
      throw std::invalid_argument(name + " has wrong shape");
    }
  }
}

struct ThreadWorkspace {
  Context<double>* plant_context{};
  std::unique_ptr<Simulator<double>> simulator;
  std::vector<double> last_actuator_effort;
};

struct NativeShapeRecord {
  std::string type;
  std::array<double, 3> parameters{};
};

NativeShapeRecord ReadNativeShape(const drake::geometry::Shape& shape) {
  return shape.Visit<NativeShapeRecord>([](const auto& value) -> NativeShapeRecord {
    using ShapeType = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<ShapeType, Sphere>) {
      return {"sphere", {value.radius(), 0.0, 0.0}};
    } else if constexpr (std::is_same_v<ShapeType, Box>) {
      return {"box", {value.size()(0), value.size()(1), value.size()(2)}};
    } else if constexpr (std::is_same_v<ShapeType, Capsule>) {
      return {"capsule", {value.radius(), value.length(), 0.0}};
    } else if constexpr (std::is_same_v<ShapeType, Cylinder>) {
      return {"cylinder", {value.radius(), value.length(), 0.0}};
    } else if constexpr (std::is_same_v<ShapeType, Ellipsoid>) {
      return {"ellipsoid", {value.a(), value.b(), value.c()}};
    } else if constexpr (std::is_same_v<ShapeType, HalfSpace>) {
      return {"half_space", {0.0, 0.0, 0.0}};
    } else {
      throw std::runtime_error(
          "native geometry identity is unsupported for Drake shape " +
          std::string(value.type_name()));
    }
  });
}

struct NativeGeometryMaterial {
  std::string name;
  bool compliant{};
  double modulus{};
  double resolution_hint{};
  bool has_dissipation{};
  double dissipation{};
};

// Native mirror of drake_uni.runtime.DrakePhysicsConfig. The Python runtime
// validates and serializes the configuration; these parsers re-validate so
// that invalid combinations fail loudly instead of silently substituting
// another mode.
struct NativePhysicsConfig {
  std::string contact_model = "point";
  std::string discrete_approximation = "sap";
  double penetration_allowance = 1.0e-4;
  std::vector<NativeGeometryMaterial> materials;
  bool frictionloss_enabled = false;
  double frictionloss_speed = 0.0;
  std::vector<double> joint_frictionloss;
  bool preserve_gravcomp = false;
  std::unordered_map<std::string, double> body_gravcomp;
};

std::string RequirePhysicsString(const py::dict& payload, const std::string& key) {
  if (!payload.contains(key.c_str())) {
    throw std::invalid_argument("physics payload is missing " + key);
  }
  return py::cast<std::string>(payload[key.c_str()]);
}

double RequirePhysicsDouble(const py::dict& payload, const std::string& key) {
  if (!payload.contains(key.c_str())) {
    throw std::invalid_argument("physics payload is missing " + key);
  }
  return py::cast<double>(payload[key.c_str()]);
}

bool RequirePhysicsBool(const py::dict& payload, const std::string& key) {
  if (!payload.contains(key.c_str())) {
    throw std::invalid_argument("physics payload is missing " + key);
  }
  return py::cast<bool>(payload[key.c_str()]);
}

std::optional<double> OptionalPhysicsDouble(const py::dict& payload, const std::string& key) {
  if (!payload.contains(key.c_str()) || payload[key.c_str()].is_none()) {
    return std::nullopt;
  }
  return py::cast<double>(payload[key.c_str()]);
}

NativePhysicsConfig ParsePhysicsConfig(const py::object& physics) {
  NativePhysicsConfig config;
  if (physics.is_none()) {
    return config;
  }
  if (!py::isinstance<py::dict>(physics)) {
    throw std::invalid_argument("physics must be a dict or None");
  }
  const py::dict payload = py::cast<py::dict>(physics);
  config.contact_model = RequirePhysicsString(payload, "contact_model");
  if (config.contact_model != "point" && config.contact_model != "hydroelastic" &&
      config.contact_model != "hydroelastic_with_fallback") {
    throw std::invalid_argument("unsupported contact_model: " + config.contact_model);
  }
  config.discrete_approximation =
      RequirePhysicsString(payload, "discrete_contact_approximation");
  if (config.discrete_approximation != "sap" &&
      config.discrete_approximation != "lagged" &&
      config.discrete_approximation != "similar") {
    throw std::invalid_argument(
        "unsupported discrete_contact_approximation: " + config.discrete_approximation);
  }
  config.penetration_allowance = RequirePhysicsDouble(payload, "penetration_allowance");
  if (!std::isfinite(config.penetration_allowance) || config.penetration_allowance <= 0.0) {
    throw std::invalid_argument("penetration_allowance must be positive and finite");
  }

  if (payload.contains("geometry_materials")) {
    const py::dict materials = py::cast<py::dict>(payload["geometry_materials"]);
    for (const auto& item : materials) {
      NativeGeometryMaterial material;
      material.name = py::cast<std::string>(item.first);
      const py::dict entry = py::cast<py::dict>(item.second);
      const std::string hydroelastic_type = RequirePhysicsString(entry, "hydroelastic_type");
      if (hydroelastic_type == "compliant") {
        material.compliant = true;
      } else if (hydroelastic_type != "rigid") {
        throw std::invalid_argument(
            "unsupported hydroelastic_type for geometry " + material.name + ": " +
            hydroelastic_type);
      }
      const auto modulus = OptionalPhysicsDouble(entry, "hydroelastic_modulus");
      const auto hint = OptionalPhysicsDouble(entry, "resolution_hint");
      if (material.compliant) {
        if (!modulus.has_value() || !hint.has_value()) {
          throw std::invalid_argument(
              "compliant geometry material " + material.name +
              " requires hydroelastic_modulus and resolution_hint");
        }
        if (*modulus <= 0.0 || *hint <= 0.0) {
          throw std::invalid_argument(
              "compliant geometry material " + material.name +
              " requires positive hydroelastic_modulus and resolution_hint");
        }
        material.modulus = *modulus;
        material.resolution_hint = *hint;
      } else if (modulus.has_value() || hint.has_value()) {
        throw std::invalid_argument(
            "rigid geometry material " + material.name +
            " does not accept hydroelastic_modulus or resolution_hint");
      }
      const auto dissipation = OptionalPhysicsDouble(entry, "hunt_crossley_dissipation");
      if (dissipation.has_value()) {
        if (*dissipation < 0.0) {
          throw std::invalid_argument(
              "hunt_crossley_dissipation must be non-negative for geometry " +
              material.name);
        }
        material.has_dissipation = true;
        material.dissipation = *dissipation;
      }
      config.materials.push_back(std::move(material));
    }
  }

  const auto speed = OptionalPhysicsDouble(payload, "frictionloss_regularization_speed");
  if (speed.has_value()) {
    if (!std::isfinite(*speed) || *speed <= 0.0) {
      throw std::invalid_argument("frictionloss_regularization_speed must be positive and finite");
    }
    config.frictionloss_enabled = true;
    config.frictionloss_speed = *speed;
    const auto losses = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
        payload["joint_frictionloss"]);
    const auto losses_info = losses.request();
    if (losses_info.ndim != 1) {
      throw std::invalid_argument("joint_frictionloss must be one-dimensional");
    }
    const auto losses_view = losses.unchecked<1>();
    config.joint_frictionloss.reserve(static_cast<std::size_t>(losses_view.shape(0)));
    for (py::ssize_t i = 0; i < losses_view.shape(0); ++i) {
      if (!std::isfinite(losses_view(i)) || losses_view(i) < 0.0) {
        throw std::invalid_argument("joint_frictionloss entries must be non-negative and finite");
      }
      config.joint_frictionloss.push_back(losses_view(i));
    }
  }
  config.preserve_gravcomp = RequirePhysicsBool(payload, "preserve_gravity_compensation");
  if (config.preserve_gravcomp && payload.contains("body_gravcomp")) {
    const py::dict gravcomp = py::cast<py::dict>(payload["body_gravcomp"]);
    for (const auto& item : gravcomp) {
      config.body_gravcomp.emplace(py::cast<std::string>(item.first),
                                   py::cast<double>(item.second));
    }
  }

  if (config.contact_model == "point" && !config.materials.empty()) {
    throw std::invalid_argument(
        "geometry_materials require a hydroelastic contact model; they cannot take "
        "effect under contact_model='point'");
  }
  if (config.contact_model == "hydroelastic" && config.materials.empty()) {
    throw std::invalid_argument(
        "contact_model='hydroelastic' is strict: it requires geometry_materials "
        "assignments for every collision geometry");
  }
  return config;
}

ContactModel ContactModelFromName(const std::string& name) {
  if (name == "point") {
    return ContactModel::kPoint;
  }
  if (name == "hydroelastic") {
    return ContactModel::kHydroelastic;
  }
  return ContactModel::kHydroelasticWithFallback;
}

std::string ContactModelName(ContactModel model) {
  switch (model) {
    case ContactModel::kPoint:
      return "point";
    case ContactModel::kHydroelastic:
      return "hydroelastic";
    case ContactModel::kHydroelasticWithFallback:
      return "hydroelastic_with_fallback";
  }
  throw std::runtime_error("unknown Drake contact model");
}

DiscreteContactApproximation DiscreteApproximationFromName(const std::string& name) {
  if (name == "sap") {
    return DiscreteContactApproximation::kSap;
  }
  if (name == "lagged") {
    return DiscreteContactApproximation::kLagged;
  }
  return DiscreteContactApproximation::kSimilar;
}

std::string DiscreteApproximationName(DiscreteContactApproximation approximation) {
  switch (approximation) {
    case DiscreteContactApproximation::kSap:
      return "sap";
    case DiscreteContactApproximation::kLagged:
      return "lagged";
    case DiscreteContactApproximation::kSimilar:
      return "similar";
  }
  throw std::runtime_error("unknown Drake discrete contact approximation");
}

void RemovePropertyIfPresent(drake::geometry::ProximityProperties* props,
                             const std::string& group, const std::string& name) {
  if (props->HasProperty(group, name)) {
    props->RemoveProperty(group, name);
  }
}

struct FrictionlossEntry {
  int velocity_index{};
  double frictionloss{};
  double regularization_speed{};
};

// Documented approximation for MJCF joint frictionloss, which Drake's MJCF
// parser ignores. Applies a bounded dissipative generalized force
// -frictionloss * tanh(v / regularization_speed) per affected single-dof
// joint: the torque is bounded by the authored dry-friction magnitude,
// opposes the joint velocity, and is smoothly regularized at the configured
// speed so the discrete plant remains well-posed at rest.
class FrictionlossBrake final : public drake::systems::LeafSystem<double> {
 public:
  FrictionlossBrake(int num_positions, int num_states, int num_velocities,
                    std::vector<FrictionlossEntry> entries)
      : num_positions_(num_positions), entries_(std::move(entries)) {
    this->DeclareVectorInputPort("plant_state", num_states);
    this->DeclareVectorOutputPort(
        "generalized_force", num_velocities, &FrictionlossBrake::CalcForce,
        {this->input_port_ticket(drake::systems::InputPortIndex(0))});
  }

 private:
  void CalcForce(const drake::systems::Context<double>& context,
                 drake::systems::BasicVector<double>* output) const {
    Eigen::VectorXd force = Eigen::VectorXd::Zero(output->size());
    const Eigen::VectorXd& state = this->get_input_port(0).Eval(context);
    for (const auto& entry : entries_) {
      const double velocity = state[num_positions_ + entry.velocity_index];
      force[entry.velocity_index] =
          -entry.frictionloss * std::tanh(velocity / entry.regularization_speed);
    }
    output->SetFromVector(force);
  }

  int num_positions_{};
  std::vector<FrictionlossEntry> entries_;
};

class DrakeEnvPool {
 public:
  DrakeEnvPool(const std::string& model_file, int nbatch, double sim_dt,
               py::array_t<double, py::array::c_style | py::array::forcecast> ctrl_limits,
               py::array_t<double, py::array::c_style | py::array::forcecast> torque_limits,
               py::array_t<int, py::array::c_style | py::array::forcecast> actuator_kind,
               py::array_t<double, py::array::c_style | py::array::forcecast> actuator_gear,
               py::array_t<double, py::array::c_style | py::array::forcecast> actuator_stiffness,
               py::array_t<double, py::array::c_style | py::array::forcecast> actuator_damping,
               py::array_t<double, py::array::c_style | py::array::forcecast> actuator_gainprm,
               py::array_t<double, py::array::c_style | py::array::forcecast> actuator_biasprm,
               py::array_t<int, py::array::c_style | py::array::forcecast> joint_layout_kind,
               py::array_t<int, py::array::c_style | py::array::forcecast> joint_layout_qpos_adr,
               py::array_t<int, py::array::c_style | py::array::forcecast> joint_layout_qvel_adr,
               py::array_t<int, py::array::c_style | py::array::forcecast> joint_layout_qpos_dim,
               py::array_t<int, py::array::c_style | py::array::forcecast> joint_layout_qvel_dim,
               const std::vector<std::string>& joint_layout_names,
               const std::vector<std::string>& joint_layout_body_names,
               const std::vector<std::string>& collision_filter_geom_names1,
               const std::vector<std::string>& collision_filter_geom_names2,
               const std::vector<int>& sensor_frame_body_indices,
               py::array_t<double, py::array::c_style | py::array::forcecast> sensor_frame_offsets,
               const std::vector<int>& sensor_frame_ref_body_indices,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   sensor_frame_ref_offsets,
               py::array_t<int, py::array::c_style | py::array::forcecast> sensor_type,
               py::array_t<int, py::array::c_style | py::array::forcecast> sensor_index,
               py::array_t<int, py::array::c_style | py::array::forcecast> sensor_adr,
               py::array_t<int, py::array::c_style | py::array::forcecast> sensor_dim,
               int nsensordata, int nthread, py::object physics)
      : nbatch_(nbatch),
        sim_dt_(sim_dt),
        ctrl_limits_(std::move(ctrl_limits)),
        torque_limits_(std::move(torque_limits)),
        actuator_kind_(std::move(actuator_kind)),
        actuator_gear_(std::move(actuator_gear)),
        actuator_stiffness_(std::move(actuator_stiffness)),
        actuator_damping_(std::move(actuator_damping)),
        actuator_gainprm_(std::move(actuator_gainprm)),
        actuator_biasprm_(std::move(actuator_biasprm)),
        joint_layout_kind_(std::move(joint_layout_kind)),
        joint_layout_qpos_adr_(std::move(joint_layout_qpos_adr)),
        joint_layout_qvel_adr_(std::move(joint_layout_qvel_adr)),
        joint_layout_qpos_dim_(std::move(joint_layout_qpos_dim)),
        joint_layout_qvel_dim_(std::move(joint_layout_qvel_dim)),
        joint_layout_names_(joint_layout_names),
        joint_layout_body_names_(joint_layout_body_names),
        collision_filter_geom_names1_(collision_filter_geom_names1),
        collision_filter_geom_names2_(collision_filter_geom_names2),
        sensor_frame_body_indices_(sensor_frame_body_indices),
        sensor_frame_offsets_(std::move(sensor_frame_offsets)),
        sensor_frame_ref_body_indices_(sensor_frame_ref_body_indices),
        sensor_frame_ref_offsets_(std::move(sensor_frame_ref_offsets)),
        sensor_type_(std::move(sensor_type)),
        sensor_index_(std::move(sensor_index)),
        sensor_adr_(std::move(sensor_adr)),
        sensor_dim_(std::move(sensor_dim)),
        nsensordata_(nsensordata) {
    if (nbatch_ < 1) {
      throw std::invalid_argument("nbatch must be >= 1");
    }
    physics_ = ParsePhysicsConfig(physics);
    nthread_ = std::max(1, std::min(nbatch_, std::max(1, nthread)));
    auto ctrl_info = ctrl_limits_.request();
    if (ctrl_info.ndim != 2 || ctrl_info.shape[1] != 2) {
      throw std::invalid_argument("ctrl_limits must have shape (nu, 2)");
    }
    nu_ = static_cast<int>(ctrl_info.shape[0]);
    RequireShape(torque_limits_.request(), {nu_}, "torque_limits");
    RequireShape(actuator_kind_.request(), {nu_}, "actuator_kind");
    RequireShape(actuator_gear_.request(), {nu_}, "actuator_gear");
    RequireShape(actuator_stiffness_.request(), {nu_}, "actuator_stiffness");
    RequireShape(actuator_damping_.request(), {nu_}, "actuator_damping");
    RequireShape(actuator_gainprm_.request(), {nu_, 3}, "actuator_gainprm");
    RequireShape(actuator_biasprm_.request(), {nu_, 3}, "actuator_biasprm");
    const auto joint_layout_info = joint_layout_kind_.request();
    if (joint_layout_info.ndim != 1) {
      throw std::invalid_argument("joint_layout_kind must be one-dimensional");
    }
    const py::ssize_t joint_layout_count = joint_layout_info.shape[0];
    RequireShape(joint_layout_qpos_adr_.request(), {joint_layout_count}, "joint_layout_qpos_adr");
    RequireShape(joint_layout_qvel_adr_.request(), {joint_layout_count}, "joint_layout_qvel_adr");
    RequireShape(joint_layout_qpos_dim_.request(), {joint_layout_count}, "joint_layout_qpos_dim");
    RequireShape(joint_layout_qvel_dim_.request(), {joint_layout_count}, "joint_layout_qvel_dim");
    if (joint_layout_names_.size() != static_cast<std::size_t>(joint_layout_count) ||
        joint_layout_body_names_.size() != static_cast<std::size_t>(joint_layout_count)) {
      throw std::invalid_argument("joint layout name vectors must match joint layout arrays");
    }
    if (collision_filter_geom_names1_.size() != collision_filter_geom_names2_.size()) {
      throw std::invalid_argument("collision filter geom name vectors must have equal length");
    }
    const py::ssize_t frame_count =
        static_cast<py::ssize_t>(sensor_frame_body_indices_.size());
    RequireShape(sensor_frame_offsets_.request(), {frame_count, 3}, "sensor_frame_offsets");
    if (sensor_frame_ref_body_indices_.size() != sensor_frame_body_indices_.size()) {
      throw std::invalid_argument(
          "sensor_frame_ref_body_indices must match sensor_frame_body_indices");
    }
    RequireShape(sensor_frame_ref_offsets_.request(), {frame_count, 3},
                 "sensor_frame_ref_offsets");
    const auto sensor_type_info = sensor_type_.request();
    if (sensor_type_info.ndim != 1) {
      throw std::invalid_argument("sensor_type must be one-dimensional");
    }
    const py::ssize_t sensor_count = sensor_type_info.shape[0];
    sensor_count_ = static_cast<int>(sensor_count);
    RequireShape(sensor_index_.request(), {sensor_count}, "sensor_index");
    RequireShape(sensor_adr_.request(), {sensor_count}, "sensor_adr");
    RequireShape(sensor_dim_.request(), {sensor_count}, "sensor_dim");
    if (nsensordata_ < 0) {
      throw std::invalid_argument("nsensordata must be non-negative");
    }

    DiagramBuilder<double> builder;
    auto [plant_ref, scene_graph_ref] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_ = &plant_ref;
    scene_graph_ = &scene_graph_ref;
    ApplyContactSettings();
    const auto model_instances = Parser(plant_).AddModels(model_file);
    if (model_instances.size() != 1) {
      throw std::runtime_error("DrakeEnvPool expected exactly one model instance");
    }
    model_instance_ = model_instances.at(0);

    auto torque = torque_limits_.unchecked<1>();
    auto kind = actuator_kind_.unchecked<1>();
    auto stiffness = actuator_stiffness_.unchecked<1>();
    auto damping = actuator_damping_.unchecked<1>();
    for (int i = 0; i < nu_; ++i) {
      auto& actuator = plant_->get_mutable_joint_actuator(JointActuatorIndex(i));
      actuator.set_effort_limit(torque(i));
      switch (kind(i)) {
        case kPositionActuator:
          actuator.set_controller_gains(PdControllerGains(stiffness(i), damping(i)));
          break;
        case kVelocityActuator:
          actuator.set_controller_gains(PdControllerGains(0.0, damping(i)));
          break;
        case kMotorActuator:
        case kDamperActuator:
        case kGeneralActuator:
          break;
        default:
          throw std::invalid_argument("unknown DrakeUni actuator kind");
      }
    }

    // Drake's MJCF parser warns that it ignores MuJoCo contype/conaffinity.
    // Apply the same mask rule ourselves before Finalize().
    num_filtered_geometries_ = ApplyMjcfCollisionFilters();
    ApplyGeometryMaterials();
    CacheGravcompForces();
    plant_->Finalize();
    CacheActuatorJointIndices();
    for (int frame_body_index : sensor_frame_body_indices_) {
      sensor_frame_bodies_.push_back(&plant_->get_body(BodyIndex(frame_body_index)));
    }
    for (int frame_body_index : sensor_frame_ref_body_indices_) {
      if (frame_body_index < 0) {
        sensor_frame_ref_bodies_.push_back(nullptr);
      } else {
        sensor_frame_ref_bodies_.push_back(&plant_->get_body(BodyIndex(frame_body_index)));
      }
    }

    nq_ = plant_->num_positions();
    nv_ = plant_->num_velocities();
    state_dim_ = 1 + nq_ + nv_;
    WireFrictionlossBrake(&builder);
    diagram_ = builder.Build();

    BuildStateLayout();
    if (nu_ != plant_->num_actuators()) {
      throw std::runtime_error("ctrl_limits length does not match plant actuators");
    }
    ValidateSensorLayout();
    compact_state_.assign(static_cast<std::size_t>(nbatch_) * state_dim_, 0.0);
    workspaces_.reserve(nthread_);
    for (int i = 0; i < nthread_; ++i) {
      workspaces_.push_back(MakeWorkspace());
    }
  }

  int nbatch() const { return nbatch_; }
  int state_dim() const { return state_dim_; }
  int num_positions() const { return nq_; }
  int num_velocities() const { return nv_; }
  int control_dim() const { return nu_; }
  int num_bodies() const { return plant_->num_bodies(); }
  int nsensordata() const { return nsensordata_; }
  int nthread() const { return nthread_; }
  int workspace_count() const { return static_cast<int>(workspaces_.size()); }
  int num_filtered_geometries() const { return num_filtered_geometries_; }

  py::array_t<double> default_state() const {
    auto state_out = MakeArray({state_dim_});
    auto state = state_out.mutable_unchecked<1>();
    const auto& runtime = workspaces_.at(0);
    state(0) = runtime.simulator->get_context().get_time();
    const Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    const Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    DrakeQposToCompact(q, &state(1), state_layout_);
    DrakeQvelToCompact(v, &state(1 + nq_), state_layout_);
    return state_out;
  }

  py::dict native_model_properties() const {
    const int body_count = plant_->num_bodies();
    std::vector<std::string> body_names;
    body_names.reserve(static_cast<std::size_t>(body_count));
    auto masses = MakeArray({body_count});
    auto coms = MakeArray({body_count, 3});
    auto inertias = MakeArray({body_count, 3, 3});
    auto masses_view = masses.mutable_unchecked<1>();
    auto coms_view = coms.mutable_unchecked<2>();
    auto inertias_view = inertias.mutable_unchecked<3>();

    std::vector<std::string> geometry_names;
    std::vector<int> geometry_body_indices;
    std::vector<std::string> geometry_types;
    std::vector<bool> geometry_collision;
    std::vector<double> geometry_parameters;
    struct NativeGeometryCandidate {
      std::string name;
      drake::geometry::GeometryId id;
      int body_index;
      bool collision;
    };
    std::vector<NativeGeometryCandidate> geometry_candidates;
    std::unordered_map<std::string, std::size_t> geometry_candidate_indices;
    const auto& inspector = scene_graph_->model_inspector();

    for (int body_index = 0; body_index < body_count; ++body_index) {
      const RigidBody<double>& body = plant_->get_body(BodyIndex(body_index));
      body_names.push_back(body.name());
      const auto& spatial_inertia = body.default_spatial_inertia();
      if (body_index == plant_->world_body().index()) {
        // Drake keeps the welded world body's default SpatialInertia invalid;
        // expose its stable physical meaning as a massless static body.
        masses_view(body_index) = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          coms_view(body_index, axis) = 0.0;
        }
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            inertias_view(body_index, row, column) = 0.0;
          }
        }
      } else {
        masses_view(body_index) = spatial_inertia.get_mass();
        const auto& com = spatial_inertia.get_com();
        for (int axis = 0; axis < 3; ++axis) {
          coms_view(body_index, axis) = com(axis);
        }
        const auto inertia = spatial_inertia.CalcRotationalInertia().CopyToFullMatrix3();
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            inertias_view(body_index, row, column) = inertia(row, column);
          }
        }
      }

      const auto frame_id = plant_->GetBodyFrameIdIfExists(BodyIndex(body_index));
      if (!frame_id.has_value()) {
        continue;
      }
      for (const auto& geometry_id : inspector.GetGeometries(*frame_id)) {
        const std::string name = NormalizeGeometryName(inspector.GetName(geometry_id));
        const bool collision = inspector.GetProximityProperties(geometry_id) != nullptr;
        auto [existing, inserted] = geometry_candidate_indices.emplace(
            name, geometry_candidates.size());
        if (inserted) {
          geometry_candidates.push_back({name, geometry_id, body_index, collision});
          continue;
        }
        NativeGeometryCandidate& candidate = geometry_candidates.at(existing->second);
        if (candidate.body_index != body_index || collision == candidate.collision) {
          throw std::runtime_error(
              "Duplicate Drake geometry name after normalization: " + name);
        }
        if (collision) {
          // One proximity record is authoritative when Drake registers separate
          // illustration and proximity instances of the same authored geometry.
          candidate.id = geometry_id;
          candidate.collision = true;
        }
      }
    }

    std::sort(
        geometry_candidates.begin(), geometry_candidates.end(),
        [](const NativeGeometryCandidate& left, const NativeGeometryCandidate& right) {
          if (left.body_index != right.body_index) {
            return left.body_index < right.body_index;
          }
          return left.name < right.name;
        });
    for (const auto& candidate : geometry_candidates) {
      const NativeShapeRecord shape = ReadNativeShape(
          inspector.GetShape(candidate.id));
      geometry_names.push_back(candidate.name);
      geometry_body_indices.push_back(candidate.body_index);
      geometry_types.push_back(shape.type);
      geometry_collision.push_back(candidate.collision);
      geometry_parameters.push_back(shape.parameters[0]);
      geometry_parameters.push_back(shape.parameters[1]);
      geometry_parameters.push_back(shape.parameters[2]);
    }

    auto parameters = MakeArray({
        static_cast<py::ssize_t>(geometry_names.size()),
        3,
    });
    auto parameters_view = parameters.mutable_unchecked<2>();
    for (std::size_t geometry = 0; geometry < geometry_names.size(); ++geometry) {
      for (int column = 0; column < 3; ++column) {
        parameters_view(geometry, column) = geometry_parameters[geometry * 3 + column];
      }
    }

    py::dict output;
    output["body_names"] = body_names;
    output["body_masses"] = masses;
    output["body_coms"] = coms;
    output["body_inertias"] = inertias;
    output["geometry_names"] = geometry_names;
    output["geometry_body_indices"] = geometry_body_indices;
    output["geometry_types"] = geometry_types;
    output["geometry_collision"] = geometry_collision;
    output["geometry_parameters"] = parameters;
    return output;
  }

  // Native effective-settings readback: the values below are queried from the
  // finalized plant and SceneGraph, never echoed from the request.
  py::dict effective_physics() const {
    py::dict output;
    output["contact_model"] = ContactModelName(plant_->get_contact_model());
    output["discrete_contact_approximation"] =
        DiscreteApproximationName(plant_->get_discrete_contact_approximation());
    // MultibodyPlant exposes a setter only; report the applied value cached
    // from the validated configuration.
    output["penetration_allowance"] = physics_.penetration_allowance;
    py::dict materials;
    const auto& inspector = scene_graph_->model_inspector();
    for (const auto& [name, geometry_id] : material_geometry_ids_) {
      const drake::geometry::ProximityProperties* props =
          inspector.GetProximityProperties(geometry_id);
      if (props == nullptr) {
        throw std::runtime_error("Missing proximity properties for geometry " + name);
      }
      py::dict entry;
      const auto compliance =
          props->GetProperty<drake::geometry::internal::HydroelasticType>(
              "hydroelastic", "compliance_type");
      entry["hydroelastic_type"] =
          drake::geometry::internal::GetStringFromHydroelasticType(compliance);
      entry["hydroelastic_modulus"] =
          props->HasProperty("hydroelastic", "hydroelastic_modulus")
              ? py::cast(props->GetProperty<double>("hydroelastic", "hydroelastic_modulus"))
              : py::none();
      entry["resolution_hint"] = props->HasProperty("hydroelastic", "resolution_hint")
                                     ? py::cast(props->GetProperty<double>("hydroelastic",
                                                                           "resolution_hint"))
                                     : py::none();
      entry["hunt_crossley_dissipation"] =
          props->HasProperty("material", "hunt_crossley_dissipation")
              ? py::cast(props->GetProperty<double>("material", "hunt_crossley_dissipation"))
              : py::none();
      materials[name.c_str()] = entry;
    }
    output["geometry_materials"] = materials;
#ifdef DRAKE_UNI_HAS_DRAKE_VERSION_HEADER
    output["drake_version"] = std::string(DRAKE_VERSION_STRING);
#else
    output["drake_version"] = py::none();
#endif
    return output;
  }

  py::dict step(py::array_t<double, py::array::c_style | py::array::forcecast> state0,
                int nstep,
                py::array_t<double, py::array::c_style | py::array::forcecast> control,
                py::object body_forces,
                bool return_sensor) {
    if (nstep < 1) {
      throw std::invalid_argument("nstep must be >= 1");
    }
    auto state_info = state0.request();
    RequireShape(state_info, {nbatch_, state_dim_}, "state0");
    auto control_info = control.request();
    const bool control_is_traj = control_info.ndim == 3;
    if (control_is_traj) {
      RequireShape(control_info, {nbatch_, nstep, nu_}, "control");
    } else {
      RequireShape(control_info, {nbatch_, nu_}, "control");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> force_array;
    bool has_forces = !body_forces.is_none();
    if (has_forces) {
      force_array = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
          body_forces);
      RequireShape(force_array.request(), {nbatch_, plant_->num_bodies(), 3}, "body_forces");
    }

    auto state_out = MakeArray({nbatch_, state_dim_});
    py::array_t<double> sensor_data;
    if (return_sensor) {
      sensor_data = MakeArray({nbatch_, nsensordata_});
    }
    auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        for (int env_index = begin; env_index < end; ++env_index) {
          StepOne(workspace, env_index, state0, control, control_is_traj, nstep,
                  has_forces ? &force_array : nullptr);
          WriteState(workspace, env_index, state_out);
          if (return_sensor) {
            WriteSensorRow(workspace, env_index, sensor_data);
          }
        }
      };
      RunChunks(worker);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double step_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed).count();

    py::dict timing;
    timing["step_ms"] = step_ms;
    py::dict output;
    output["state"] = state_out;
    if (return_sensor) {
      output["sensor_data"] = sensor_data;
    }
    output["timing"] = timing;
    return output;
  }

  py::dict reset(py::array_t<int, py::array::c_style | py::array::forcecast> env_ids,
                 py::array_t<double, py::array::c_style | py::array::forcecast> initial_state,
                 bool return_sensor) {
    auto ids_info = env_ids.request();
    if (ids_info.ndim != 1) {
      throw std::invalid_argument("env_ids must be one-dimensional");
    }
    const int rows = static_cast<int>(ids_info.shape[0]);
    RequireShape(initial_state.request(), {rows, state_dim_}, "initial_state");
    auto ids = env_ids.unchecked<1>();
    for (int row = 0; row < rows; ++row) {
      const int env_index = ids(row);
      if (env_index < 0 || env_index >= nbatch_) {
        throw std::out_of_range("env_id out of range");
      }
    }

    auto state_rows = MakeArray({rows, state_dim_});
    py::array_t<double> sensor_rows;
    if (return_sensor) {
      sensor_rows = MakeArray({rows, nsensordata_});
    }
    {
      py::gil_scoped_release release;
      auto state = initial_state.unchecked<2>();
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        for (int row = begin; row < end; ++row) {
          const int env_index = ids(row);
          LoadState(workspace, &state(row, 0));
          WriteCompactStateRow(workspace, row, state_rows);
          SaveCompactState(env_index, &state_rows.mutable_unchecked<2>()(row, 0));
          if (return_sensor) {
            WriteSensorRow(workspace, row, sensor_rows);
          }
        }
      };
      RunRowChunks(rows, worker);
    }
    py::dict output;
    output["env_ids"] = env_ids;
    output["state"] = state_rows;
    if (return_sensor) {
      output["sensor_data"] = sensor_rows;
    }
    return output;
  }

  py::dict snapshot(bool return_sensor) { return Snapshot(return_sensor); }

  py::dict compute_body_state(
      py::array_t<double, py::array::c_style | py::array::forcecast> state0,
      py::array_t<int, py::array::c_style | py::array::forcecast> body_indices) {
    RequireShape(state0.request(), {nbatch_, state_dim_}, "state0");
    const auto body_info = body_indices.request();
    if (body_info.ndim != 1) {
      throw std::invalid_argument("body_indices must be one-dimensional");
    }
    const int body_count = static_cast<int>(body_info.shape[0]);
    auto body_view = body_indices.unchecked<1>();
    std::vector<const RigidBody<double>*> bodies;
    bodies.reserve(body_count);
    for (int i = 0; i < body_count; ++i) {
      const int body_index = body_view(i);
      if (body_index < 0 || body_index >= plant_->num_bodies()) {
        throw std::out_of_range("body index out of range");
      }
      bodies.push_back(&plant_->get_body(BodyIndex(body_index)));
    }

    auto pos = MakeArray({nbatch_, body_count, 3});
    auto quat = MakeArray({nbatch_, body_count, 4});
    auto linvel = MakeArray({nbatch_, body_count, 3});
    auto angvel = MakeArray({nbatch_, body_count, 3});
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        auto state = state0.unchecked<2>();
        for (int env_index = begin; env_index < end; ++env_index) {
          LoadState(workspace, &state(env_index, 0));
          WriteBodyStateRow(workspace, env_index, bodies, pos, quat, linvel, angvel);
        }
      };
      RunChunks(worker);
    }
    py::dict output;
    output["pos"] = pos;
    output["quat"] = quat;
    output["linvel"] = linvel;
    output["angvel"] = angvel;
    return output;
  }

 private:
  void MarkCovered(std::vector<bool>& covered, int start, int dim,
                   const std::string& description) const {
    if (start < 0 || dim < 0 || start + dim > static_cast<int>(covered.size())) {
      throw std::runtime_error(description + " is outside compact/Drake state bounds");
    }
    for (int i = start; i < start + dim; ++i) {
      if (covered.at(i)) {
        throw std::runtime_error(description + " overlaps another compact/Drake state segment");
      }
      covered.at(i) = true;
    }
  }

  const Joint<double>& FindJointByChildBody(const std::string& body_name,
                                            int qpos_dim, int qvel_dim) const {
    const Joint<double>* match = nullptr;
    for (int i = 0; i < plant_->num_joints(); ++i) {
      const Joint<double>& joint = plant_->get_joint(JointIndex(i));
      if (joint.num_positions() != qpos_dim || joint.num_velocities() != qvel_dim) {
        continue;
      }
      if (joint.child_body().name() != body_name) {
        continue;
      }
      if (match != nullptr) {
        throw std::runtime_error("Multiple Drake joints match MJCF child body " + body_name);
      }
      match = &joint;
    }
    if (match == nullptr) {
      throw std::runtime_error("Could not map MJCF joint on child body " + body_name +
                               " into Drake state layout");
    }
    return *match;
  }

  void BuildStateLayout() {
    const auto kind = joint_layout_kind_.unchecked<1>();
    const auto qpos_adr = joint_layout_qpos_adr_.unchecked<1>();
    const auto qvel_adr = joint_layout_qvel_adr_.unchecked<1>();
    const auto qpos_dim = joint_layout_qpos_dim_.unchecked<1>();
    const auto qvel_dim = joint_layout_qvel_dim_.unchecked<1>();
    const int joint_count = static_cast<int>(joint_layout_names_.size());

    std::vector<bool> compact_qpos_covered(nq_, false);
    std::vector<bool> compact_qvel_covered(nv_, false);
    std::vector<bool> drake_qpos_covered(nq_, false);
    std::vector<bool> drake_qvel_covered(nv_, false);
    for (int i = 0; i < joint_count; ++i) {
      const int compact_q = qpos_adr(i);
      const int compact_v = qvel_adr(i);
      const int q_dim = qpos_dim(i);
      const int v_dim = qvel_dim(i);
      MarkCovered(compact_qpos_covered, compact_q, q_dim, "MJCF compact qpos layout");
      MarkCovered(compact_qvel_covered, compact_v, v_dim, "MJCF compact qvel layout");

      if (kind(i) == kFreeJoint) {
        const Joint<double>& joint =
            FindJointByChildBody(joint_layout_body_names_.at(i), kRootQposDim, kRootQvelDim);
        state_layout_.free_joints.push_back(FreeJointMapping{
            compact_q, compact_v, joint.position_start(), joint.velocity_start()});
        MarkCovered(drake_qpos_covered, joint.position_start(), kRootQposDim,
                    "Drake free-joint qpos layout");
        MarkCovered(drake_qvel_covered, joint.velocity_start(), kRootQvelDim,
                    "Drake free-joint qvel layout");
        continue;
      }

      if (kind(i) == kBallJoint) {
        const Joint<double>& joint = FindJointByChildBody(joint_layout_body_names_.at(i), 4, 3);
        state_layout_.direct_joints.push_back(
            DirectJointMapping{compact_q, compact_v, joint.position_start(),
                               joint.velocity_start(), q_dim, v_dim});
        MarkCovered(drake_qpos_covered, joint.position_start(), q_dim,
                    "Drake ball-joint qpos layout");
        MarkCovered(drake_qvel_covered, joint.velocity_start(), v_dim,
                    "Drake ball-joint qvel layout");
        continue;
      }

      if (kind(i) != kSlideJoint && kind(i) != kHingeJoint) {
        throw std::runtime_error("Unknown MJCF compact joint kind in DrakeUni state layout");
      }
      const Joint<double>& joint =
          FindJointByChildBody(joint_layout_body_names_.at(i), 1, 1);
      const std::string& compact_name = joint_layout_names_.at(i);
      if (!compact_name.empty() && joint.name() != compact_name) {
        throw std::runtime_error("MJCF one-dof joint " + compact_name +
                                 " mapped to unexpected Drake joint " + joint.name());
      }
      state_layout_.direct_joints.push_back(
          DirectJointMapping{compact_q, compact_v, joint.position_start(),
                             joint.velocity_start(), q_dim, v_dim});
      MarkCovered(drake_qpos_covered, joint.position_start(), q_dim,
                  "Drake one-dof qpos layout");
      MarkCovered(drake_qvel_covered, joint.velocity_start(), v_dim,
                  "Drake one-dof qvel layout");
    }

    if (!AllCovered(compact_qpos_covered) || !AllCovered(compact_qvel_covered) ||
        !AllCovered(drake_qpos_covered) || !AllCovered(drake_qvel_covered)) {
      throw std::runtime_error(
          "DrakeUni compact/Drake state layout did not cover all qpos/qvel entries");
    }
  }

  ThreadWorkspace MakeWorkspace() {
    ThreadWorkspace runtime;
    auto context = diagram_->CreateDefaultContext();
    runtime.plant_context = &plant_->GetMyMutableContextFromRoot(context.get());
    runtime.last_actuator_effort.assign(nu_, 0.0);
    plant_->get_actuation_input_port(model_instance_)
        .FixValue(runtime.plant_context, Eigen::VectorXd::Zero(nu_));
    SetNeutralActuatorInputs(runtime.plant_context);
    runtime.simulator = std::make_unique<Simulator<double>>(*diagram_, std::move(context));
    runtime.plant_context =
        &plant_->GetMyMutableContextFromRoot(&runtime.simulator->get_mutable_context());
    runtime.simulator->set_target_realtime_rate(0.0);
    runtime.simulator->Initialize();
    return runtime;
  }

  void LoadState(ThreadWorkspace& runtime, const double* state_row) {
    for (int i = 0; i < state_dim_; ++i) {
      if (!std::isfinite(state_row[i])) {
        throw std::invalid_argument("state contains non-finite values");
      }
    }
    runtime.simulator->get_mutable_context().SetTime(state_row[0]);
    plant_->SetPositions(runtime.plant_context,
                         CompactQposToDrake(state_row + 1, nq_, state_layout_));
    plant_->SetVelocities(runtime.plant_context,
                          CompactQvelToDrake(state_row + 1 + nq_, nv_, state_layout_));
    if (nu_ > 0) {
      SetNeutralActuatorInputs(runtime.plant_context);
      std::fill(runtime.last_actuator_effort.begin(), runtime.last_actuator_effort.end(), 0.0);
    }
    runtime.simulator->Initialize();
  }

  void StepOne(ThreadWorkspace& runtime, int env_index,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& state0,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& control,
               bool control_is_traj, int nstep,
               const py::array_t<double, py::array::c_style | py::array::forcecast>* body_forces) {
    auto state = state0.unchecked<2>();
    LoadState(runtime, &state(env_index, 0));
    auto ctrl_limits = ctrl_limits_.unchecked<2>();
    for (int substep = 0; substep < nstep; ++substep) {
      Eigen::VectorXd target(nu_);
      if (control_is_traj) {
        auto control_values = control.unchecked<3>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, substep, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      } else {
        auto control_values = control.unchecked<2>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      }
      ApplyActuatorControl(target, runtime);
      if (body_forces != nullptr) {
        SetExternalBodyForces(runtime, env_index, body_forces);
      } else {
        SetExternalBodyForces(runtime, env_index, nullptr);
      }
      runtime.simulator->AdvanceTo(runtime.simulator->get_context().get_time() + sim_dt_);
    }
  }

  void CacheActuatorJointIndices() {
    actuator_position_start_.clear();
    actuator_velocity_start_.clear();
    actuator_position_start_.reserve(nu_);
    actuator_velocity_start_.reserve(nu_);
    for (int i = 0; i < nu_; ++i) {
      const auto& actuator = plant_->get_joint_actuator(JointActuatorIndex(i));
      if (actuator.num_inputs() != 1 || actuator.joint().num_positions() != 1 ||
          actuator.joint().num_velocities() != 1) {
        throw std::invalid_argument(
            "DrakeUni actuator support is limited to single-dof joint actuators");
      }
      actuator_position_start_.push_back(actuator.joint().position_start());
      actuator_velocity_start_.push_back(actuator.joint().velocity_start());
    }
  }

  void SetNeutralActuatorInputs(Context<double>* plant_context) {
    Eigen::VectorXd direct = Eigen::VectorXd::Zero(nu_);
    Eigen::VectorXd desired = Eigen::VectorXd::Zero(2 * nu_);
    const Eigen::VectorXd q = plant_->GetPositions(*plant_context);
    const Eigen::VectorXd v = plant_->GetVelocities(*plant_context);
    auto kind = actuator_kind_.unchecked<1>();
    for (int i = 0; i < nu_; ++i) {
      if (kind(i) == kPositionActuator) {
        desired[i] = q[actuator_position_start_.at(i)];
      } else if (kind(i) == kVelocityActuator) {
        desired[nu_ + i] = v[actuator_velocity_start_.at(i)];
      }
    }
    plant_->get_actuation_input_port(model_instance_).FixValue(plant_context, direct);
    plant_->get_desired_state_input_port(model_instance_).FixValue(plant_context, desired);
  }

  void ApplyActuatorControl(const Eigen::VectorXd& control, ThreadWorkspace& runtime) {
    Context<double>* plant_context = runtime.plant_context;
    Eigen::VectorXd direct = Eigen::VectorXd::Zero(nu_);
    Eigen::VectorXd desired = Eigen::VectorXd::Zero(2 * nu_);
    const Eigen::VectorXd q = plant_->GetPositions(*plant_context);
    const Eigen::VectorXd v = plant_->GetVelocities(*plant_context);
    auto kind = actuator_kind_.unchecked<1>();
    auto gear = actuator_gear_.unchecked<1>();
    auto stiffness = actuator_stiffness_.unchecked<1>();
    auto damping = actuator_damping_.unchecked<1>();
    auto gainprm = actuator_gainprm_.unchecked<2>();
    auto biasprm = actuator_biasprm_.unchecked<2>();
    runtime.last_actuator_effort.assign(nu_, 0.0);
    for (int i = 0; i < nu_; ++i) {
      const double u = control[i];
      const int q_index = actuator_position_start_.at(i);
      const int v_index = actuator_velocity_start_.at(i);
      const double g = gear(i);
      switch (kind(i)) {
        case kPositionActuator:
          desired[i] = u;
          desired[nu_ + i] = 0.0;
          runtime.last_actuator_effort[i] =
              LimitEffort(i, stiffness(i) * (u - q[q_index]) - damping(i) * v[v_index]);
          break;
        case kVelocityActuator:
          desired[i] = q[q_index];
          desired[nu_ + i] = u;
          runtime.last_actuator_effort[i] = LimitEffort(i, damping(i) * (u - v[v_index]));
          break;
        case kMotorActuator:
          direct[i] = LimitEffort(i, g * u);
          runtime.last_actuator_effort[i] = direct[i];
          break;
        case kDamperActuator: {
          const double length_rate = g * v[v_index];
          direct[i] = LimitEffort(i, g * (-damping(i) * length_rate * u));
          runtime.last_actuator_effort[i] = direct[i];
          break;
        }
        case kGeneralActuator: {
          const double length = g * q[q_index];
          const double length_rate = g * v[v_index];
          const double gain = gainprm(i, 0) + gainprm(i, 1) * length +
                              gainprm(i, 2) * length_rate;
          const double bias = biasprm(i, 0) + biasprm(i, 1) * length +
                              biasprm(i, 2) * length_rate;
          direct[i] = LimitEffort(i, g * (gain * u + bias));
          runtime.last_actuator_effort[i] = direct[i];
          break;
        }
        default:
          throw std::invalid_argument("unknown DrakeUni actuator kind");
      }
    }
    plant_->get_actuation_input_port(model_instance_).FixValue(plant_context, direct);
    plant_->get_desired_state_input_port(model_instance_).FixValue(plant_context, desired);
  }

  double LimitEffort(int actuator_index, double effort) const {
    auto torque = torque_limits_.unchecked<1>();
    const double limit = torque(actuator_index);
    if (!std::isfinite(limit)) {
      return effort;
    }
    return std::clamp(effort, -limit, limit);
  }

  void SetExternalBodyForces(
      ThreadWorkspace& runtime, int env_index,
      const py::array_t<double, py::array::c_style | py::array::forcecast>* body_forces) {
    std::vector<ExternallyAppliedSpatialForce<double>> forces = gravcomp_forces_;
    if (body_forces != nullptr) {
      auto values = body_forces->unchecked<3>();
      for (int body_index = 0; body_index < plant_->num_bodies(); ++body_index) {
        const double fx = values(env_index, body_index, 0);
        const double fy = values(env_index, body_index, 1);
        const double fz = values(env_index, body_index, 2);
        if (std::abs(fx) > 0.0 || std::abs(fy) > 0.0 || std::abs(fz) > 0.0) {
          ExternallyAppliedSpatialForce<double> applied;
          applied.body_index = BodyIndex(body_index);
          applied.p_BoBq_B.setZero();
          applied.F_Bq_W =
              SpatialForce<double>(Eigen::Vector3d::Zero(), Eigen::Vector3d(fx, fy, fz));
          forces.push_back(applied);
        }
      }
    }
    plant_->get_applied_spatial_force_input_port().FixValue(runtime.plant_context, forces);
  }

  void WriteState(const ThreadWorkspace& runtime, int env_index, py::array_t<double>& state_out) {
    WriteCompactStateRow(runtime, env_index, state_out);
    SaveCompactState(env_index, &state_out.mutable_unchecked<2>()(env_index, 0));
  }

  void WriteCompactStateRow(const ThreadWorkspace& runtime, int row,
                            py::array_t<double>& state_out) const {
    auto state = state_out.mutable_unchecked<2>();
    state(row, 0) = runtime.simulator->get_context().get_time();
    Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    DrakeQposToCompact(q, &state(row, 1), state_layout_);
    DrakeQvelToCompact(v, &state(row, 1 + nq_), state_layout_);
  }

  void ValidateSensorLayout() const {
    auto type = sensor_type_.unchecked<1>();
    auto index = sensor_index_.unchecked<1>();
    auto adr = sensor_adr_.unchecked<1>();
    auto dim = sensor_dim_.unchecked<1>();
    for (int i = 0; i < sensor_count_; ++i) {
      if (adr(i) < 0 || dim(i) < 0 || adr(i) + dim(i) > nsensordata_) {
        throw std::invalid_argument("sensor layout exceeds nsensordata");
      }
      switch (type(i)) {
        case kGyro:
        case kAccelerometer:
        case kVelocimeter:
        case kFramePosition:
        case kFrameLinvel:
        case kFrameAngvel:
        case kFrameZAxis:
          if (dim(i) != 3) {
            throw std::invalid_argument("frame sensor dim must be 3");
          }
          if (index(i) < 0 || index(i) >= static_cast<int>(sensor_frame_bodies_.size())) {
            throw std::invalid_argument("frame sensor index is out of range");
          }
          break;
        case kFrameQuat:
          if (dim(i) != 4) {
            throw std::invalid_argument("frame quaternion sensor dim must be 4");
          }
          if (index(i) < 0 || index(i) >= static_cast<int>(sensor_frame_bodies_.size())) {
            throw std::invalid_argument("frame quaternion sensor index is out of range");
          }
          break;
        case kContactForce:
          if (dim(i) != 3) {
            throw std::invalid_argument("contact force sensor dim must be 3");
          }
          if (index(i) < -1 || index(i) >= plant_->num_bodies()) {
            throw std::invalid_argument("contact force sensor body index is out of range");
          }
          break;
        case kContactFound:
          if (dim(i) != 1) {
            throw std::invalid_argument("contact found sensor dim must be 1");
          }
          if (index(i) < -1 || index(i) >= plant_->num_bodies()) {
            throw std::invalid_argument("contact found sensor body index is out of range");
          }
          break;
        case kJointPosition:
        case kJointVelocity:
        case kJointActuatorForce:
          if (dim(i) != 1) {
            throw std::invalid_argument("joint sensor dim must be 1");
          }
          if (index(i) < 0 || index(i) >= nu_) {
            throw std::invalid_argument("joint sensor actuator index is out of range");
          }
          break;
        default:
          throw std::invalid_argument("unknown DrakeUni sensor type");
      }
    }
  }

  void WriteSensorRow(const ThreadWorkspace& runtime, int env_index,
                      py::array_t<double>& sensor_data) const {
    std::vector<Eigen::Vector3d> frame_positions(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_linvel_w(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_angvel_w(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_linvel_local(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_angvel_local(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_linaccel_local(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_zaxis_w(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector4d> frame_quat_w(
        sensor_frame_bodies_.size(), Eigen::Vector4d::Zero());
    std::vector<Eigen::Matrix3d> frame_rot_w(
        sensor_frame_bodies_.size(), Eigen::Matrix3d::Identity());
    std::vector<bool> frame_has_ref(sensor_frame_bodies_.size(), false);
    std::vector<Eigen::Vector3d> frame_ref_positions(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_ref_linvel_w(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> frame_ref_angvel_w(
        sensor_frame_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix3d> frame_ref_rot_w(
        sensor_frame_bodies_.size(), Eigen::Matrix3d::Identity());
    std::vector<Eigen::Vector3d> contact_forces(
        plant_->num_bodies(), Eigen::Vector3d::Zero());
    std::vector<int> contact_found(plant_->num_bodies(), 0);
    const Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    const Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    auto sensor_frame_offsets = sensor_frame_offsets_.unchecked<2>();
    auto sensor_frame_ref_offsets = sensor_frame_ref_offsets_.unchecked<2>();
    for (int frame = 0; frame < static_cast<int>(sensor_frame_bodies_.size()); ++frame) {
      const RigidTransform<double> x_wb =
          plant_->EvalBodyPoseInWorld(*runtime.plant_context, *sensor_frame_bodies_[frame]);
      const auto velocity_w =
          plant_->EvalBodySpatialVelocityInWorld(*runtime.plant_context,
                                                 *sensor_frame_bodies_[frame]);
      const auto acceleration_w =
          sensor_frame_bodies_[frame]->EvalSpatialAccelerationInWorld(*runtime.plant_context);
      const Eigen::Matrix3d r_wb = x_wb.rotation().matrix();
      const Eigen::Quaterniond quat_wb = x_wb.rotation().ToQuaternion();
      const Eigen::Matrix3d r_bw = r_wb.transpose();
      const Eigen::Vector3d offset(sensor_frame_offsets(frame, 0),
                                   sensor_frame_offsets(frame, 1),
                                   sensor_frame_offsets(frame, 2));
      const Eigen::Vector3d offset_w = r_wb * offset;
      frame_positions[frame] = x_wb.translation() + offset_w;
      frame_angvel_w[frame] = velocity_w.rotational();
      frame_linvel_w[frame] = velocity_w.translational() + velocity_w.rotational().cross(offset_w);
      frame_angvel_local[frame] = r_bw * frame_angvel_w[frame];
      frame_linvel_local[frame] = r_bw * frame_linvel_w[frame];
      frame_linaccel_local[frame] =
          r_bw * (acceleration_w.translational() + acceleration_w.rotational().cross(offset_w) +
                  velocity_w.rotational().cross(velocity_w.rotational().cross(offset_w)));
      frame_zaxis_w[frame] = r_wb.col(2);
      frame_quat_w[frame] =
          Eigen::Vector4d(quat_wb.w(), quat_wb.x(), quat_wb.y(), quat_wb.z());
      frame_rot_w[frame] = r_wb;

      if (sensor_frame_ref_bodies_[frame] != nullptr) {
        const RigidTransform<double> x_wr = plant_->EvalBodyPoseInWorld(
            *runtime.plant_context, *sensor_frame_ref_bodies_[frame]);
        const auto ref_velocity_w = plant_->EvalBodySpatialVelocityInWorld(
            *runtime.plant_context, *sensor_frame_ref_bodies_[frame]);
        const Eigen::Matrix3d r_wr = x_wr.rotation().matrix();
        const Eigen::Vector3d ref_offset(sensor_frame_ref_offsets(frame, 0),
                                         sensor_frame_ref_offsets(frame, 1),
                                         sensor_frame_ref_offsets(frame, 2));
        const Eigen::Vector3d ref_offset_w = r_wr * ref_offset;
        frame_has_ref[frame] = true;
        frame_ref_rot_w[frame] = r_wr;
        frame_ref_positions[frame] = x_wr.translation() + ref_offset_w;
        frame_ref_angvel_w[frame] = ref_velocity_w.rotational();
        frame_ref_linvel_w[frame] =
            ref_velocity_w.translational() + ref_velocity_w.rotational().cross(ref_offset_w);
      }
    }

    const auto& contact_results =
        plant_->get_contact_results_output_port().Eval<ContactResults<double>>(
            *runtime.plant_context);
    for (int i = 0; i < contact_results.num_point_pair_contacts(); ++i) {
      const auto& contact = contact_results.point_pair_contact_info(i);
      const int body_a = static_cast<int>(contact.bodyA_index());
      const int body_b = static_cast<int>(contact.bodyB_index());
      if (body_a >= 0 && body_a < static_cast<int>(contact_forces.size())) {
        contact_found[body_a] = 1;
        contact_forces[body_a] -= contact.contact_force();
      }
      if (body_b >= 0 && body_b < static_cast<int>(contact_forces.size())) {
        contact_found[body_b] = 1;
        contact_forces[body_b] += contact.contact_force();
      }
    }

    auto sensor = sensor_data.mutable_unchecked<2>();
    auto type = sensor_type_.unchecked<1>();
    auto index = sensor_index_.unchecked<1>();
    auto adr = sensor_adr_.unchecked<1>();
    auto dim = sensor_dim_.unchecked<1>();
    for (int item = 0; item < sensor_count_; ++item) {
      const int start = adr(item);
      const int frame = index(item);
      switch (type(item)) {
        case kGyro:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = frame_angvel_local[frame][axis];
          }
          break;
        case kAccelerometer:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = frame_linaccel_local[frame][axis];
          }
          break;
        case kVelocimeter:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = frame_linvel_local[frame][axis];
          }
          break;
        case kFramePosition:
          {
            Eigen::Vector3d value = frame_positions[frame];
            if (frame_has_ref[frame]) {
              value =
                  frame_ref_rot_w[frame].transpose() * (value - frame_ref_positions[frame]);
            }
            for (int axis = 0; axis < 3; ++axis) {
              sensor(env_index, start + axis) = value[axis];
            }
          }
          break;
        case kFrameLinvel:
          {
            Eigen::Vector3d value = frame_linvel_w[frame];
            if (frame_has_ref[frame]) {
              value =
                  frame_ref_rot_w[frame].transpose() * (value - frame_ref_linvel_w[frame]);
            }
            for (int axis = 0; axis < 3; ++axis) {
              sensor(env_index, start + axis) = value[axis];
            }
          }
          break;
        case kFrameAngvel:
          {
            Eigen::Vector3d value = frame_angvel_w[frame];
            if (frame_has_ref[frame]) {
              value =
                  frame_ref_rot_w[frame].transpose() * (value - frame_ref_angvel_w[frame]);
            }
            for (int axis = 0; axis < 3; ++axis) {
              sensor(env_index, start + axis) = value[axis];
            }
          }
          break;
        case kFrameZAxis:
          {
            Eigen::Vector3d value = frame_zaxis_w[frame];
            if (frame_has_ref[frame]) {
              value = frame_ref_rot_w[frame].transpose() * value;
            }
            for (int axis = 0; axis < 3; ++axis) {
              sensor(env_index, start + axis) = value[axis];
            }
          }
          break;
        case kFrameQuat:
          {
            Eigen::Vector4d value = frame_quat_w[frame];
            if (frame_has_ref[frame]) {
              const Eigen::Matrix3d relative_rot =
                  frame_ref_rot_w[frame].transpose() * frame_rot_w[frame];
              const Eigen::Quaterniond relative_quat(relative_rot);
              value = Eigen::Vector4d(relative_quat.w(), relative_quat.x(),
                                      relative_quat.y(), relative_quat.z());
            }
            for (int axis = 0; axis < 4; ++axis) {
              sensor(env_index, start + axis) = value[axis];
            }
          }
          break;
        case kContactForce:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) =
                index(item) < 0 ? 0.0 : contact_forces[index(item)][axis];
          }
          break;
        case kContactFound:
          sensor(env_index, start) =
              index(item) < 0 ? 0.0 : static_cast<double>(contact_found[index(item)]);
          break;
        case kJointPosition:
          sensor(env_index, start) = q[actuator_position_start_.at(index(item))];
          break;
        case kJointVelocity:
          sensor(env_index, start) = v[actuator_velocity_start_.at(index(item))];
          break;
        case kJointActuatorForce:
          sensor(env_index, start) = runtime.last_actuator_effort.at(index(item));
          break;
      }
    }
  }

  void WriteBodyStateRow(const ThreadWorkspace& runtime, int env_index,
                         const std::vector<const RigidBody<double>*>& bodies,
                         py::array_t<double>& pos_out, py::array_t<double>& quat_out,
                         py::array_t<double>& linvel_out,
                         py::array_t<double>& angvel_out) const {
    auto pos = pos_out.mutable_unchecked<3>();
    auto quat = quat_out.mutable_unchecked<3>();
    auto linvel = linvel_out.mutable_unchecked<3>();
    auto angvel = angvel_out.mutable_unchecked<3>();
    for (int body = 0; body < static_cast<int>(bodies.size()); ++body) {
      const RigidTransform<double> x_wb =
          plant_->EvalBodyPoseInWorld(*runtime.plant_context, *bodies[body]);
      const auto velocity_w =
          plant_->EvalBodySpatialVelocityInWorld(*runtime.plant_context, *bodies[body]);
      for (int axis = 0; axis < 3; ++axis) {
        pos(env_index, body, axis) = x_wb.translation()[axis];
        linvel(env_index, body, axis) = velocity_w.translational()[axis];
        angvel(env_index, body, axis) = velocity_w.rotational()[axis];
      }
      const auto q = x_wb.rotation().ToQuaternion();
      quat(env_index, body, 0) = q.w();
      quat(env_index, body, 1) = q.x();
      quat(env_index, body, 2) = q.y();
      quat(env_index, body, 3) = q.z();
    }
  }

  py::dict Snapshot(bool return_sensor) {
    auto state_out = MakeArray({nbatch_, state_dim_});
    py::array_t<double> sensor_data;
    if (return_sensor) {
      sensor_data = MakeArray({nbatch_, nsensordata_});
    }
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        for (int env_index = begin; env_index < end; ++env_index) {
          LoadState(workspace, CompactStateRow(env_index));
          WriteState(workspace, env_index, state_out);
          if (return_sensor) {
            WriteSensorRow(workspace, env_index, sensor_data);
          }
        }
      };
      RunChunks(worker);
    }
    py::dict output;
    output["state"] = state_out;
    if (return_sensor) {
      output["sensor_data"] = sensor_data;
    }
    return output;
  }

  void SaveCompactState(int env_index, const double* state_row) {
    const auto row_start =
        compact_state_.begin() + static_cast<std::ptrdiff_t>(env_index) * state_dim_;
    std::copy(state_row, state_row + state_dim_, row_start);
  }

  const double* CompactStateRow(int env_index) const {
    return compact_state_.data() + static_cast<std::size_t>(env_index) * state_dim_;
  }

  std::unordered_map<std::string, drake::geometry::GeometryId> CollisionGeometriesByName()
      const {
    std::unordered_map<std::string, drake::geometry::GeometryId> geometry_by_name;
    const auto& inspector = scene_graph_->model_inspector();
    for (int body_index = 0; body_index < plant_->num_bodies(); ++body_index) {
      const RigidBody<double>& body = plant_->get_body(BodyIndex(body_index));
      for (const auto& geometry_id : plant_->GetCollisionGeometriesForBody(body)) {
        const std::string name = NormalizeGeometryName(inspector.GetName(geometry_id));
        if (!geometry_by_name.emplace(name, geometry_id).second) {
          throw std::runtime_error("Duplicate Drake collision geometry name after normalization: " +
                                   name);
        }
      }
    }
    return geometry_by_name;
  }

  void ApplyContactSettings() {
    plant_->set_contact_model(ContactModelFromName(physics_.contact_model));
    if (plant_->is_discrete()) {
      plant_->set_discrete_contact_approximation(
          DiscreteApproximationFromName(physics_.discrete_approximation));
    } else if (physics_.discrete_approximation != "sap") {
      throw std::invalid_argument(
          "discrete_contact_approximation requires a discrete plant (sim_dt > 0)");
    }
    plant_->set_penetration_allowance(physics_.penetration_allowance);
  }

  void ApplyGeometryMaterials() {
    if (physics_.materials.empty()) {
      return;
    }
    const std::optional<drake::geometry::SourceId> source_id = plant_->get_source_id();
    if (!source_id.has_value()) {
      throw std::runtime_error("DrakeEnvPool plant is not registered as a geometry source");
    }
    const auto collision = CollisionGeometriesByName();
    if (physics_.contact_model == "hydroelastic") {
      // Strict mode: never fall back to point contact. Every collision
      // geometry must carry an explicit material assignment.
      std::vector<std::string> missing;
      for (const auto& [name, geometry_id] : collision) {
        (void)geometry_id;
        const bool assigned = std::any_of(
            physics_.materials.begin(), physics_.materials.end(),
            [&](const NativeGeometryMaterial& material) { return material.name == name; });
        if (!assigned) {
          missing.push_back(name);
        }
      }
      if (!missing.empty()) {
        std::sort(missing.begin(), missing.end());
        std::string joined;
        for (const auto& name : missing) {
          joined += (joined.empty() ? "" : ", ") + name;
        }
        throw std::invalid_argument(
            "contact_model='hydroelastic' is strict and requires a geometry_materials "
            "assignment for every collision geometry; missing: " +
            joined);
      }
    }
    const auto& inspector = scene_graph_->model_inspector();
    for (const auto& material : physics_.materials) {
      const auto found = collision.find(material.name);
      if (found == collision.end()) {
        throw std::invalid_argument(
            "geometry_materials refer to unknown collision geometry " + material.name);
      }
      const drake::geometry::GeometryId geometry_id = found->second;
      const drake::geometry::ProximityProperties* existing =
          inspector.GetProximityProperties(geometry_id);
      if (existing == nullptr) {
        throw std::runtime_error(
            "Collision geometry " + material.name + " has no proximity role in Drake");
      }
      // Start from the parsed properties so MJCF-authored contact material
      // (e.g. Coulomb friction) is preserved, then overwrite the hydroelastic
      // fields with the requested assignment.
      drake::geometry::ProximityProperties props(*existing);
      RemovePropertyIfPresent(&props, "hydroelastic", "compliance_type");
      RemovePropertyIfPresent(&props, "hydroelastic", "hydroelastic_modulus");
      RemovePropertyIfPresent(&props, "hydroelastic", "resolution_hint");
      RemovePropertyIfPresent(&props, "hydroelastic", "slab_thickness");
      if (material.compliant) {
        drake::geometry::AddCompliantHydroelasticProperties(material.resolution_hint,
                                                            material.modulus, &props);
      } else {
        drake::geometry::AddRigidHydroelasticProperties(&props);
      }
      if (material.has_dissipation) {
        RemovePropertyIfPresent(&props, "material", "hunt_crossley_dissipation");
        drake::geometry::AddContactMaterial(material.dissipation, std::nullopt, std::nullopt,
                                            &props);
      }
      scene_graph_->AssignRole(*source_id, geometry_id, props,
                               drake::geometry::RoleAssign::kReplace);
      material_geometry_ids_.push_back({material.name, geometry_id});
    }
  }

  // Documented approximation for MJCF body gravcomp, which Drake's MJCF
  // parser ignores: a constant force -gravcomp * mass * gravity applied at
  // the body center of mass (zero torque), matching MuJoCo's semantics of
  // compensating that fraction of the body's weight.
  void CacheGravcompForces() {
    gravcomp_forces_.clear();
    if (!physics_.preserve_gravcomp || physics_.body_gravcomp.empty()) {
      return;
    }
    const Eigen::Vector3d gravity = plant_->gravity_field().gravity_vector();
    for (const auto& [name, fraction] : physics_.body_gravcomp) {
      const RigidBody<double>& body = plant_->GetBodyByName(name, model_instance_);
      const auto& inertia = body.default_spatial_inertia();
      ExternallyAppliedSpatialForce<double> applied;
      applied.body_index = body.index();
      applied.p_BoBq_B = inertia.get_com();
      applied.F_Bq_W = SpatialForce<double>(Eigen::Vector3d::Zero(),
                                            -fraction * inertia.get_mass() * gravity);
      gravcomp_forces_.push_back(applied);
    }
  }

  void WireFrictionlossBrake(DiagramBuilder<double>* builder) {
    if (!physics_.frictionloss_enabled) {
      return;
    }
    const int joint_count = static_cast<int>(joint_layout_names_.size());
    if (static_cast<int>(physics_.joint_frictionloss.size()) != joint_count) {
      throw std::invalid_argument(
          "joint_frictionloss must have one entry per MJCF joint layout entry");
    }
    auto kind = joint_layout_kind_.unchecked<1>();
    std::vector<FrictionlossEntry> entries;
    for (int i = 0; i < joint_count; ++i) {
      const double loss = physics_.joint_frictionloss.at(i);
      if (loss == 0.0) {
        continue;
      }
      if (kind(i) != kSlideJoint && kind(i) != kHingeJoint) {
        throw std::invalid_argument(
            "the frictionloss approximation only supports single-dof slide/hinge "
            "joints; nonzero frictionloss on child body " +
            joint_layout_body_names_.at(i));
      }
      const Joint<double>& joint =
          FindJointByChildBody(joint_layout_body_names_.at(i), 1, 1);
      entries.push_back(
          FrictionlossEntry{joint.velocity_start(), loss, physics_.frictionloss_speed});
    }
    if (entries.empty()) {
      return;
    }
    auto* brake = builder->AddSystem<FrictionlossBrake>(
        nq_, plant_->num_multibody_states(), nv_, std::move(entries));
    builder->Connect(plant_->get_state_output_port(), brake->get_input_port(0));
    builder->Connect(brake->get_output_port(0),
                     plant_->get_applied_generalized_force_input_port());
  }

  int ApplyMjcfCollisionFilters() {
    const auto geometry_by_name = CollisionGeometriesByName();

    int count = 0;
    for (std::size_t pair = 0; pair < collision_filter_geom_names1_.size(); ++pair) {
      const std::string& name_a = collision_filter_geom_names1_.at(pair);
      const std::string& name_b = collision_filter_geom_names2_.at(pair);
      auto geom_a = geometry_by_name.find(name_a);
      auto geom_b = geometry_by_name.find(name_b);
      if (geom_a == geometry_by_name.end() || geom_b == geometry_by_name.end()) {
        throw std::runtime_error("Could not apply MJCF collision filter for geoms " + name_a +
                                 " and " + name_b + "; geometry was not found in Drake");
      }
      scene_graph_->collision_filter_manager().Apply(
          CollisionFilterDeclaration().ExcludeBetween(GeometrySet(geom_a->second),
                                                     GeometrySet(geom_b->second)));
      ++count;
    }
    return count;
  }

  template <typename Worker>
  void RunChunks(Worker worker) {
    RunRowChunks(nbatch_, worker);
  }

  template <typename Worker>
  void RunRowChunks(int row_count, Worker worker) {
    const int thread_count = std::min(nthread_, row_count);
    if (thread_count <= 1) {
      worker(0, 0, row_count);
      return;
    }
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions(thread_count);
    threads.reserve(thread_count);
    for (int thread = 0; thread < thread_count; ++thread) {
      const int begin = thread * row_count / thread_count;
      const int end = (thread + 1) * row_count / thread_count;
      threads.emplace_back([&, thread, begin, end]() {
        try {
          worker(thread, begin, end);
        } catch (...) {
          exceptions[thread] = std::current_exception();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    for (const auto& exception : exceptions) {
      if (exception != nullptr) {
        std::rethrow_exception(exception);
      }
    }
  }

  int nbatch_{};
  double sim_dt_{};
  int nq_{};
  int nv_{};
  int nu_{};
  int state_dim_{};
  py::array_t<double> ctrl_limits_;
  py::array_t<double> torque_limits_;
  py::array_t<int> actuator_kind_;
  py::array_t<double> actuator_gear_;
  py::array_t<double> actuator_stiffness_;
  py::array_t<double> actuator_damping_;
  py::array_t<double> actuator_gainprm_;
  py::array_t<double> actuator_biasprm_;
  py::array_t<int> joint_layout_kind_;
  py::array_t<int> joint_layout_qpos_adr_;
  py::array_t<int> joint_layout_qvel_adr_;
  py::array_t<int> joint_layout_qpos_dim_;
  py::array_t<int> joint_layout_qvel_dim_;
  std::vector<std::string> joint_layout_names_;
  std::vector<std::string> joint_layout_body_names_;
  std::vector<std::string> collision_filter_geom_names1_;
  std::vector<std::string> collision_filter_geom_names2_;
  StateLayout state_layout_;
  std::vector<int> sensor_frame_body_indices_;
  py::array_t<double> sensor_frame_offsets_;
  std::vector<int> sensor_frame_ref_body_indices_;
  py::array_t<double> sensor_frame_ref_offsets_;
  py::array_t<int> sensor_type_;
  py::array_t<int> sensor_index_;
  py::array_t<int> sensor_adr_;
  py::array_t<int> sensor_dim_;
  int sensor_count_{};
  int nsensordata_{};
  int nthread_{};
  int num_filtered_geometries_{};
  NativePhysicsConfig physics_;
  std::vector<std::pair<std::string, drake::geometry::GeometryId>> material_geometry_ids_;
  std::vector<ExternallyAppliedSpatialForce<double>> gravcomp_forces_;
  std::unique_ptr<Diagram<double>> diagram_;
  MultibodyPlant<double>* plant_{};
  SceneGraph<double>* scene_graph_{};
  ModelInstanceIndex model_instance_;
  std::vector<const RigidBody<double>*> sensor_frame_bodies_;
  std::vector<const RigidBody<double>*> sensor_frame_ref_bodies_;
  std::vector<int> actuator_position_start_;
  std::vector<int> actuator_velocity_start_;
  std::vector<double> compact_state_;
  std::vector<ThreadWorkspace> workspaces_;
};

bool BatchAvailable() { return true; }

}  // namespace

PYBIND11_MODULE(_drake_env_pool, m) {
  m.doc() = "Compiled DrakeEnvPool for UniLab DrakeUni batch experiments.";
  m.def("batch_available", &BatchAvailable);
  py::class_<DrakeEnvPool>(m, "DrakeEnvPool")
      .def(py::init<const std::string&, int, double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<int>&,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::vector<int>&,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>, int, int,
                    py::object>(),
           py::arg("model_file"), py::arg("nbatch"), py::arg("sim_dt"),
           py::arg("ctrl_limits"), py::arg("torque_limits"),
           py::arg("actuator_kind"), py::arg("actuator_gear"),
           py::arg("actuator_stiffness"), py::arg("actuator_damping"),
           py::arg("actuator_gainprm"), py::arg("actuator_biasprm"),
           py::arg("joint_layout_kind"), py::arg("joint_layout_qpos_adr"),
           py::arg("joint_layout_qvel_adr"), py::arg("joint_layout_qpos_dim"),
           py::arg("joint_layout_qvel_dim"), py::arg("joint_layout_names"),
           py::arg("joint_layout_body_names"),
           py::arg("collision_filter_geom_names1"), py::arg("collision_filter_geom_names2"),
           py::arg("sensor_frame_body_indices"),
           py::arg("sensor_frame_offsets"),
           py::arg("sensor_frame_ref_body_indices"),
           py::arg("sensor_frame_ref_offsets"),
           py::arg("sensor_type"), py::arg("sensor_index"), py::arg("sensor_adr"),
           py::arg("sensor_dim"), py::arg("nsensordata"),
           py::arg("nthread") = 1, py::arg("physics") = py::none())
      .def_property_readonly("nbatch", &DrakeEnvPool::nbatch)
      .def_property_readonly("state_dim", &DrakeEnvPool::state_dim)
      .def_property_readonly("num_positions", &DrakeEnvPool::num_positions)
      .def_property_readonly("num_velocities", &DrakeEnvPool::num_velocities)
      .def_property_readonly("control_dim", &DrakeEnvPool::control_dim)
      .def_property_readonly("num_bodies", &DrakeEnvPool::num_bodies)
      .def_property_readonly("nsensordata", &DrakeEnvPool::nsensordata)
      .def_property_readonly("nthread", &DrakeEnvPool::nthread)
      .def_property_readonly("workspace_count", &DrakeEnvPool::workspace_count)
      .def_property_readonly("num_filtered_geometries",
                             &DrakeEnvPool::num_filtered_geometries)
      .def("step", &DrakeEnvPool::step, py::arg("state0"), py::arg("nstep"),
           py::arg("control"), py::arg("body_forces") = py::none(),
           py::arg("return_sensor") = false)
      .def("compute_body_state", &DrakeEnvPool::compute_body_state, py::arg("state0"),
           py::arg("body_indices"))
      .def("reset", &DrakeEnvPool::reset, py::arg("env_ids"), py::arg("initial_state"),
           py::arg("return_sensor") = false)
      .def("default_state", &DrakeEnvPool::default_state)
      .def("native_model_properties", &DrakeEnvPool::native_model_properties)
      .def("effective_physics", &DrakeEnvPool::effective_physics)
      .def("snapshot", &DrakeEnvPool::snapshot, py::arg("return_sensor") = false);
}
