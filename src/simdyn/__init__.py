"""
simdyn - A Modular Dynamics Library for Control Systems Research.

This library provides standardized implementations of dynamical systems
commonly used in robotics and aerospace control research.

Visualization utilities are available in the `simdyn.visualization` submodule:
    from simdyn.visualization import plot_states, animate_pendulum
"""

__version__ = "0.1.0"

from simdyn.base import DynamicalSystem
from simdyn.integrators import (
    euler_integrate,
    euler_step,
    get_integrator,
    list_integrators,
    rk4_integrate,
    rk4_step,
)
from simdyn.systems import (
    # Cart-Pole
    CartPole,
    CartPoleParams,
    # Double Integrator
    DoubleIntegrator,
    DoubleIntegrator1D,
    DoubleIntegrator2D,
    DoubleIntegrator3D,
    DoubleIntegratorParams,
    # Pendulum
    Pendulum,
    PendulumParams,
    # Rocket 3-DoF
    Rocket3DoF,
    Rocket3DoFParams,
    # Rocket 6-DoF
    Rocket6DoF,
    Rocket6DoFParams,
    # Unicycle
    Unicycle,
    UnicycleParams,
    create_cartpole,
    create_damped_pendulum,
    create_gym_cartpole,
    create_normalized_cartpole,
    create_normalized_pendulum,
    create_normalized_rocket3dof,
    create_pendulum,
    create_rocket3dof,
    create_rocket6dof,
    create_szmuk_rocket6dof,
    create_unicycle,
    forward_only_unicycle,
)
from simdyn.utils import (
    angle_difference,
    angular_velocity_to_euler_rates,
    dcm_angle,
    dcm_axis,
    dcm_from_axis_angle,
    dcm_from_two_vectors,
    dcm_is_valid,
    dcm_normalize,
    dcm_to_euler,
    dcm_to_quat,
    euler_rates_to_angular_velocity,
    euler_to_dcm,
    euler_to_quat,
    omega_matrix,
    quat_angle,
    quat_axis,
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_from_axis_angle,
    quat_identity,
    quat_integrate,
    quat_integrate_exact,
    quat_inverse,
    # Quaternion operations
    quat_multiply,
    quat_normalize,
    quat_random,
    quat_rotate,
    quat_rotate_inverse,
    quat_slerp,
    quat_to_dcm,
    quat_to_euler,
    # Rotation utilities
    rotx,
    roty,
    rotz,
    skew,
    unskew,
    wrap_angle,
    wrap_angle_positive,
)

__all__ = [
    # Systems - Cart-Pole
    "CartPole",
    "CartPoleParams",
    # Systems - Double Integrator
    "DoubleIntegrator",
    "DoubleIntegrator1D",
    "DoubleIntegrator2D",
    "DoubleIntegrator3D",
    "DoubleIntegratorParams",
    # Core
    "DynamicalSystem",
    # Systems - Pendulum
    "Pendulum",
    "PendulumParams",
    # Systems - Rocket 3-DoF
    "Rocket3DoF",
    "Rocket3DoFParams",
    # Systems - Rocket 6-DoF
    "Rocket6DoF",
    "Rocket6DoFParams",
    # Systems - Unicycle
    "Unicycle",
    "UnicycleParams",
    "__version__",
    "angle_difference",
    "angular_velocity_to_euler_rates",
    "create_cartpole",
    "create_damped_pendulum",
    "create_gym_cartpole",
    "create_normalized_cartpole",
    "create_normalized_pendulum",
    "create_normalized_rocket3dof",
    "create_pendulum",
    "create_rocket3dof",
    "create_rocket6dof",
    "create_szmuk_rocket6dof",
    "create_unicycle",
    "dcm_angle",
    "dcm_axis",
    "dcm_from_axis_angle",
    "dcm_from_two_vectors",
    "dcm_is_valid",
    "dcm_normalize",
    "dcm_to_euler",
    "dcm_to_quat",
    "euler_integrate",
    "euler_rates_to_angular_velocity",
    # Integrators
    "euler_step",
    "euler_to_dcm",
    "euler_to_quat",
    "forward_only_unicycle",
    "get_integrator",
    "list_integrators",
    "omega_matrix",
    "quat_angle",
    "quat_axis",
    "quat_conjugate",
    "quat_derivative",
    "quat_error",
    "quat_from_axis_angle",
    "quat_identity",
    "quat_integrate",
    "quat_integrate_exact",
    "quat_inverse",
    # Quaternion operations
    "quat_multiply",
    "quat_normalize",
    "quat_random",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_slerp",
    "quat_to_dcm",
    "quat_to_euler",
    "rk4_integrate",
    "rk4_step",
    # Rotation utilities
    "rotx",
    "roty",
    "rotz",
    "skew",
    "unskew",
    "wrap_angle",
    "wrap_angle_positive",
]
