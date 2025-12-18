"""
Dynamical systems implementations.

This module contains concrete implementations of various dynamical systems
for robotics and aerospace applications.

Available Systems
-----------------
DoubleIntegrator : Simple linear point-mass system (1D/2D/3D)
Unicycle : Nonholonomic 2D mobile robot model
Rocket3DoF : Point-mass rocket for powered descent
Rocket6DoF : Rigid-body rocket with attitude dynamics
Pendulum : Simple pendulum (classic nonlinear benchmark)
CartPole : Cart-pole / inverted pendulum (underactuated benchmark)
"""

from simdyn.systems.cartpole import (
    CartPole,
    CartPoleParams,
    create_cartpole,
    create_gym_cartpole,
    create_normalized_cartpole,
)
from simdyn.systems.double_integrator import (
    DoubleIntegrator,
    DoubleIntegrator1D,
    DoubleIntegrator2D,
    DoubleIntegrator3D,
    DoubleIntegratorParams,
    default_params_1d,
    default_params_2d,
    default_params_3d,
)
from simdyn.systems.pendulum import (
    Pendulum,
    PendulumParams,
    create_damped_pendulum,
    create_normalized_pendulum,
    create_pendulum,
)
from simdyn.systems.rocket3dof import (
    Rocket3DoF,
    Rocket3DoFParams,
    create_normalized_rocket3dof,
    create_rocket3dof,
)
from simdyn.systems.rocket3dof import (
    normalized_params as rocket3dof_normalized_params,
)
from simdyn.systems.rocket6dof import (
    Rocket6DoF,
    Rocket6DoFParams,
    create_rocket6dof,
    create_szmuk_rocket6dof,
)
from simdyn.systems.rocket6dof import (
    szmuk_params as rocket6dof_szmuk_params,
)
from simdyn.systems.unicycle import (
    Unicycle,
    UnicycleParams,
    create_unicycle,
    forward_only_unicycle,
)

__all__ = [
    # Cart-Pole
    "CartPole",
    "CartPoleParams",
    # Double Integrator
    "DoubleIntegrator",
    "DoubleIntegrator1D",
    "DoubleIntegrator2D",
    "DoubleIntegrator3D",
    "DoubleIntegratorParams",
    # Pendulum
    "Pendulum",
    "PendulumParams",
    # Rocket 3-DoF
    "Rocket3DoF",
    "Rocket3DoFParams",
    # Rocket 6-DoF
    "Rocket6DoF",
    "Rocket6DoFParams",
    # Unicycle
    "Unicycle",
    "UnicycleParams",
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
    "default_params_1d",
    "default_params_2d",
    "default_params_3d",
    "forward_only_unicycle",
    "rocket3dof_normalized_params",
    "rocket6dof_szmuk_params",
]
