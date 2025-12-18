"""
Numerical integration methods for dynamical systems.

This module provides various integration schemes for propagating
continuous-time dynamics in discrete time steps.

Available Methods
-----------------
euler : Forward Euler (1st order)
    Fast but less accurate. Good for testing or when speed is critical.

rk4 : 4th-order Runge-Kutta
    Excellent balance of accuracy and speed. Recommended default.

Usage
-----
>>> from simdyn.integrators import rk4_step, euler_step
>>> x_next = rk4_step(dynamics_fn, x, u, dt)

Or use the integrator registry:
>>> from simdyn.integrators import get_integrator
>>> step_fn = get_integrator('rk4')
>>> x_next = step_fn(dynamics_fn, x, u, dt)
"""

from typing import Callable

from .euler import euler_integrate, euler_step
from .rk4 import rk4_discretize_jacobians, rk4_integrate, rk4_step

# Registry mapping method names to step functions
INTEGRATOR_REGISTRY = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def get_integrator(method: str) -> Callable:
    """
    Get an integrator step function by name.

    Parameters
    ----------
    method : str
        Name of the integration method. Options: 'euler', 'rk4'.

    Returns
    -------
    callable
        The step function with signature f(dynamics, x, u, dt, w) -> x_next.

    Raises
    ------
    ValueError
        If the method name is not recognized.

    Examples
    --------
    >>> step_fn = get_integrator('rk4')
    >>> x_next = step_fn(my_dynamics, x, u, dt=0.01)
    """
    if method not in INTEGRATOR_REGISTRY:
        available = ", ".join(INTEGRATOR_REGISTRY.keys())
        raise ValueError(f"Unknown integrator '{method}'. Available: {available}")

    return INTEGRATOR_REGISTRY[method]


def list_integrators() -> list:
    """
    List all available integration methods.

    Returns
    -------
    list of str
        Names of available integrators.
    """
    return list(INTEGRATOR_REGISTRY.keys())


__all__ = [
    "INTEGRATOR_REGISTRY",
    # Trajectory integration
    "euler_integrate",
    # Step functions
    "euler_step",
    # Registry
    "get_integrator",
    "list_integrators",
    # Utilities
    "rk4_discretize_jacobians",
    "rk4_integrate",
    "rk4_step",
]
