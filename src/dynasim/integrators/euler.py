"""
Forward Euler integration method.

The simplest explicit integration scheme. First-order accurate.
"""

from typing import Callable, Optional

import numpy as np


def euler_step(
    f: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Perform one forward Euler integration step.

    Computes: x_{k+1} = x_k + dt * f(x_k, u_k, w_k)

    This is a first-order method with local truncation error O(dt²)
    and global error O(dt).

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(x, u, w) -> x_dot.
    x : np.ndarray, shape (n,)
        Current state vector.
    u : np.ndarray, shape (m,)
        Control input vector (held constant over the time step).
    dt : float
        Time step duration.
    w : np.ndarray, shape (p,), optional
        Disturbance vector. If None, passed as None to f.

    Returns
    -------
    x_next : np.ndarray, shape (n,)
        State at the next time step.

    Examples
    --------
    >>> def simple_dynamics(x, u, w=None):
    ...     return -x + u  # Simple stable linear system
    >>> x = np.array([1.0])
    >>> u = np.array([0.0])
    >>> x_next = euler_step(simple_dynamics, x, u, dt=0.1)
    >>> print(x_next)  # Should be 0.9
    [0.9]

    Notes
    -----
    Euler integration is fast but can be unstable for stiff systems
    or large time steps. Use RK4 for better accuracy and stability.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    x_dot = f(x, u, w)
    x_next = x + dt * x_dot

    return x_next


def euler_integrate(
    f: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    x0: np.ndarray,
    u_sequence: np.ndarray,
    dt: float,
    w_sequence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Integrate dynamics over multiple time steps using Euler method.

    Parameters
    ----------
    f : callable
        Dynamics function with signature f(x, u, w) -> x_dot.
    x0 : np.ndarray, shape (n,)
        Initial state vector.
    u_sequence : np.ndarray, shape (N, m)
        Control inputs for each time step.
    dt : float
        Time step duration.
    w_sequence : np.ndarray, shape (N, p), optional
        Disturbance vectors for each time step. If None, no disturbance.

    Returns
    -------
    x_trajectory : np.ndarray, shape (N+1, n)
        State trajectory including initial state.
    """
    N = len(u_sequence)
    n_state = len(x0)

    x_trajectory = np.zeros((N + 1, n_state))
    x_trajectory[0] = x0

    for k in range(N):
        u_k = u_sequence[k]
        w_k = w_sequence[k] if w_sequence is not None else None
        x_trajectory[k + 1] = euler_step(f, x_trajectory[k], u_k, dt, w_k)

    return x_trajectory
