"""
4th-order Runge-Kutta integration method.

A classic explicit integration scheme with good accuracy and stability.
Fourth-order accurate.
"""

from typing import Callable, Optional

import numpy as np


def rk4_step(
    f: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Perform one 4th-order Runge-Kutta integration step.

    Computes:
        k1 = f(x, u, w)
        k2 = f(x + dt/2 * k1, u, w)
        k3 = f(x + dt/2 * k2, u, w)
        k4 = f(x + dt * k3, u, w)
        x_{k+1} = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    This is a fourth-order method with local truncation error O(dt⁵)
    and global error O(dt⁴).

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
    >>> x_next = rk4_step(simple_dynamics, x, u, dt=0.1)
    >>> # Analytical: x(0.1) = exp(-0.1) ≈ 0.9048
    >>> print(f"{x_next[0]:.4f}")
    0.9048

    Notes
    -----
    RK4 is the workhorse of numerical integration. It provides excellent
    accuracy for most systems while being computationally efficient.

    The control u and disturbance w are held constant over the time step
    (zero-order hold assumption). This is standard for discrete-time
    control systems.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    # Compute the four stages
    k1 = f(x, u, w)
    k2 = f(x + 0.5 * dt * k1, u, w)
    k3 = f(x + 0.5 * dt * k2, u, w)
    k4 = f(x + dt * k3, u, w)

    # Combine stages
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_next


def rk4_integrate(
    f: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
    x0: np.ndarray,
    u_sequence: np.ndarray,
    dt: float,
    w_sequence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Integrate dynamics over multiple time steps using RK4 method.

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

    Examples
    --------
    >>> def pendulum(x, u, w=None):
    ...     theta, theta_dot = x
    ...     return np.array([theta_dot, -np.sin(theta) + u[0]])
    >>> x0 = np.array([0.1, 0.0])  # Small initial angle
    >>> u_seq = np.zeros((100, 1))  # No control
    >>> traj = rk4_integrate(pendulum, x0, u_seq, dt=0.01)
    >>> print(traj.shape)  # (101, 2)
    """
    N = len(u_sequence)
    n_state = len(x0)

    x_trajectory = np.zeros((N + 1, n_state))
    x_trajectory[0] = x0

    for k in range(N):
        u_k = u_sequence[k]
        w_k = w_sequence[k] if w_sequence is not None else None
        x_trajectory[k + 1] = rk4_step(f, x_trajectory[k], u_k, dt, w_k)

    return x_trajectory


def rk4_discretize_jacobians(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple:
    """
    Discretize continuous-time Jacobians using RK4-consistent approximation.

    For a linear system ẋ = Ax + Bu, this computes the discrete-time
    matrices A_d, B_d such that x_{k+1} = A_d x_k + B_d u_k.

    This uses the approximation consistent with RK4 integration:
        A_d ≈ I + dt*A + (dt²/2)*A² + (dt³/6)*A³ + (dt⁴/24)*A⁴
        B_d ≈ dt * (I + dt/2*A + (dt²/6)*A² + (dt³/24)*A³) * B

    Parameters
    ----------
    A_c : np.ndarray, shape (n, n)
        Continuous-time state matrix.
    B_c : np.ndarray, shape (n, m)
        Continuous-time input matrix.
    dt : float
        Time step.

    Returns
    -------
    A_d : np.ndarray, shape (n, n)
        Discrete-time state matrix.
    B_d : np.ndarray, shape (n, m)
        Discrete-time input matrix.

    Notes
    -----
    For more accurate discretization of linear systems, consider using
    scipy.linalg.expm for the exact matrix exponential solution.
    """
    n = A_c.shape[0]
    I = np.eye(n)

    A2 = A_c @ A_c
    A3 = A2 @ A_c
    A4 = A3 @ A_c

    # State matrix discretization (Taylor expansion of matrix exponential)
    A_d = I + dt * A_c + (dt**2 / 2) * A2 + (dt**3 / 6) * A3 + (dt**4 / 24) * A4

    # Input matrix discretization
    B_int = I + (dt / 2) * A_c + (dt**2 / 6) * A2 + (dt**3 / 24) * A3
    B_d = dt * B_int @ B_c

    return A_d, B_d
