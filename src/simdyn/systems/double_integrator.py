"""
Double Integrator dynamical system.

A simple linear system commonly used for testing control algorithms.
Supports 1D, 2D, and 3D configurations.

Dynamics
--------
    ṗ = v
    v̇ = u/m + w

where:
    p : position (n_dim,)
    v : velocity (n_dim,)
    u : control force/acceleration (n_dim,)
    m : mass (scalar)
    w : disturbance (2*n_dim,)

State Vector
------------
    x = [p, v] of dimension (2 * n_dim,)

    1D: x = [p, v]           (n=2)
    2D: x = [px, py, vx, vy] (n=4)
    3D: x = [px, py, pz, vx, vy, vz] (n=6)

Control Vector
--------------
    u = [force/acceleration] of dimension (n_dim,)

References
----------
- Standard linear systems textbook example
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from simdyn.base import DynamicalSystem


@dataclass
class DoubleIntegratorParams:
    """
    Parameters for the Double Integrator system.

    Attributes
    ----------
    n_dim : int
        Spatial dimension (1, 2, or 3). Default is 2.
    mass : float
        Mass of the point mass. Default is 1.0 (makes control = acceleration).
    p_min : np.ndarray or None
        Minimum position bounds. Shape (n_dim,). None for unbounded.
    p_max : np.ndarray or None
        Maximum position bounds. Shape (n_dim,). None for unbounded.
    v_min : np.ndarray or None
        Minimum velocity bounds. Shape (n_dim,). None for unbounded.
    v_max : np.ndarray or None
        Maximum velocity bounds. Shape (n_dim,). None for unbounded.
    u_min : np.ndarray or None
        Minimum control bounds. Shape (n_dim,). None for unbounded.
    u_max : np.ndarray or None
        Maximum control bounds. Shape (n_dim,). None for unbounded.
    """

    n_dim: int = 2
    mass: float = 1.0

    # State bounds (None = unbounded)
    p_min: Optional[np.ndarray] = None
    p_max: Optional[np.ndarray] = None
    v_min: Optional[np.ndarray] = None
    v_max: Optional[np.ndarray] = None

    # Control bounds (None = unbounded)
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.n_dim not in [1, 2, 3]:
            raise ValueError(f"n_dim must be 1, 2, or 3, got {self.n_dim}")
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")


def default_params_1d() -> DoubleIntegratorParams:
    """Create default parameters for 1D double integrator."""
    return DoubleIntegratorParams(
        n_dim=1,
        mass=1.0,
        p_min=np.array([-10.0]),
        p_max=np.array([10.0]),
        v_min=np.array([-5.0]),
        v_max=np.array([5.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
    )


def default_params_2d() -> DoubleIntegratorParams:
    """Create default parameters for 2D double integrator."""
    return DoubleIntegratorParams(
        n_dim=2,
        mass=1.0,
        p_min=np.array([-10.0, -10.0]),
        p_max=np.array([10.0, 10.0]),
        v_min=np.array([-5.0, -5.0]),
        v_max=np.array([5.0, 5.0]),
        u_min=np.array([-2.0, -2.0]),
        u_max=np.array([2.0, 2.0]),
    )


def default_params_3d() -> DoubleIntegratorParams:
    """Create default parameters for 3D double integrator."""
    return DoubleIntegratorParams(
        n_dim=3,
        mass=1.0,
        p_min=np.array([-10.0, -10.0, -10.0]),
        p_max=np.array([10.0, 10.0, 10.0]),
        v_min=np.array([-5.0, -5.0, -5.0]),
        v_max=np.array([5.0, 5.0, 5.0]),
        u_min=np.array([-2.0, -2.0, -2.0]),
        u_max=np.array([2.0, 2.0, 2.0]),
    )


class DoubleIntegrator(DynamicalSystem):
    """
    Double Integrator dynamical system.

    A simple point-mass system with position and velocity states,
    controlled by force/acceleration. This is a linear system,
    making it ideal for testing control algorithms.

    Parameters
    ----------
    params : DoubleIntegratorParams, optional
        System parameters. If None, uses 2D default parameters.

    Examples
    --------
    >>> # Create a 2D double integrator
    >>> system = DoubleIntegrator()
    >>> x = np.array([0.0, 0.0, 1.0, 0.0])  # at origin, moving in x
    >>> u = np.array([0.0, 1.0])  # accelerate in y
    >>> x_dot = system.f(x, u)

    >>> # Create a 1D double integrator
    >>> params = DoubleIntegratorParams(n_dim=1)
    >>> system_1d = DoubleIntegrator(params)

    >>> # Simulate with a controller
    >>> x0 = np.array([1.0, 0.0, 0.0, 0.0])
    >>> controller = lambda t, x: -0.5 * x[:2] - 0.5 * x[2:]  # PD control
    >>> t, x, u = system.simulate(x0, controller, (0, 10), dt=0.01)

    Notes
    -----
    The system is linear with dynamics:
        ẋ = A @ x + B @ u

    where A and B are constant matrices. This makes it easy to verify
    control algorithms against analytical solutions.

    For unit mass (m=1), the control u represents acceleration directly.
    """

    def __init__(self, params: Optional[DoubleIntegratorParams] = None):
        if params is None:
            params = default_params_2d()
        super().__init__(params)

        # Cache dimensions
        self._n_dim = params.n_dim
        self._n_state = 2 * params.n_dim
        self._n_control = params.n_dim

        # Pre-compute constant matrices for efficiency
        self._A = self._build_A_matrix()
        self._B = self._build_B_matrix()

    def _build_A_matrix(self) -> np.ndarray:
        """Build the constant state matrix A."""
        n = self._n_dim
        A = np.zeros((2 * n, 2 * n))
        # ṗ = v  =>  A[0:n, n:2n] = I
        A[:n, n:] = np.eye(n)
        return A

    def _build_B_matrix(self) -> np.ndarray:
        """Build the constant control matrix B."""
        n = self._n_dim
        m = self.params.mass
        B = np.zeros((2 * n, n))
        # v̇ = u/m  =>  B[n:2n, :] = I/m
        B[n:, :] = np.eye(n) / m
        return B

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector (2 * n_dim)."""
        return self._n_state

    @property
    def n_control(self) -> int:
        """Dimension of control vector (n_dim)."""
        return self._n_control

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector (2 * n_dim)."""
        return self._n_state

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        dim_names = ["x", "y", "z"][: self._n_dim]
        pos_names = [f"p_{d}" for d in dim_names]
        vel_names = [f"v_{d}" for d in dim_names]
        return pos_names + vel_names

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        dim_names = ["x", "y", "z"][: self._n_dim]
        return [f"u_{d}" for d in dim_names]

    @property
    def n_dim(self) -> int:
        """Spatial dimension (1, 2, or 3)."""
        return self._n_dim

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """
        Extract position from state vector.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector.

        Returns
        -------
        np.ndarray, shape (n_dim,)
            Position vector.
        """
        return np.asarray(x)[: self._n_dim]

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """
        Extract velocity from state vector.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector.

        Returns
        -------
        np.ndarray, shape (n_dim,)
            Velocity vector.
        """
        return np.asarray(x)[self._n_dim :]

    def pack_state(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Pack position and velocity into state vector.

        Parameters
        ----------
        position : np.ndarray, shape (n_dim,)
            Position vector.
        velocity : np.ndarray, shape (n_dim,)
            Velocity vector.

        Returns
        -------
        np.ndarray, shape (n_state,)
            State vector.
        """
        return np.concatenate([np.asarray(position), np.asarray(velocity)])

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Dynamics:
            ṗ = v + w_p
            v̇ = u/m + w_v

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector [position, velocity].
        u : np.ndarray, shape (n_control,)
            Control vector (force or acceleration if m=1).
        w : np.ndarray, shape (n_disturbance,), optional
            Disturbance vector [w_position, w_velocity].

        Returns
        -------
        np.ndarray, shape (n_state,)
            State derivative ẋ.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(self.n_disturbance) if w is None else np.asarray(w, dtype=np.float64)

        # Linear dynamics: ẋ = Ax + Bu + w
        x_dot = self._A @ x + self._B @ u + w

        return x_dot

    # =========================================================================
    # Jacobians (Constant for linear system)
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        State Jacobian: ∂f/∂x.

        For the double integrator, this is constant:
            A = [0  I]
                [0  0]

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector (unused, A is constant).
        u : np.ndarray, shape (n_control,)
            Control vector (unused, A is constant).

        Returns
        -------
        np.ndarray, shape (n_state, n_state)
            State Jacobian matrix.
        """
        return self._A.copy()

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Control Jacobian: ∂f/∂u.

        For the double integrator, this is constant:
            B = [0  ]
                [I/m]

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector (unused, B is constant).
        u : np.ndarray, shape (n_control,)
            Control vector (unused, B is constant).

        Returns
        -------
        np.ndarray, shape (n_state, n_control)
            Control Jacobian matrix.
        """
        return self._B.copy()

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state bounds [p_min, v_min], [p_max, v_max].

        Returns
        -------
        lb : np.ndarray, shape (n_state,)
            Lower bounds.
        ub : np.ndarray, shape (n_state,)
            Upper bounds.
        """
        p = self.params
        n = self._n_dim

        # Position bounds
        p_min = np.asarray(p.p_min) if p.p_min is not None else np.full(n, -np.inf)

        p_max = np.asarray(p.p_max) if p.p_max is not None else np.full(n, np.inf)

        # Velocity bounds
        v_min = np.asarray(p.v_min) if p.v_min is not None else np.full(n, -np.inf)

        v_max = np.asarray(p.v_max) if p.v_max is not None else np.full(n, np.inf)

        lb = np.concatenate([p_min, v_min])
        ub = np.concatenate([p_max, v_max])

        return lb, ub

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get control bounds.

        Returns
        -------
        lb : np.ndarray, shape (n_control,)
            Lower bounds.
        ub : np.ndarray, shape (n_control,)
            Upper bounds.
        """
        p = self.params
        n = self._n_dim

        u_min = np.asarray(p.u_min) if p.u_min is not None else np.full(n, -np.inf)

        u_max = np.asarray(p.u_max) if p.u_max is not None else np.full(n, np.inf)

        return u_min, u_max

    # =========================================================================
    # Analytical Solutions (for testing)
    # =========================================================================

    def analytical_solution(self, x0: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Compute analytical solution for constant control.

        For constant control u, the analytical solution is:
            p(t) = p0 + v0*t + 0.5*(u/m)*t²
            v(t) = v0 + (u/m)*t

        Parameters
        ----------
        x0 : np.ndarray, shape (n_state,)
            Initial state.
        u : np.ndarray, shape (n_control,)
            Constant control input.
        t : float
            Time.

        Returns
        -------
        np.ndarray, shape (n_state,)
            State at time t.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        p0 = self.get_position(x0)
        v0 = self.get_velocity(x0)
        a = u / self.params.mass

        p_t = p0 + v0 * t + 0.5 * a * t**2
        v_t = v0 + a * t

        return self.pack_state(p_t, v_t)

    def equilibrium(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium state for given control.

        For the double integrator, equilibrium only exists for u=0,
        where any position with zero velocity is an equilibrium.

        Parameters
        ----------
        u : np.ndarray, shape (n_control,), optional
            Control input. Default is zero.

        Returns
        -------
        np.ndarray, shape (n_state,)
            Equilibrium state (at origin with zero velocity).

        Raises
        ------
        ValueError
            If u is non-zero (no equilibrium exists).
        """
        u = np.zeros(self._n_dim) if u is None else np.asarray(u, dtype=np.float64)

        if not np.allclose(u, 0):
            raise ValueError(
                "Double integrator has no equilibrium for non-zero control. "
                "With constant non-zero force, the system accelerates indefinitely."
            )

        # Equilibrium: any position with zero velocity
        # Return origin as canonical equilibrium
        return np.zeros(self._n_state)

    def energy(self, x: np.ndarray) -> float:
        """
        Compute kinetic energy of the system.

        E = 0.5 * m * ||v||²

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector.

        Returns
        -------
        float
            Kinetic energy.
        """
        v = self.get_velocity(x)
        return 0.5 * self.params.mass * np.dot(v, v)

    # =========================================================================
    # Discrete-Time System Matrices
    # =========================================================================

    def discrete_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact discrete-time system matrices.

        For the double integrator, the exact discretization is:
            x_{k+1} = A_d @ x_k + B_d @ u_k

        where:
            A_d = [I   dt*I]    B_d = [0.5*dt²/m * I]
                  [0   I   ]          [dt/m * I     ]

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        A_d : np.ndarray, shape (n_state, n_state)
            Discrete-time state matrix.
        B_d : np.ndarray, shape (n_state, n_control)
            Discrete-time control matrix.
        """
        n = self._n_dim
        m = self.params.mass

        # Discrete state matrix
        A_d = np.eye(2 * n)
        A_d[:n, n:] = dt * np.eye(n)

        # Discrete control matrix
        B_d = np.zeros((2 * n, n))
        B_d[:n, :] = 0.5 * dt**2 / m * np.eye(n)
        B_d[n:, :] = dt / m * np.eye(n)

        return A_d, B_d


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def DoubleIntegrator1D(
    mass: float = 1.0,
    p_bounds: Tuple[float, float] = (-10.0, 10.0),
    v_bounds: Tuple[float, float] = (-5.0, 5.0),
    u_bounds: Tuple[float, float] = (-2.0, 2.0),
) -> DoubleIntegrator:
    """
    Create a 1D double integrator with specified bounds.

    Parameters
    ----------
    mass : float
        Mass of the point. Default 1.0.
    p_bounds : tuple
        (min, max) position bounds.
    v_bounds : tuple
        (min, max) velocity bounds.
    u_bounds : tuple
        (min, max) control bounds.

    Returns
    -------
    DoubleIntegrator
        Configured 1D system.
    """
    params = DoubleIntegratorParams(
        n_dim=1,
        mass=mass,
        p_min=np.array([p_bounds[0]]),
        p_max=np.array([p_bounds[1]]),
        v_min=np.array([v_bounds[0]]),
        v_max=np.array([v_bounds[1]]),
        u_min=np.array([u_bounds[0]]),
        u_max=np.array([u_bounds[1]]),
    )
    return DoubleIntegrator(params)


def DoubleIntegrator2D(
    mass: float = 1.0,
    p_bounds: Tuple[float, float] = (-10.0, 10.0),
    v_bounds: Tuple[float, float] = (-5.0, 5.0),
    u_bounds: Tuple[float, float] = (-2.0, 2.0),
) -> DoubleIntegrator:
    """
    Create a 2D double integrator with specified bounds.

    Parameters
    ----------
    mass : float
        Mass of the point. Default 1.0.
    p_bounds : tuple
        (min, max) position bounds (same for x and y).
    v_bounds : tuple
        (min, max) velocity bounds (same for x and y).
    u_bounds : tuple
        (min, max) control bounds (same for x and y).

    Returns
    -------
    DoubleIntegrator
        Configured 2D system.
    """
    params = DoubleIntegratorParams(
        n_dim=2,
        mass=mass,
        p_min=np.array([p_bounds[0], p_bounds[0]]),
        p_max=np.array([p_bounds[1], p_bounds[1]]),
        v_min=np.array([v_bounds[0], v_bounds[0]]),
        v_max=np.array([v_bounds[1], v_bounds[1]]),
        u_min=np.array([u_bounds[0], u_bounds[0]]),
        u_max=np.array([u_bounds[1], u_bounds[1]]),
    )
    return DoubleIntegrator(params)


def DoubleIntegrator3D(
    mass: float = 1.0,
    p_bounds: Tuple[float, float] = (-10.0, 10.0),
    v_bounds: Tuple[float, float] = (-5.0, 5.0),
    u_bounds: Tuple[float, float] = (-2.0, 2.0),
) -> DoubleIntegrator:
    """
    Create a 3D double integrator with specified bounds.

    Parameters
    ----------
    mass : float
        Mass of the point. Default 1.0.
    p_bounds : tuple
        (min, max) position bounds (same for all axes).
    v_bounds : tuple
        (min, max) velocity bounds (same for all axes).
    u_bounds : tuple
        (min, max) control bounds (same for all axes).

    Returns
    -------
    DoubleIntegrator
        Configured 3D system.
    """
    params = DoubleIntegratorParams(
        n_dim=3,
        mass=mass,
        p_min=np.array([p_bounds[0]] * 3),
        p_max=np.array([p_bounds[1]] * 3),
        v_min=np.array([v_bounds[0]] * 3),
        v_max=np.array([v_bounds[1]] * 3),
        u_min=np.array([u_bounds[0]] * 3),
        u_max=np.array([u_bounds[1]] * 3),
    )
    return DoubleIntegrator(params)
