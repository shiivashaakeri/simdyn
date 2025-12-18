"""
Simple Pendulum dynamical system.

A classic nonlinear benchmark system for control algorithm testing.
Models a point mass on a massless rod with optional damping.

Dynamics
--------
    θ̇ = ω
    ω̇ = -(g/l)·sin(θ) - (b/(m·l²))·ω + τ/(m·l²) + w

where:
    θ : angle from vertical (down position = 0)
    ω : angular velocity
    g : gravitational acceleration
    l : pendulum length
    m : point mass
    b : damping coefficient
    τ : applied torque at pivot

State Vector (n=2)
------------------
    x = [θ, ω]

    Index 0: angle [rad] (0 = hanging down, π = upright)
    Index 1: angular velocity [rad/s]

Control Vector (m=1)
--------------------
    u = [τ] - torque applied at pivot

Angle Convention
----------------
    θ = 0: pendulum hanging straight down (stable equilibrium)
    θ = π: pendulum pointing straight up (unstable equilibrium)
    θ > 0: counter-clockwise from down position

    This convention means potential energy is:
    PE = m·g·l·(1 - cos(θ))
    which is 0 at bottom and 2·m·g·l at top.

References
----------
- Underactuated Robotics (Tedrake) - Chapter on Pendulum
- Nonlinear Dynamics and Chaos (Strogatz)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from simdyn.base import DynamicalSystem
from simdyn.utils.rotations import wrap_angle

# =============================================================================
# Constants
# =============================================================================

G_DEFAULT = 9.81  # Standard gravity [m/s²]


# =============================================================================
# Parameter Dataclass
# =============================================================================


@dataclass
class PendulumParams:
    """
    Parameters for the Simple Pendulum system.

    Attributes
    ----------
    m : float
        Point mass [kg]. Default 1.0.
    l : float
        Pendulum length [m]. Default 1.0.
    g : float
        Gravitational acceleration [m/s²]. Default 9.81.
    b : float
        Damping coefficient [N·m·s/rad]. Default 0.0 (no damping).
    tau_max : float
        Maximum torque magnitude [N·m]. Default inf (unlimited).
    omega_max : float
        Maximum angular velocity [rad/s]. Default inf.
    """

    m: float = 1.0
    l: float = 1.0
    g: float = G_DEFAULT
    b: float = 0.0  # No damping by default
    tau_max: float = np.inf
    omega_max: float = np.inf

    def __post_init__(self):
        """Validate parameters."""
        if self.m <= 0:
            raise ValueError(f"Mass must be positive, got {self.m}")
        if self.l <= 0:
            raise ValueError(f"Length must be positive, got {self.l}")
        if self.g < 0:
            raise ValueError(f"Gravity must be non-negative, got {self.g}")
        if self.b < 0:
            raise ValueError(f"Damping must be non-negative, got {self.b}")
        if self.tau_max <= 0:
            raise ValueError(f"tau_max must be positive, got {self.tau_max}")
        if self.omega_max <= 0:
            raise ValueError(f"omega_max must be positive, got {self.omega_max}")

    @property
    def inertia(self) -> float:
        """Moment of inertia about pivot: I = m·l²."""
        return self.m * self.l**2

    @property
    def natural_frequency(self) -> float:
        """Natural frequency for small oscillations: ω_n = √(g/l)."""
        return np.sqrt(self.g / self.l)

    @property
    def period(self) -> float:
        """Period for small oscillations: T = 2π/ω_n."""
        if self.g == 0:
            return np.inf
        return 2 * np.pi / self.natural_frequency

    @property
    def damping_ratio(self) -> float:
        """Damping ratio: ζ = b / (2·m·l²·ω_n)."""
        if self.g == 0:
            return np.inf if self.b > 0 else 0.0
        omega_n = self.natural_frequency
        return self.b / (2 * self.inertia * omega_n)


def default_params() -> PendulumParams:
    """Create default pendulum parameters."""
    return PendulumParams()


def normalized_params() -> PendulumParams:
    """
    Create normalized (non-dimensional) parameters.

    With m=1, l=1, g=1, the natural frequency is 1 rad/s
    and the period is 2π seconds.
    """
    return PendulumParams(m=1.0, l=1.0, g=1.0, b=0.0)


# =============================================================================
# Main System Class
# =============================================================================


class Pendulum(DynamicalSystem):
    """
    Simple Pendulum dynamical system.

    A point mass on a massless rod, pivoting about a fixed point.
    Classic nonlinear system used for testing control algorithms,
    especially swing-up and stabilization.

    Parameters
    ----------
    params : PendulumParams, optional
        System parameters. If None, uses default parameters.

    Examples
    --------
    >>> # Create default pendulum
    >>> pendulum = Pendulum()
    >>>
    >>> # Initial state: slightly displaced from down position
    >>> x0 = np.array([0.1, 0.0])  # 0.1 rad from vertical, at rest
    >>>
    >>> # Free oscillation (no control)
    >>> controller = lambda t, x: np.array([0.0])
    >>> t, x, u = pendulum.simulate(x0, controller, (0, 10), dt=0.01)

    >>> # Swing-up example
    >>> x0 = np.array([0.0, 0.0])  # hanging down
    >>> # Energy-based swing-up controller
    >>> def swing_up(t, x):
    ...     E = pendulum.energy(x)['total']
    ...     E_target = pendulum.energy_at_top()
    ...     return np.array([np.sign(x[1]) * (E - E_target)])

    Notes
    -----
    The system has two equilibrium points:
    - θ = 0 (down): stable
    - θ = π (up): unstable

    Energy-based control is often used for swing-up, followed by
    LQR or other linear control for stabilization near the top.
    """

    def __init__(self, params: Optional[PendulumParams] = None):
        if params is None:
            params = default_params()
        super().__init__(params)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector [θ, ω]."""
        return 2

    @property
    def n_control(self) -> int:
        """Dimension of control vector [τ]."""
        return 1

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector."""
        return 2

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        return ["theta", "omega"]

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        return ["tau"]

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_angle(self, x: np.ndarray) -> float:
        """Extract angle from state vector."""
        return float(np.asarray(x)[0])

    def get_angular_velocity(self, x: np.ndarray) -> float:
        """Extract angular velocity from state vector."""
        return float(np.asarray(x)[1])

    def pack_state(self, theta: float, omega: float) -> np.ndarray:
        """
        Pack angle and angular velocity into state vector.

        Parameters
        ----------
        theta : float
            Angle from vertical [rad].
        omega : float
            Angular velocity [rad/s].

        Returns
        -------
        np.ndarray, shape (2,)
            State vector [θ, ω].
        """
        return np.array([theta, omega])

    def normalize_angle(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state by wrapping angle to [-π, π].

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            State with wrapped angle.
        """
        x = np.asarray(x, dtype=np.float64).copy()
        x[0] = wrap_angle(x[0])
        return x

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """
        Get Cartesian position of pendulum bob.

        Returns position in 2D plane with origin at pivot,
        x-axis horizontal (right positive), y-axis vertical (up positive).

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            Position [x, y] of pendulum bob.
        """
        theta = self.get_angle(x)
        l = self.params.l
        # θ=0 is down, so position is:
        # x = l·sin(θ), y = -l·cos(θ)
        return np.array([l * np.sin(theta), -l * np.cos(theta)])

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """
        Get Cartesian velocity of pendulum bob.

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            Velocity [vx, vy] of pendulum bob.
        """
        theta = self.get_angle(x)
        omega = self.get_angular_velocity(x)
        l = self.params.l
        # v = l·ω · [cos(θ), sin(θ)]
        return l * omega * np.array([np.cos(theta), np.sin(theta)])

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Dynamics:
            θ̇ = ω
            ω̇ = -(g/l)·sin(θ) - (b/(m·l²))·ω + τ/(m·l²)

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector [θ, ω].
        u : np.ndarray, shape (1,)
            Control vector [τ].
        w : np.ndarray, shape (2,), optional
            Disturbance vector.

        Returns
        -------
        np.ndarray, shape (2,)
            State derivative [θ̇, ω̇].
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(2) if w is None else np.asarray(w, dtype=np.float64)

        theta = x[0]
        omega = x[1]
        tau = u[0]

        p = self.params
        _, l, g, b = p.m, p.l, p.g, p.b
        I = p.inertia  # m * l²

        # θ̇ = ω
        theta_dot = omega + w[0]

        # ω̇ = -(g/l)·sin(θ) - (b/I)·ω + τ/I
        omega_dot = -(g / l) * np.sin(theta) - (b / I) * omega + tau / I + w[1]

        return np.array([theta_dot, omega_dot])

    # =========================================================================
    # Jacobians
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        State Jacobian: ∂f/∂x.

        A = [0           1        ]
            [-(g/l)cos(θ)  -b/(ml²)]

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.
        u : np.ndarray, shape (1,)
            Control vector (unused for A).

        Returns
        -------
        np.ndarray, shape (2, 2)
            State Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)

        theta = x[0]

        p = self.params
        g, l, b, I = p.g, p.l, p.b, p.inertia

        A = np.array([[0.0, 1.0], [-(g / l) * np.cos(theta), -b / I]])

        return A

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Control Jacobian: ∂f/∂u.

        B = [0    ]
            [1/(ml²)]

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector (unused for B).
        u : np.ndarray, shape (1,)
            Control vector (unused for B).

        Returns
        -------
        np.ndarray, shape (2, 1)
            Control Jacobian matrix.
        """
        I = self.params.inertia

        B = np.array([[0.0], [1.0 / I]])

        return B

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state bounds.

        Note: θ can take any value (wraps around).

        Returns
        -------
        lb : np.ndarray, shape (2,)
            Lower bounds.
        ub : np.ndarray, shape (2,)
            Upper bounds.
        """
        p = self.params
        lb = np.array([-np.inf, -p.omega_max])
        ub = np.array([np.inf, p.omega_max])
        return lb, ub

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get control bounds.

        Returns
        -------
        lb : np.ndarray, shape (1,)
            Lower bounds.
        ub : np.ndarray, shape (1,)
            Upper bounds.
        """
        tau_max = self.params.tau_max
        lb = np.array([-tau_max])
        ub = np.array([tau_max])
        return lb, ub

    # =========================================================================
    # Energy Methods
    # =========================================================================

    def energy(self, x: np.ndarray) -> dict:
        """
        Compute kinetic and potential energy.

        Potential energy reference: PE = 0 at bottom (θ = 0).

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.

        Returns
        -------
        dict
            Dictionary with 'kinetic', 'potential', and 'total' energy.
        """
        theta = self.get_angle(x)
        omega = self.get_angular_velocity(x)

        p = self.params
        m, l, g, I = p.m, p.l, p.g, p.inertia

        # Kinetic energy: KE = (1/2)·I·ω² = (1/2)·m·l²·ω²
        KE = 0.5 * I * omega**2

        # Potential energy: PE = m·g·l·(1 - cos(θ))
        # PE = 0 at bottom, PE = 2·m·g·l at top
        PE = m * g * l * (1 - np.cos(theta))

        return {"kinetic": KE, "potential": PE, "total": KE + PE}

    def energy_at_top(self) -> float:
        """
        Compute total energy when pendulum is at top (θ = π, ω = 0).

        Returns
        -------
        float
            Energy at upright equilibrium: 2·m·g·l.
        """
        p = self.params
        return 2 * p.m * p.g * p.l

    def energy_at_bottom(self) -> float:
        """
        Compute total energy when pendulum is at bottom (θ = 0, ω = 0).

        Returns
        -------
        float
            Energy at bottom equilibrium: 0.
        """
        return 0.0

    # =========================================================================
    # Equilibrium and Linearization
    # =========================================================================

    def equilibrium(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium state(s) for given control.

        With τ = 0:
        - θ = 0 (down): stable equilibrium
        - θ = π (up): unstable equilibrium

        With τ ≠ 0:
        - Equilibrium at θ = arcsin(τ / (m·g·l)) if |τ| ≤ m·g·l

        Parameters
        ----------
        u : np.ndarray, shape (1,), optional
            Control input. Default is zero.

        Returns
        -------
        np.ndarray, shape (2,)
            Equilibrium state [θ_eq, 0].
            Returns the stable (down) equilibrium for τ = 0.
        """
        u = np.zeros(1) if u is None else np.asarray(u, dtype=np.float64)

        tau = u[0]
        p = self.params

        if np.abs(tau) < 1e-10:
            # No torque: return down position
            return np.array([0.0, 0.0])

        # With torque: (g/l)·sin(θ) = τ/(m·l²)
        # sin(θ) = τ/(m·g·l)
        max_torque = p.m * p.g * p.l

        if np.abs(tau) > max_torque:
            raise ValueError(f"No equilibrium exists for |τ| > m·g·l = {max_torque:.3f}. Got τ = {tau:.3f}")

        theta_eq = np.arcsin(tau / max_torque)
        return np.array([theta_eq, 0.0])

    def linearize_at_bottom(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize around the stable (down) equilibrium.

        Returns
        -------
        A : np.ndarray, shape (2, 2)
            Linearized state matrix.
        B : np.ndarray, shape (2, 1)
            Linearized control matrix.
        """
        x0 = np.array([0.0, 0.0])
        u0 = np.array([0.0])
        return self.A(x0, u0), self.B(x0, u0)

    def linearize_at_top(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize around the unstable (up) equilibrium.

        Returns
        -------
        A : np.ndarray, shape (2, 2)
            Linearized state matrix.
        B : np.ndarray, shape (2, 1)
            Linearized control matrix.
        """
        x0 = np.array([np.pi, 0.0])
        u0 = np.array([0.0])
        return self.A(x0, u0), self.B(x0, u0)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_upright(self, x: np.ndarray, tol: float = 0.1) -> bool:
        """
        Check if pendulum is approximately upright.

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.
        tol : float
            Tolerance in radians.

        Returns
        -------
        bool
            True if |θ - π| < tol (or |θ + π| < tol for wrapped angles).
        """
        theta = self.get_angle(x)
        theta_wrapped = wrap_angle(theta)
        return np.abs(np.abs(theta_wrapped) - np.pi) < tol

    def is_at_bottom(self, x: np.ndarray, tol: float = 0.1) -> bool:
        """
        Check if pendulum is approximately at bottom.

        Parameters
        ----------
        x : np.ndarray, shape (2,)
            State vector.
        tol : float
            Tolerance in radians.

        Returns
        -------
        bool
            True if |θ| < tol.
        """
        theta = self.get_angle(x)
        theta_wrapped = wrap_angle(theta)
        return np.abs(theta_wrapped) < tol

    def phase_portrait_vector_field(
        self,
        theta_range: Tuple[float, float] = (-np.pi, np.pi),
        omega_range: Tuple[float, float] = (-5, 5),
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vector field for phase portrait visualization.

        Parameters
        ----------
        theta_range : tuple
            (min, max) for θ axis.
        omega_range : tuple
            (min, max) for ω axis.
        n_points : int
            Number of grid points per axis.

        Returns
        -------
        Theta : np.ndarray
            θ grid values.
        Omega : np.ndarray
            ω grid values.
        dTheta : np.ndarray
            θ̇ at each grid point.
        dOmega : np.ndarray
            ω̇ at each grid point.
        """
        theta_vals = np.linspace(theta_range[0], theta_range[1], n_points)
        omega_vals = np.linspace(omega_range[0], omega_range[1], n_points)

        Theta, Omega = np.meshgrid(theta_vals, omega_vals)
        dTheta = np.zeros_like(Theta)
        dOmega = np.zeros_like(Omega)

        u = np.array([0.0])

        for i in range(n_points):
            for j in range(n_points):
                x = np.array([Theta[i, j], Omega[i, j]])
                x_dot = self.f(x, u)
                dTheta[i, j] = x_dot[0]
                dOmega[i, j] = x_dot[1]

        return Theta, Omega, dTheta, dOmega


# =============================================================================
# Factory Functions
# =============================================================================


def create_pendulum(
    m: float = 1.0,
    l: float = 1.0,
    g: float = G_DEFAULT,
    b: float = 0.0,
    tau_max: float = np.inf,
) -> Pendulum:
    """
    Create a pendulum with specified parameters.

    Parameters
    ----------
    m : float
        Point mass [kg].
    l : float
        Pendulum length [m].
    g : float
        Gravitational acceleration [m/s²].
    b : float
        Damping coefficient.
    tau_max : float
        Maximum torque.

    Returns
    -------
    Pendulum
        Configured pendulum system.
    """
    params = PendulumParams(m=m, l=l, g=g, b=b, tau_max=tau_max)
    return Pendulum(params)


def create_damped_pendulum(
    damping_ratio: float = 0.1,
    m: float = 1.0,
    l: float = 1.0,
    g: float = G_DEFAULT,
) -> Pendulum:
    """
    Create a pendulum with specified damping ratio.

    Parameters
    ----------
    damping_ratio : float
        Damping ratio ζ (0 = undamped, 1 = critically damped).
    m : float
        Point mass.
    l : float
        Pendulum length.
    g : float
        Gravitational acceleration.

    Returns
    -------
    Pendulum
        Damped pendulum system.
    """
    # ζ = b / (2·m·l²·ω_n), where ω_n = √(g/l)
    # b = 2·ζ·m·l²·ω_n = 2·ζ·m·l²·√(g/l) = 2·ζ·m·l·√(g·l)
    omega_n = np.sqrt(g / l)
    I = m * l**2
    b = 2 * damping_ratio * I * omega_n

    params = PendulumParams(m=m, l=l, g=g, b=b)
    return Pendulum(params)


def create_normalized_pendulum() -> Pendulum:
    """
    Create a pendulum with normalized (non-dimensional) parameters.

    With m=1, l=1, g=1:
    - Natural frequency: ω_n = 1 rad/s
    - Period: T = 2π s

    Returns
    -------
    Pendulum
        Normalized pendulum system.
    """
    return Pendulum(normalized_params())
