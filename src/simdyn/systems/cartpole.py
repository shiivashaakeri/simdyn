"""
Cart-Pole (Inverted Pendulum on Cart) dynamical system.

A classic underactuated benchmark system for control algorithm testing.
Models a cart on a frictionless track with a pendulum attached at a pivot.

Dynamics
--------
The equations of motion are derived from Lagrangian mechanics:

    (M + m)ẍ + ml(θ̈ cos θ - θ̇² sin θ) = F - b_c ẋ
    l θ̈ + ẍ cos θ = g sin θ - b_p θ̇/m

Solving for ẍ and θ̈:

    θ̈ = [g sin θ - cos θ (F + ml θ̇² sin θ - b_c ẋ)/(M+m) - b_p θ̇/(ml)] /
         [l (4/3 - m cos² θ/(M+m))]

    ẍ = [F + ml(θ̇² sin θ - θ̈ cos θ) - b_c ẋ] / (M + m)

where:
    x : cart position
    θ : pole angle from vertical (up = 0)
    M : cart mass
    m : pole mass (point mass at end)
    l : pole half-length (distance to center of mass)
    g : gravitational acceleration
    F : force applied to cart
    b_c : cart friction coefficient
    b_p : pole friction coefficient

State Vector (n=4)
------------------
    x = [x, ẋ, θ, θ̇]

    Index 0: cart position [m]
    Index 1: cart velocity [m/s]
    Index 2: pole angle [rad] (0 = up, π = down)
    Index 3: pole angular velocity [rad/s]

Control Vector (m=1)
--------------------
    u = [F] - force applied to cart [N]

Angle Convention
----------------
    θ = 0: pole pointing straight up (unstable equilibrium)
    θ = π: pole pointing straight down (stable equilibrium)
    θ > 0: pole tilted to the right (clockwise from up)

References
----------
- Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system.
- Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive
  elements that can solve difficult learning control problems.
- Tedrake, R. Underactuated Robotics (MIT OCW).
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
class CartPoleParams:
    """
    Parameters for the Cart-Pole system.

    Attributes
    ----------
    M : float
        Cart mass [kg]. Default 1.0.
    m : float
        Pole mass (point mass at tip) [kg]. Default 0.1.
    l : float
        Pole half-length (distance from pivot to center of mass) [m]. Default 0.5.
    g : float
        Gravitational acceleration [m/s²]. Default 9.81.
    b_c : float
        Cart friction coefficient [N·s/m]. Default 0.0.
    b_p : float
        Pole friction coefficient [N·m·s/rad]. Default 0.0.

    # Constraints
    x_max : float
        Maximum cart position (track half-length) [m]. Default 2.4.
    v_max : float
        Maximum cart velocity [m/s]. Default inf.
    theta_max : float
        Maximum pole angle from vertical [rad]. Default inf.
    omega_max : float
        Maximum pole angular velocity [rad/s]. Default inf.
    F_max : float
        Maximum force magnitude [N]. Default 10.0.
    """

    # Physical parameters
    M: float = 1.0  # cart mass
    m: float = 0.1  # pole mass
    l: float = 0.5  # pole half-length
    g: float = G_DEFAULT

    # Friction
    b_c: float = 0.0  # cart friction
    b_p: float = 0.0  # pole friction

    # Constraints
    x_max: float = 2.4
    v_max: float = np.inf
    theta_max: float = np.inf  # often set to 12° for balancing
    omega_max: float = np.inf
    F_max: float = 10.0

    def __post_init__(self):
        """Validate parameters."""
        if self.M <= 0:
            raise ValueError(f"Cart mass M must be positive, got {self.M}")
        if self.m <= 0:
            raise ValueError(f"Pole mass m must be positive, got {self.m}")
        if self.l <= 0:
            raise ValueError(f"Pole length l must be positive, got {self.l}")
        if self.g < 0:
            raise ValueError(f"Gravity g must be non-negative, got {self.g}")
        if self.b_c < 0:
            raise ValueError(f"Cart friction b_c must be non-negative, got {self.b_c}")
        if self.b_p < 0:
            raise ValueError(f"Pole friction b_p must be non-negative, got {self.b_p}")
        if self.x_max <= 0:
            raise ValueError(f"x_max must be positive, got {self.x_max}")
        if self.F_max <= 0:
            raise ValueError(f"F_max must be positive, got {self.F_max}")

    @property
    def total_mass(self) -> float:
        """Total system mass: M + m."""
        return self.M + self.m

    @property
    def pole_inertia(self) -> float:
        """Pole moment of inertia about pivot (point mass): I = m·l²."""
        return self.m * self.l**2

    @property
    def mass_ratio(self) -> float:
        """Ratio of pole mass to total mass: m / (M + m)."""
        return self.m / self.total_mass


def default_params() -> CartPoleParams:
    """Create default cart-pole parameters."""
    return CartPoleParams()


def gym_params() -> CartPoleParams:
    """
    Create parameters matching OpenAI Gym's CartPole-v1.

    Note: Gym uses θ=0 as up, same as our convention.
    """
    return CartPoleParams(
        M=1.0,
        m=0.1,
        l=0.5,  # total pole length is 1.0m, half-length is 0.5m
        g=9.8,
        b_c=0.0,
        b_p=0.0,
        x_max=2.4,
        theta_max=np.deg2rad(12),  # 12° = ~0.209 rad
        F_max=10.0,
    )


def normalized_params() -> CartPoleParams:
    """
    Create normalized (non-dimensional) parameters.
    """
    return CartPoleParams(
        M=1.0,
        m=0.1,
        l=1.0,
        g=1.0,
        b_c=0.0,
        b_p=0.0,
        x_max=10.0,
        F_max=10.0,
    )


# =============================================================================
# Main System Class
# =============================================================================


class CartPole(DynamicalSystem):
    """
    Cart-Pole (Inverted Pendulum) dynamical system.

    An underactuated system with a cart on a track and a pole attached
    at a pivot. The control input is a force on the cart. Classic
    benchmark for swing-up and balancing control.

    Parameters
    ----------
    params : CartPoleParams, optional
        System parameters. If None, uses default parameters.

    Examples
    --------
    >>> # Create default cart-pole
    >>> cartpole = CartPole()
    >>>
    >>> # Initial state: pole slightly tilted from upright
    >>> x0 = np.array([0.0, 0.0, 0.05, 0.0])  # small angle
    >>>
    >>> # Simple proportional controller for balancing
    >>> def balance_controller(t, x):
    ...     theta, theta_dot = x[2], x[3]
    ...     return np.array([-50*theta - 20*theta_dot])
    >>>
    >>> t, x, u = cartpole.simulate(x0, balance_controller, (0, 5), dt=0.02)

    >>> # Swing-up from down position
    >>> x0 = np.array([0.0, 0.0, np.pi, 0.0])  # pole down
    >>> # Energy-based swing-up controller needed

    Notes
    -----
    The system is underactuated (1 control, 2 DOF). The balancing problem
    (stabilizing θ ≈ 0) is a classic control benchmark. The swing-up
    problem (moving from θ = π to θ = 0) requires nonlinear control.

    Common control approaches:
    - LQR for local stabilization near upright
    - Energy-based swing-up
    - Model predictive control (MPC)
    - Reinforcement learning
    """

    def __init__(self, params: Optional[CartPoleParams] = None):
        if params is None:
            params = default_params()
        super().__init__(params)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector [x, ẋ, θ, θ̇]."""
        return 4

    @property
    def n_control(self) -> int:
        """Dimension of control vector [F]."""
        return 1

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector."""
        return 4

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        return ["x", "x_dot", "theta", "theta_dot"]

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        return ["F"]

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_cart_position(self, x: np.ndarray) -> float:
        """Extract cart position from state vector."""
        return float(np.asarray(x)[0])

    def get_cart_velocity(self, x: np.ndarray) -> float:
        """Extract cart velocity from state vector."""
        return float(np.asarray(x)[1])

    def get_pole_angle(self, x: np.ndarray) -> float:
        """Extract pole angle from state vector."""
        return float(np.asarray(x)[2])

    def get_pole_angular_velocity(self, x: np.ndarray) -> float:
        """Extract pole angular velocity from state vector."""
        return float(np.asarray(x)[3])

    def pack_state(self, cart_pos: float, cart_vel: float, pole_angle: float, pole_omega: float) -> np.ndarray:
        """
        Pack components into state vector.

        Parameters
        ----------
        cart_pos : float
            Cart position [m].
        cart_vel : float
            Cart velocity [m/s].
        pole_angle : float
            Pole angle [rad].
        pole_omega : float
            Pole angular velocity [rad/s].

        Returns
        -------
        np.ndarray, shape (4,)
            State vector [x, ẋ, θ, θ̇].
        """
        return np.array([cart_pos, cart_vel, pole_angle, pole_omega])

    def normalize_angle(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state by wrapping pole angle to [-π, π].

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.

        Returns
        -------
        np.ndarray, shape (4,)
            State with wrapped angle.
        """
        x = np.asarray(x, dtype=np.float64).copy()
        x[2] = wrap_angle(x[2])
        return x

    def get_pole_tip_position(self, x: np.ndarray) -> np.ndarray:
        """
        Get Cartesian position of pole tip.

        Returns position in 2D: [horizontal, vertical] relative to track.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            Position [x_tip, y_tip] of pole tip.
        """
        cart_x = self.get_cart_position(x)
        theta = self.get_pole_angle(x)
        l = self.params.l * 2  # full pole length

        # θ=0 is up, so:
        # x_tip = cart_x + l·sin(θ)
        # y_tip = l·cos(θ)
        return np.array([cart_x + l * np.sin(theta), l * np.cos(theta)])

    def get_pole_tip_velocity(self, x: np.ndarray) -> np.ndarray:
        """
        Get Cartesian velocity of pole tip.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            Velocity [vx_tip, vy_tip] of pole tip.
        """
        cart_v = self.get_cart_velocity(x)
        theta = self.get_pole_angle(x)
        omega = self.get_pole_angular_velocity(x)
        l = self.params.l * 2  # full pole length

        # v_tip = [cart_v + l·ω·cos(θ), -l·ω·sin(θ)]
        return np.array([cart_v + l * omega * np.cos(theta), -l * omega * np.sin(theta)])

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector [x, ẋ, θ, θ̇].
        u : np.ndarray, shape (1,)
            Control vector [F].
        w : np.ndarray, shape (4,), optional
            Disturbance vector.

        Returns
        -------
        np.ndarray, shape (4,)
            State derivative [ẋ, ẍ, θ̇, θ̈].
        """
        x_state = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(4) if w is None else np.asarray(w, dtype=np.float64)

        # Extract state
        _ = x_state[0]
        cart_v = x_state[1]
        theta = x_state[2]
        omega = x_state[3]

        F = u[0]

        p = self.params
        M, m, l, g = p.M, p.m, p.l, p.g
        b_c, b_p = p.b_c, p.b_p

        # Precompute trig
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Total mass
        total_mass = M + m

        # Denominator for θ̈
        # Using the standard formulation with uniform rod (I = ml²/3 about pivot)
        # But for point mass at tip: I = m·(2l)² = 4ml²
        # For point mass at center: I = ml²
        # We'll use the simplified point-mass-at-center formulation

        # Effective length factor (4/3 for uniform rod, 1 for point mass at center)
        # Using 4/3 (uniform rod assumption is standard)
        len_factor = 4.0 / 3.0

        denom = l * (len_factor - m * cos_theta**2 / total_mass)

        # Pole angular acceleration
        # θ̈ = [g·sin(θ) - cos(θ)·(F + m·l·θ̇²·sin(θ) - b_c·ẋ)/(M+m) - b_p·θ̇/(m·l)] / denom
        temp = (F + m * l * omega**2 * sin_theta - b_c * cart_v) / total_mass
        theta_ddot = (g * sin_theta - cos_theta * temp - b_p * omega / (m * l)) / denom

        # Cart acceleration
        # ẍ = [F + m·l·(θ̇²·sin(θ) - θ̈·cos(θ)) - b_c·ẋ] / (M + m)
        cart_a = (F + m * l * (omega**2 * sin_theta - theta_ddot * cos_theta) - b_c * cart_v) / total_mass

        # State derivative
        x_dot = np.array([cart_v + w[0], cart_a + w[1], omega + w[2], theta_ddot + w[3]])

        return x_dot

    # =========================================================================
    # Jacobians
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian: ∂f/∂x.

        Computed analytically for the cart-pole dynamics.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.
        u : np.ndarray, shape (1,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (4, 4)
            State Jacobian matrix.
        """
        x_state = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        # Use numerical differentiation for robustness
        # (analytical Jacobian is complex due to coupling)
        eps = 1e-7
        A = np.zeros((4, 4))

        f0 = self.f(x_state, u, None)

        for i in range(4):
            x_plus = x_state.copy()
            x_plus[i] += eps
            f_plus = self.f(x_plus, u, None)
            A[:, i] = (f_plus - f0) / eps

        return A

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Control Jacobian: ∂f/∂u.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.
        u : np.ndarray, shape (1,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (4, 1)
            Control Jacobian matrix.
        """
        x_state = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        theta = x_state[2]

        p = self.params
        M, m, l = p.M, p.m, p.l

        cos_t = np.cos(theta)
        total_mass = M + m
        len_factor = 4.0 / 3.0

        denom = l * (len_factor - m * cos_t**2 / total_mass)

        # ∂θ̈/∂F
        d_theta_ddot_dF = -cos_t / (total_mass * denom)

        # ∂ẍ/∂F = [1 - m·l·cos(θ)·(∂θ̈/∂F)] / (M + m)
        #       = [1 + m·l·cos²(θ) / (total_mass·denom)] / total_mass
        d_cart_a_dF = (1 - m * l * cos_t * d_theta_ddot_dF) / total_mass

        B = np.array([[0.0], [d_cart_a_dF], [0.0], [d_theta_ddot_dF]])

        return B

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state bounds.

        Returns
        -------
        lb : np.ndarray, shape (4,)
            Lower bounds.
        ub : np.ndarray, shape (4,)
            Upper bounds.
        """
        p = self.params
        lb = np.array([-p.x_max, -p.v_max, -p.theta_max if p.theta_max < np.inf else -np.inf, -p.omega_max])
        ub = np.array([p.x_max, p.v_max, p.theta_max if p.theta_max < np.inf else np.inf, p.omega_max])
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
        F_max = self.params.F_max
        lb = np.array([-F_max])
        ub = np.array([F_max])
        return lb, ub

    def is_within_track(self, x: np.ndarray) -> bool:
        """Check if cart is within track bounds."""
        cart_x = self.get_cart_position(x)
        return np.abs(cart_x) <= self.params.x_max

    def is_balanced(self, x: np.ndarray, tol: float = 0.1) -> bool:
        """
        Check if pole is approximately balanced (upright).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        tol : float
            Tolerance in radians.

        Returns
        -------
        bool
            True if |θ| < tol.
        """
        theta = self.get_pole_angle(x)
        theta_wrapped = wrap_angle(theta)
        return np.abs(theta_wrapped) < tol

    def is_fallen(self, x: np.ndarray, tol: float = 0.5) -> bool:
        """
        Check if pole has fallen (near horizontal or down).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        tol : float
            Tolerance from π/2.

        Returns
        -------
        bool
            True if pole angle is near or past horizontal.
        """
        theta = self.get_pole_angle(x)
        theta_wrapped = wrap_angle(theta)
        return np.abs(theta_wrapped) > (np.pi / 2 - tol)

    # =========================================================================
    # Energy Methods
    # =========================================================================

    def energy(self, x: np.ndarray) -> dict:
        """
        Compute kinetic and potential energy.

        Potential energy reference: PE = 0 when pole is horizontal (θ = π/2).

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector.

        Returns
        -------
        dict
            Dictionary with 'kinetic', 'potential', and 'total' energy.
        """
        _ = self.get_cart_position(x)
        cart_v = self.get_cart_velocity(x)
        theta = self.get_pole_angle(x)
        omega = self.get_pole_angular_velocity(x)

        p = self.params
        M, m, l, g = p.M, p.m, p.l, p.g

        # Kinetic energy
        # Cart: (1/2)·M·ẋ²
        # Pole: (1/2)·m·v_cm² + (1/2)·I·ω²
        # where v_cm is velocity of pole center of mass

        # Pole center of mass velocity
        v_cm_x = cart_v + l * omega * np.cos(theta)
        v_cm_y = -l * omega * np.sin(theta)
        v_cm_sq = v_cm_x**2 + v_cm_y**2

        # For point mass, I = 0 about center of mass
        # But we use I = m·l²/3 for uniform rod
        I_rod = m * l**2 / 3.0

        KE_cart = 0.5 * M * cart_v**2
        KE_pole = 0.5 * m * v_cm_sq + 0.5 * I_rod * omega**2
        KE = KE_cart + KE_pole

        # Potential energy
        # Reference: PE = 0 when pole center of mass is at pivot height
        # PE = m·g·l·cos(θ) (positive when up, negative when down)
        PE = m * g * l * np.cos(theta)

        return {"kinetic": KE, "potential": PE, "total": KE + PE}

    def energy_upright(self) -> float:
        """
        Energy when pole is upright and system is at rest.

        Returns
        -------
        float
            Energy at upright equilibrium: m·g·l.
        """
        p = self.params
        return p.m * p.g * p.l

    def energy_down(self) -> float:
        """
        Energy when pole is down and system is at rest.

        Returns
        -------
        float
            Energy at down equilibrium: -m·g·l.
        """
        p = self.params
        return -p.m * p.g * p.l

    # =========================================================================
    # Equilibrium and Linearization
    # =========================================================================

    def equilibrium(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium state for given control.

        With F = 0:
        - θ = 0 (up): unstable equilibrium
        - θ = π (down): stable equilibrium

        Parameters
        ----------
        u : np.ndarray, shape (1,), optional
            Control input. Default is zero.

        Returns
        -------
        np.ndarray, shape (4,)
            Equilibrium state [0, 0, θ_eq, 0].
            Returns upright equilibrium for F = 0.
        """
        u = np.zeros(1) if u is None else np.asarray(u, dtype=np.float64)

        F = u[0]

        if np.abs(F) < 1e-10:
            # No force: return upright position
            return np.array([0.0, 0.0, 0.0, 0.0])
        else:
            # With constant force, no static equilibrium exists
            # (cart would accelerate)
            raise ValueError(f"No static equilibrium exists with non-zero force. Got F = {F:.3f}")

    def linearize_upright(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize around the upright (balanced) equilibrium.

        Returns
        -------
        A : np.ndarray, shape (4, 4)
            Linearized state matrix.
        B : np.ndarray, shape (4, 1)
            Linearized control matrix.
        """
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        u0 = np.array([0.0])
        return self.A(x0, u0), self.B(x0, u0)

    def linearize_down(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize around the down equilibrium.

        Returns
        -------
        A : np.ndarray, shape (4, 4)
            Linearized state matrix.
        B : np.ndarray, shape (4, 1)
            Linearized control matrix.
        """
        x0 = np.array([0.0, 0.0, np.pi, 0.0])
        u0 = np.array([0.0])
        return self.A(x0, u0), self.B(x0, u0)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_upright(self, x: np.ndarray, tol: float = 0.1) -> bool:
        """Check if pole is approximately upright (θ ≈ 0)."""
        return self.is_balanced(x, tol)

    def is_down(self, x: np.ndarray, tol: float = 0.1) -> bool:
        """
        Check if pole is approximately down (θ ≈ π).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        tol : float
            Tolerance in radians.

        Returns
        -------
        bool
            True if |θ - π| < tol or |θ + π| < tol.
        """
        theta = self.get_pole_angle(x)
        theta_wrapped = wrap_angle(theta)
        return np.abs(np.abs(theta_wrapped) - np.pi) < tol


# =============================================================================
# Factory Functions
# =============================================================================


def create_cartpole(
    M: float = 1.0,
    m: float = 0.1,
    l: float = 0.5,
    g: float = G_DEFAULT,
    b_c: float = 0.0,
    b_p: float = 0.0,
    F_max: float = 10.0,
) -> CartPole:
    """
    Create a cart-pole with specified parameters.

    Parameters
    ----------
    M : float
        Cart mass [kg].
    m : float
        Pole mass [kg].
    l : float
        Pole half-length [m].
    g : float
        Gravitational acceleration [m/s²].
    b_c : float
        Cart friction coefficient.
    b_p : float
        Pole friction coefficient.
    F_max : float
        Maximum force [N].

    Returns
    -------
    CartPole
        Configured cart-pole system.
    """
    params = CartPoleParams(M=M, m=m, l=l, g=g, b_c=b_c, b_p=b_p, F_max=F_max)
    return CartPole(params)


def create_gym_cartpole() -> CartPole:
    """
    Create a cart-pole matching OpenAI Gym's CartPole-v1.

    Returns
    -------
    CartPole
        Gym-compatible cart-pole system.
    """
    return CartPole(gym_params())


def create_normalized_cartpole() -> CartPole:
    """
    Create a cart-pole with normalized parameters.

    Returns
    -------
    CartPole
        Normalized cart-pole system.
    """
    return CartPole(normalized_params())
