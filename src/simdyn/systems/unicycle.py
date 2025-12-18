"""
Unicycle dynamical system.

A simple nonholonomic kinematic model commonly used for mobile robots
and ground vehicles. The unicycle cannot move sideways (nonholonomic
constraint).

Dynamics
--------
    ẋ = v·cos(θ)
    ẏ = v·sin(θ)
    θ̇ = ω

where:
    x, y : position in 2D plane
    θ : heading angle (orientation)
    v : linear velocity (forward speed)
    ω : angular velocity (turn rate)

State Vector
------------
    x = [x, y, θ] of dimension (3,)

Control Vector
--------------
    u = [v, ω] of dimension (2,)

Notes
-----
- This is a kinematic model (no inertia/dynamics of velocities)
- The system is nonholonomic: dim(control) < dim(configuration space)
- Good for algorithm testing before moving to more complex systems

References
----------
- LaValle (2006) - Planning Algorithms, Chapter 13
- Siegwart & Nourbakhsh - Introduction to Autonomous Mobile Robots
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from simdyn.base import DynamicalSystem
from simdyn.utils.rotations import wrap_angle


@dataclass
class UnicycleParams:
    """
    Parameters for the Unicycle system.

    Attributes
    ----------
    x_min : float
        Minimum x position. Default -10.0.
    x_max : float
        Maximum x position. Default 10.0.
    y_min : float
        Minimum y position. Default -10.0.
    y_max : float
        Maximum y position. Default 10.0.
    v_min : float
        Minimum linear velocity. Default -2.0 (allows reverse).
    v_max : float
        Maximum linear velocity. Default 2.0.
    omega_min : float
        Minimum angular velocity. Default -π rad/s.
    omega_max : float
        Maximum angular velocity. Default π rad/s.
    """

    # Position bounds
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0

    # Control bounds
    v_min: float = -2.0
    v_max: float = 2.0
    omega_min: float = -np.pi
    omega_max: float = np.pi

    def __post_init__(self):
        """Validate parameters."""
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be less than y_max")
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be less than v_max")
        if self.omega_min >= self.omega_max:
            raise ValueError("omega_min must be less than omega_max")


def default_params() -> UnicycleParams:
    """Create default parameters for the unicycle."""
    return UnicycleParams()


class Unicycle(DynamicalSystem):
    """
    Unicycle kinematic model.

    A simple 2D mobile robot model with position (x, y) and heading (θ),
    controlled by linear velocity (v) and angular velocity (ω).

    This is a nonholonomic system - the robot cannot move sideways,
    only forward/backward and rotate in place.

    Parameters
    ----------
    params : UnicycleParams, optional
        System parameters. If None, uses default parameters.

    Examples
    --------
    >>> system = Unicycle()
    >>> x = np.array([0.0, 0.0, 0.0])  # at origin, facing +x
    >>> u = np.array([1.0, 0.0])  # move forward at v=1
    >>> x_dot = system.f(x, u)  # should be [1, 0, 0]

    >>> # Simulate circular motion
    >>> x0 = np.array([0.0, 0.0, 0.0])
    >>> controller = lambda t, x: np.array([1.0, 1.0])  # v=1, ω=1
    >>> t, x, u = system.simulate(x0, controller, (0, 2*np.pi), dt=0.01)

    Notes
    -----
    The Jacobians are state-dependent (unlike the linear double integrator),
    making this a good test case for nonlinear control algorithms.

    State conventions:
    - θ = 0: facing positive x direction
    - θ = π/2: facing positive y direction
    - θ increases counter-clockwise
    """

    def __init__(self, params: Optional[UnicycleParams] = None):
        if params is None:
            params = default_params()
        super().__init__(params)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector [x, y, θ]."""
        return 3

    @property
    def n_control(self) -> int:
        """Dimension of control vector [v, ω]."""
        return 2

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector."""
        return 3

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        return ["x", "y", "theta"]

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        return ["v", "omega"]

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """
        Extract position from state vector.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].

        Returns
        -------
        np.ndarray, shape (2,)
            Position [x, y].
        """
        return np.asarray(x)[:2]

    def get_heading(self, x: np.ndarray) -> float:
        """
        Extract heading from state vector.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].

        Returns
        -------
        float
            Heading angle θ in radians.
        """
        return float(np.asarray(x)[2])

    def pack_state(self, position: np.ndarray, heading: float) -> np.ndarray:
        """
        Pack position and heading into state vector.

        Parameters
        ----------
        position : np.ndarray, shape (2,)
            Position [x, y].
        heading : float
            Heading angle θ in radians.

        Returns
        -------
        np.ndarray, shape (3,)
            State vector [x, y, θ].
        """
        position = np.asarray(position)
        return np.array([position[0], position[1], heading])

    def get_direction_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Get unit vector in the heading direction.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector.

        Returns
        -------
        np.ndarray, shape (2,)
            Unit vector [cos(θ), sin(θ)].
        """
        theta = self.get_heading(x)
        return np.array([np.cos(theta), np.sin(theta)])

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Dynamics:
            ẋ = v·cos(θ) + w_x
            ẏ = v·sin(θ) + w_y
            θ̇ = ω + w_θ

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].
        u : np.ndarray, shape (2,)
            Control vector [v, ω].
        w : np.ndarray, shape (3,), optional
            Disturbance vector [w_x, w_y, w_θ].

        Returns
        -------
        np.ndarray, shape (3,)
            State derivative [ẋ, ẏ, θ̇].
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(3) if w is None else np.asarray(w, dtype=np.float64)

        theta = x[2]
        v = u[0]
        omega = u[1]

        x_dot = np.array([v * np.cos(theta) + w[0], v * np.sin(theta) + w[1], omega + w[2]])

        return x_dot

    # =========================================================================
    # Jacobians
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian: ∂f/∂x.

        For the unicycle:
            A = [0  0  -v·sin(θ)]
                [0  0   v·cos(θ)]
                [0  0   0       ]

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].
        u : np.ndarray, shape (2,)
            Control vector [v, ω].

        Returns
        -------
        np.ndarray, shape (3, 3)
            State Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        theta = x[2]
        v = u[0]

        A = np.array([[0.0, 0.0, -v * np.sin(theta)], [0.0, 0.0, v * np.cos(theta)], [0.0, 0.0, 0.0]])

        return A

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Control Jacobian: ∂f/∂u.

        For the unicycle:
            B = [cos(θ)  0]
                [sin(θ)  0]
                [0       1]

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].
        u : np.ndarray, shape (2,)
            Control vector [v, ω] (unused, B only depends on x).

        Returns
        -------
        np.ndarray, shape (3, 2)
            Control Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)

        theta = x[2]

        B = np.array([[np.cos(theta), 0.0], [np.sin(theta), 0.0], [0.0, 1.0]])

        return B

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state bounds.

        Note: θ is unbounded (can wrap around).

        Returns
        -------
        lb : np.ndarray, shape (3,)
            Lower bounds [x_min, y_min, -inf].
        ub : np.ndarray, shape (3,)
            Upper bounds [x_max, y_max, inf].
        """
        p = self.params
        lb = np.array([p.x_min, p.y_min, -np.inf])
        ub = np.array([p.x_max, p.y_max, np.inf])
        return lb, ub

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get control bounds.

        Returns
        -------
        lb : np.ndarray, shape (2,)
            Lower bounds [v_min, omega_min].
        ub : np.ndarray, shape (2,)
            Upper bounds [v_max, omega_max].
        """
        p = self.params
        lb = np.array([p.v_min, p.omega_min])
        ub = np.array([p.v_max, p.omega_max])
        return lb, ub

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state by wrapping heading to [-π, π].

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            State vector [x, y, θ].

        Returns
        -------
        np.ndarray, shape (3,)
            State with θ wrapped to [-π, π].
        """
        x = np.asarray(x, dtype=np.float64).copy()
        x[2] = wrap_angle(x[2])
        return x

    def distance_to_point(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Compute Euclidean distance from current position to target.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            Current state.
        target : np.ndarray, shape (2,)
            Target position [x, y].

        Returns
        -------
        float
            Euclidean distance.
        """
        pos = self.get_position(x)
        target = np.asarray(target)
        return np.linalg.norm(pos - target)

    def heading_to_point(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Compute heading angle to reach target point.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            Current state.
        target : np.ndarray, shape (2,)
            Target position [x, y].

        Returns
        -------
        float
            Desired heading angle in radians.
        """
        pos = self.get_position(x)
        target = np.asarray(target)
        delta = target - pos
        return np.arctan2(delta[1], delta[0])

    def heading_error(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Compute heading error to reach target point.

        Parameters
        ----------
        x : np.ndarray, shape (3,)
            Current state.
        target : np.ndarray, shape (2,)
            Target position [x, y].

        Returns
        -------
        float
            Signed heading error in [-π, π].
        """
        desired_heading = self.heading_to_point(x, target)
        current_heading = self.get_heading(x)
        return wrap_angle(desired_heading - current_heading)

    def equilibrium(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium state for given control.

        For the unicycle, equilibrium requires v=0 and ω=0,
        which means the robot is stationary at any position/heading.

        Parameters
        ----------
        u : np.ndarray, shape (2,), optional
            Control input. Default is zero.

        Returns
        -------
        np.ndarray, shape (3,)
            Equilibrium state (origin with θ=0 as canonical).

        Raises
        ------
        ValueError
            If u has non-zero velocity components.
        """
        u = np.zeros(2) if u is None else np.asarray(u, dtype=np.float64)

        if not np.allclose(u, 0):
            raise ValueError(
                "Unicycle has no fixed equilibrium for non-zero control. "
                "With v≠0, the robot moves; with ω≠0, it rotates."
            )

        # Any state with v=0, ω=0 is equilibrium
        # Return origin as canonical
        return np.zeros(3)

    # =========================================================================
    # Trajectory Generation
    # =========================================================================

    def circular_trajectory(
        self, radius: float, speed: float, center: np.ndarray = None, clockwise: bool = False
    ) -> Tuple[np.ndarray, callable]:
        """
        Generate initial state and controller for circular motion.

        Parameters
        ----------
        radius : float
            Circle radius.
        speed : float
            Linear speed (positive).
        center : np.ndarray, shape (2,), optional
            Circle center. Default is origin.
        clockwise : bool
            If True, rotate clockwise. Default False (counter-clockwise).

        Returns
        -------
        x0 : np.ndarray, shape (3,)
            Initial state on the circle.
        controller : callable
            Controller function u = controller(t, x).
        """
        if center is None:
            center = np.array([0.0, 0.0])
        center = np.asarray(center)

        # Start at rightmost point of circle, heading up (or down if clockwise)
        x0 = np.array([center[0] + radius, center[1], np.pi / 2 if not clockwise else -np.pi / 2])

        # Angular velocity for circular motion: ω = v / r
        omega = speed / radius
        if clockwise:
            omega = -omega

        def controller(t, x):  # noqa: ARG001
            return np.array([speed, omega])

        return x0, controller

    def straight_line_controller(
        self, target: np.ndarray, k_p: float = 1.0, k_theta: float = 2.0, v_max: Optional[float] = None
    ) -> callable:
        """
        Create a simple go-to-point controller.

        Uses proportional control on distance and heading error.

        Parameters
        ----------
        target : np.ndarray, shape (2,)
            Target position.
        k_p : float
            Position gain (affects speed).
        k_theta : float
            Heading gain (affects turn rate).
        v_max : float, optional
            Maximum velocity. Uses param bounds if None.

        Returns
        -------
        callable
            Controller function u = controller(t, x).
        """
        target = np.asarray(target)

        if v_max is None:
            v_max = self.params.v_max

        def controller(t, x):  # noqa: ARG001
            # Distance to target
            dist = self.distance_to_point(x, target)

            # Heading error
            theta_err = self.heading_error(x, target)

            # Proportional control
            v = k_p * dist
            omega = k_theta * theta_err

            # Saturate velocity
            v = np.clip(v, 0, v_max)

            # Reduce speed when heading error is large
            v *= np.cos(theta_err) ** 2

            return np.array([v, omega])

        return controller


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_unicycle(
    x_bounds: Tuple[float, float] = (-10.0, 10.0),
    y_bounds: Tuple[float, float] = (-10.0, 10.0),
    v_bounds: Tuple[float, float] = (-2.0, 2.0),
    omega_bounds: Tuple[float, float] = (-np.pi, np.pi),
) -> Unicycle:
    """
    Create a unicycle with specified bounds.

    Parameters
    ----------
    x_bounds : tuple
        (min, max) x position bounds.
    y_bounds : tuple
        (min, max) y position bounds.
    v_bounds : tuple
        (min, max) linear velocity bounds.
    omega_bounds : tuple
        (min, max) angular velocity bounds.

    Returns
    -------
    Unicycle
        Configured unicycle system.
    """
    params = UnicycleParams(
        x_min=x_bounds[0],
        x_max=x_bounds[1],
        y_min=y_bounds[0],
        y_max=y_bounds[1],
        v_min=v_bounds[0],
        v_max=v_bounds[1],
        omega_min=omega_bounds[0],
        omega_max=omega_bounds[1],
    )
    return Unicycle(params)


def forward_only_unicycle(
    v_max: float = 2.0,
    omega_max: float = np.pi,
) -> Unicycle:
    """
    Create a unicycle that can only move forward (v >= 0).

    This is more realistic for many ground robots.

    Parameters
    ----------
    v_max : float
        Maximum forward velocity.
    omega_max : float
        Maximum angular velocity magnitude.

    Returns
    -------
    Unicycle
        Forward-only unicycle system.
    """
    params = UnicycleParams(
        v_min=0.0,
        v_max=v_max,
        omega_min=-omega_max,
        omega_max=omega_max,
    )
    return Unicycle(params)
