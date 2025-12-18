"""
Rocket 3-DoF (Three Degrees of Freedom) dynamical system.

A point-mass rocket model for powered descent guidance and control.
Includes mass depletion, gravity, and optional aerodynamic drag.

Dynamics
--------
    ṁ = -alpha·‖T‖
    ṙ = v
    v̇ = T/m + g + F_drag/m + w

where:
    m : mass (scalar)
    r : position in inertial frame (3,)
    v : velocity in inertial frame (3,)
    T : thrust vector in inertial frame (3,)
    g : gravity vector (3,)
    alpha : mass flow rate coefficient = 1/(g₀·Isp)
    F_drag : aerodynamic drag force (optional)

State Vector (n=7)
------------------
    x = [m, r_x, r_y, r_z, v_x, v_y, v_z]

    Index 0: mass
    Index 1-3: position (inertial frame)
    Index 4-6: velocity (inertial frame)

Control Vector (m=3)
--------------------
    u = [T_x, T_y, T_z] - thrust vector in inertial frame

Frame Convention
----------------
Following Szmuk et al. (2018):
    - Inertial frame: Up-East-North (UEN)
    - x-axis: Up (opposite to gravity)
    - y-axis: East
    - z-axis: North
    - Origin: Landing site

References
----------
- Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for 6-DoF
  powered descent guidance with compound state-triggered constraints.
- Blackmore, L., Açikmeşe, B., & Scharf, D. P. (2010). Minimum-landing-error
  powered-flight guidance for Mars landing using convex optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from simdyn.base import DynamicalSystem

# =============================================================================
# Constants
# =============================================================================

G0_EARTH = 9.80665  # Standard gravity [m/s²]


# =============================================================================
# Parameter Dataclass
# =============================================================================


@dataclass
class Rocket3DoFParams:
    """
    Parameters for the Rocket 3-DoF system.

    Attributes
    ----------
    m_dry : float
        Dry mass (mass without fuel) [kg or normalized].
    m_wet : float
        Wet mass (initial mass with fuel) [kg or normalized].
    I_sp : float
        Specific impulse [s].
    g0 : float
        Reference gravity for Isp calculation [m/s² or normalized].
    g_vec : np.ndarray
        Gravity vector in inertial frame [m/s² or normalized].
    T_min : float
        Minimum thrust magnitude.
    T_max : float
        Maximum thrust magnitude.

    # Drag parameters (optional)
    enable_drag : bool
        Whether to include aerodynamic drag.
    rho : float
        Atmospheric density [kg/m³].
    C_D : float
        Drag coefficient.
    A_ref : float
        Reference area for drag [m²].

    # Constraint parameters
    gamma_gs : float
        Glide slope angle [rad]. Cone half-angle from vertical.
    v_max : float
        Maximum velocity magnitude.

    # Position bounds
    r_min : np.ndarray or None
        Minimum position bounds (3,).
    r_max : np.ndarray or None
        Maximum position bounds (3,).
    """

    # Mass properties
    m_dry: float = 1.0
    m_wet: float = 2.0

    # Propulsion
    I_sp: float = 225.0  # seconds
    g0: float = G0_EARTH

    # Gravity
    g_vec: np.ndarray = field(default_factory=lambda: np.array([-9.80665, 0.0, 0.0]))

    # Thrust bounds
    T_min: float = 0.0
    T_max: float = 24000.0

    # Drag (optional)
    enable_drag: bool = False
    rho: float = 1.225  # sea level air density
    C_D: float = 0.5
    A_ref: float = 1.0  # m²

    # Constraints
    gamma_gs: float = np.pi / 4  # 45° glide slope
    v_max: float = 100.0  # m/s

    # Position bounds
    r_min: Optional[np.ndarray] = None
    r_max: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.m_dry <= 0:
            raise ValueError(f"m_dry must be positive, got {self.m_dry}")
        if self.m_wet < self.m_dry:
            raise ValueError(f"m_wet must be >= m_dry, got {self.m_wet} < {self.m_dry}")
        if self.I_sp <= 0:
            raise ValueError(f"I_sp must be positive, got {self.I_sp}")
        if self.T_min < 0:
            raise ValueError(f"T_min must be non-negative, got {self.T_min}")
        if self.T_max < self.T_min:
            raise ValueError("T_max must be >= T_min")
        if not 0 < self.gamma_gs < np.pi / 2:
            raise ValueError(f"gamma_gs must be in (0, π/2), got {self.gamma_gs}")

        # Ensure g_vec is numpy array
        self.g_vec = np.asarray(self.g_vec, dtype=np.float64)
        if self.g_vec.shape != (3,):
            raise ValueError(f"g_vec must have shape (3,), got {self.g_vec.shape}")

        # Convert bounds to arrays if provided
        if self.r_min is not None:
            self.r_min = np.asarray(self.r_min, dtype=np.float64)
        if self.r_max is not None:
            self.r_max = np.asarray(self.r_max, dtype=np.float64)

    @property
    def alpha(self) -> float:
        """Mass flow rate coefficient: alpha = 1/(g₀·Isp)."""
        return 1.0 / (self.g0 * self.I_sp)

    @property
    def fuel_mass(self) -> float:
        """Available fuel mass."""
        return self.m_wet - self.m_dry


def default_params() -> Rocket3DoFParams:
    """Create default parameters (SI units, Earth-like)."""
    return Rocket3DoFParams()


def normalized_params() -> Rocket3DoFParams:
    """
    Create normalized parameters matching Szmuk et al. (2018).

    These are non-dimensional parameters for algorithm testing.
    """
    return Rocket3DoFParams(
        m_dry=1.0,
        m_wet=2.0,
        I_sp=30.0,
        g0=1.0,
        g_vec=np.array([-1.0, 0.0, 0.0]),
        T_min=1.5,
        T_max=6.5,
        enable_drag=False,
        gamma_gs=np.deg2rad(30),  # 30° glide slope
        v_max=np.inf,
    )


# =============================================================================
# Main System Class
# =============================================================================


class Rocket3DoF(DynamicalSystem):
    """
    Rocket 3-DoF (point-mass) dynamical system.

    A translational dynamics model for rocket powered descent,
    including mass depletion and optional aerodynamic drag.

    Parameters
    ----------
    params : Rocket3DoFParams, optional
        System parameters. If None, uses default parameters.

    Examples
    --------
    >>> # Create system with default parameters
    >>> rocket = Rocket3DoF()
    >>>
    >>> # Initial state: hovering at 100m altitude
    >>> x0 = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> # Thrust to counteract gravity (hover)
    >>> m, g = x0[0], rocket.params.g_vec
    >>> T_hover = -m * g  # [19613, 0, 0] N
    >>>
    >>> # Compute dynamics
    >>> x_dot = rocket.f(x0, T_hover)

    >>> # Create with normalized parameters
    >>> rocket_norm = Rocket3DoF(normalized_params())

    Notes
    -----
    State ordering: [m, r_x, r_y, r_z, v_x, v_y, v_z]
    - Mass is first to match Szmuk formulation
    - Position and velocity in inertial (UEN) frame

    The system has state-dependent Jacobians due to:
    - T/m term (depends on mass)
    - Drag term (depends on velocity)
    """

    def __init__(self, params: Optional[Rocket3DoFParams] = None):
        if params is None:
            params = default_params()
        super().__init__(params)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector [m, r, v]."""
        return 7

    @property
    def n_control(self) -> int:
        """Dimension of control vector [T]."""
        return 3

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector."""
        return 7

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        return ["m", "r_x", "r_y", "r_z", "v_x", "v_y", "v_z"]

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        return ["T_x", "T_y", "T_z"]

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_mass(self, x: np.ndarray) -> float:
        """Extract mass from state vector."""
        return float(np.asarray(x)[0])

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position vector from state."""
        return np.asarray(x)[1:4]

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity vector from state."""
        return np.asarray(x)[4:7]

    def get_altitude(self, x: np.ndarray) -> float:
        """
        Get altitude (height above landing site).

        In UEN frame, altitude is the x-component of position.
        """
        return float(np.asarray(x)[1])

    def get_speed(self, x: np.ndarray) -> float:
        """Get speed (velocity magnitude)."""
        v = self.get_velocity(x)
        return float(np.linalg.norm(v))

    def pack_state(self, mass: float, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Pack mass, position, and velocity into state vector.

        Parameters
        ----------
        mass : float
            Current mass.
        position : np.ndarray, shape (3,)
            Position in inertial frame.
        velocity : np.ndarray, shape (3,)
            Velocity in inertial frame.

        Returns
        -------
        np.ndarray, shape (7,)
            State vector [m, r, v].
        """
        position = np.asarray(position)
        velocity = np.asarray(velocity)
        return np.concatenate([[mass], position, velocity])

    # =========================================================================
    # Thrust Utilities
    # =========================================================================

    def get_thrust_magnitude(self, u: np.ndarray) -> float:
        """Get thrust magnitude from control vector."""
        return float(np.linalg.norm(u))

    def get_thrust_direction(self, u: np.ndarray) -> np.ndarray:
        """
        Get unit thrust direction from control vector.

        Returns zero vector if thrust is zero.
        """
        u = np.asarray(u)
        T_mag = np.linalg.norm(u)
        if T_mag < 1e-10:
            return np.zeros(3)
        return u / T_mag

    # =========================================================================
    # Drag Model
    # =========================================================================

    def compute_drag(self, v: np.ndarray) -> np.ndarray:
        """
        Compute aerodynamic drag force.

        F_drag = -0.5 * rho * C_D * A_ref * ‖v‖ * v

        (Quadratic drag opposing velocity)

        Parameters
        ----------
        v : np.ndarray, shape (3,)
            Velocity vector.

        Returns
        -------
        np.ndarray, shape (3,)
            Drag force vector.
        """
        if not self.params.enable_drag:
            return np.zeros(3)

        v = np.asarray(v)
        v_mag = np.linalg.norm(v)

        if v_mag < 1e-10:
            return np.zeros(3)

        p = self.params
        # F_drag = -0.5 * rho * C_D * A_ref * v_mag² * v_hat
        #        = -0.5 * rho * C_D * A_ref * v_mag * v
        drag_coeff = 0.5 * p.rho * p.C_D * p.A_ref
        F_drag = -drag_coeff * v_mag * v

        return F_drag

    def compute_drag_jacobian(self, v: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of drag acceleration w.r.t. velocity.

        Parameters
        ----------
        v : np.ndarray, shape (3,)
            Velocity vector.

        Returns
        -------
        np.ndarray, shape (3, 3)
            ∂(F_drag/m)/∂v (note: depends on mass too, handled in A matrix)
        """
        if not self.params.enable_drag:
            return np.zeros((3, 3))

        v = np.asarray(v)
        v_mag = np.linalg.norm(v)

        if v_mag < 1e-10:
            return np.zeros((3, 3))

        p = self.params
        drag_coeff = 0.5 * p.rho * p.C_D * p.A_ref

        # F_drag = -drag_coeff * v_mag * v
        # ∂F_drag/∂v = -drag_coeff * (v_mag * I + v ⊗ v / v_mag)
        #            = -drag_coeff * (v_mag * I + v ⊗ v_hat)
        v_hat = v / v_mag
        dFdv = -drag_coeff * (v_mag * np.eye(3) + np.outer(v, v_hat))

        return dFdv

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Dynamics:
            ṁ = -alpha·‖T‖ + w_m
            ṙ = v + w_r
            v̇ = T/m + g + F_drag/m + w_v

        Parameters
        ----------
        x : np.ndarray, shape (7,)
            State vector [m, r_x, r_y, r_z, v_x, v_y, v_z].
        u : np.ndarray, shape (3,)
            Control vector [T_x, T_y, T_z] (thrust in inertial frame).
        w : np.ndarray, shape (7,), optional
            Disturbance vector.

        Returns
        -------
        np.ndarray, shape (7,)
            State derivative [ṁ, ṙ, v̇].
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(7) if w is None else np.asarray(w, dtype=np.float64)

        # Extract state
        m = x[0]
        v = x[4:7]

        # Thrust
        T = u
        T_mag = np.linalg.norm(T)

        # Parameters
        p = self.params
        alpha = p.alpha
        g = p.g_vec

        # Mass rate (fuel consumption)
        m_dot = -alpha * T_mag + w[0]

        # Position rate
        r_dot = v + w[1:4]

        # Velocity rate (acceleration)
        # a = T/m + g + F_drag/m
        a = T / m + g

        if p.enable_drag:
            F_drag = self.compute_drag(v)
            a = a + F_drag / m

        v_dot = a + w[4:7]

        return np.concatenate([[m_dot], r_dot, v_dot])

    # =========================================================================
    # Jacobians
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian: ∂f/∂x.

        Structure (7x7):
            ∂ṁ/∂m=0   ∂ṁ/∂r=0   ∂ṁ/∂v=0
            ∂ṙ/∂m=0   ∂ṙ/∂r=0   ∂ṙ/∂v=I
            ∂v̇/∂m     ∂v̇/∂r=0   ∂v̇/∂v

        Parameters
        ----------
        x : np.ndarray, shape (7,)
            State vector.
        u : np.ndarray, shape (3,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (7, 7)
            State Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        m = x[0]
        v = x[4:7]
        T = u

        A = np.zeros((7, 7))

        # ∂ṙ/∂v = I (position rate depends on velocity)
        A[1:4, 4:7] = np.eye(3)

        # ∂v̇/∂m = -T/m² (thrust acceleration depends on mass)
        A[4:7, 0] = -T / (m**2)

        # Add drag contribution if enabled
        if self.params.enable_drag:
            F_drag = self.compute_drag(v)

            # ∂v̇/∂m contribution from drag: -F_drag/m²
            A[4:7, 0] += -F_drag / (m**2)

            # ∂v̇/∂v contribution from drag: (1/m) * ∂F_drag/∂v
            dFdv = self.compute_drag_jacobian(v)
            A[4:7, 4:7] = dFdv / m

        return A

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Control Jacobian: ∂f/∂u.

        Structure (7x3):
            ∂ṁ/∂T = -alpha * T/‖T‖ (thrust direction)
            ∂ṙ/∂T = 0
            ∂v̇/∂T = I/m

        Parameters
        ----------
        x : np.ndarray, shape (7,)
            State vector.
        u : np.ndarray, shape (3,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (7, 3)
            Control Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        m = x[0]
        T = u
        T_mag = np.linalg.norm(T)

        B = np.zeros((7, 3))

        # ∂ṁ/∂T = -alpha * ∂‖T‖/∂T = -alpha * T/‖T‖
        if T_mag > 1e-10:
            B[0, :] = -self.params.alpha * T / T_mag
        # else: B[0, :] = 0 (subgradient at zero)

        # ∂v̇/∂T = I/m
        B[4:7, :] = np.eye(3) / m

        return B

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state bounds.

        Returns
        -------
        lb : np.ndarray, shape (7,)
            Lower bounds [m_dry, r_min, -inf].
        ub : np.ndarray, shape (7,)
            Upper bounds [m_wet, r_max, inf].
        """
        p = self.params

        # Mass bounds
        m_lb = p.m_dry
        m_ub = p.m_wet

        # Position bounds
        r_lb = p.r_min if p.r_min is not None else np.array([-np.inf, -np.inf, -np.inf])

        r_ub = p.r_max if p.r_max is not None else np.array([np.inf, np.inf, np.inf])

        # Velocity bounds (use v_max)
        v_lb = np.array([-p.v_max, -p.v_max, -p.v_max])
        v_ub = np.array([p.v_max, p.v_max, p.v_max])

        lb = np.concatenate([[m_lb], r_lb, v_lb])
        ub = np.concatenate([[m_ub], r_ub, v_ub])

        return lb, ub

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get control bounds.

        Note: These are component-wise bounds. The actual constraint
        is on thrust magnitude: T_min ≤ ‖T‖ ≤ T_max.

        Returns
        -------
        lb : np.ndarray, shape (3,)
            Lower bounds [-T_max, -T_max, -T_max].
        ub : np.ndarray, shape (3,)
            Upper bounds [T_max, T_max, T_max].
        """
        T_max = self.params.T_max
        lb = np.array([-T_max, -T_max, -T_max])
        ub = np.array([T_max, T_max, T_max])
        return lb, ub

    def thrust_constraint(self, u: np.ndarray) -> dict:
        """
        Evaluate thrust magnitude constraints.

        Constraints (negative = satisfied):
            T_min - ‖T‖ ≤ 0  (minimum thrust)
            ‖T‖ - T_max ≤ 0  (maximum thrust)

        Parameters
        ----------
        u : np.ndarray, shape (3,)
            Control vector (thrust).

        Returns
        -------
        dict
            Constraint values (negative = satisfied).
        """
        T_mag = np.linalg.norm(u)
        p = self.params

        return {
            "thrust_min": p.T_min - T_mag,  # T_min ≤ ‖T‖
            "thrust_max": T_mag - p.T_max,  # ‖T‖ ≤ T_max
        }

    def is_thrust_valid(self, u: np.ndarray) -> bool:
        """Check if thrust satisfies magnitude constraints."""
        constraints = self.thrust_constraint(u)
        return all(v <= 1e-10 for v in constraints.values())

    def glide_slope_constraint(self, x: np.ndarray) -> float:
        """
        Evaluate glide slope constraint.

        The glide slope constraint ensures the rocket stays within
        a cone centered at the landing site:
            ‖r_yz‖ ≤ r_x * tan(gamma_gs)

        Or equivalently:
            r_x * tan(gamma_gs) - ‖r_yz‖ ≥ 0

        Parameters
        ----------
        x : np.ndarray, shape (7,)
            State vector.

        Returns
        -------
        float
            Constraint value (negative = violated).
        """
        r = self.get_position(x)
        r_x = r[0]  # altitude (up component)
        r_yz = r[1:3]  # horizontal components

        gamma = self.params.gamma_gs

        # r_x * tan(gamma) - ‖r_yz‖ ≥ 0
        return r_x * np.tan(gamma) - np.linalg.norm(r_yz)

    def is_glide_slope_satisfied(self, x: np.ndarray) -> bool:
        """Check if state satisfies glide slope constraint."""
        return self.glide_slope_constraint(x) >= -1e-10

    def state_constraints(self, x: np.ndarray) -> dict:
        """
        Evaluate all state constraints.

        Parameters
        ----------
        x : np.ndarray, shape (7,)
            State vector.

        Returns
        -------
        dict
            Constraint values (negative = satisfied for bounds).
        """
        # Get base class bound constraints
        constraints = super().state_constraints(x)

        # Add glide slope (positive = satisfied, so negate for consistency)
        constraints["glide_slope"] = -self.glide_slope_constraint(x)

        # Add speed constraint
        speed = self.get_speed(x)
        constraints["speed"] = speed - self.params.v_max

        return constraints

    def control_constraints(self, u: np.ndarray) -> dict:
        """
        Evaluate all control constraints.

        Parameters
        ----------
        u : np.ndarray, shape (3,)
            Control vector.

        Returns
        -------
        dict
            Constraint values (negative = satisfied).
        """
        # Get base class bound constraints
        constraints = super().control_constraints(u)

        # Add thrust magnitude constraints
        thrust_constraints = self.thrust_constraint(u)
        constraints.update(thrust_constraints)

        return constraints

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def fuel_remaining(self, x: np.ndarray) -> float:
        """
        Compute remaining fuel mass.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Remaining fuel mass.
        """
        m = self.get_mass(x)
        return m - self.params.m_dry

    def fuel_fraction(self, x: np.ndarray) -> float:
        """
        Compute remaining fuel as fraction of initial fuel.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Fuel fraction in [0, 1].
        """
        return self.fuel_remaining(x) / self.params.fuel_mass

    def hover_thrust(self, x: np.ndarray) -> np.ndarray:
        """
        Compute thrust required to hover (counteract gravity).

        Parameters
        ----------
        x : np.ndarray
            State vector (need mass).

        Returns
        -------
        np.ndarray, shape (3,)
            Thrust vector for hovering.
        """
        m = self.get_mass(x)
        g = self.params.g_vec
        return -m * g

    def time_of_flight_estimate(self, x: np.ndarray, thrust_fraction: float = 0.8) -> float:
        """
        Estimate time of flight based on fuel and thrust.

        Simple estimate: t_f ≈ fuel_mass / (alpha * T_avg)

        Parameters
        ----------
        x : np.ndarray
            State vector.
        thrust_fraction : float
            Fraction of max thrust to assume (0 to 1).

        Returns
        -------
        float
            Estimated time of flight.
        """
        fuel = self.fuel_remaining(x)
        T_avg = thrust_fraction * self.params.T_max
        alpha = self.params.alpha

        if T_avg < 1e-10:
            return np.inf

        return fuel / (alpha * T_avg)

    def equilibrium(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find equilibrium state for given control (hover).

        For hovering: v=0, T = -m·g

        Parameters
        ----------
        u : np.ndarray, shape (3,), optional
            Thrust vector. If None, computes hover thrust.

        Returns
        -------
        np.ndarray, shape (7,)
            Equilibrium state.

        Notes
        -----
        True equilibrium with T≠0 doesn't exist due to mass depletion.
        This returns the "instantaneous" equilibrium for hovering.
        """
        if u is None:
            # Use wet mass for hover calculation
            m = self.params.m_wet
            u = -m * self.params.g_vec

        # Check if thrust can support hover
        T_required = np.linalg.norm(u)
        if T_required > self.params.T_max:
            raise ValueError(f"Hover thrust {T_required:.1f} exceeds T_max {self.params.T_max:.1f}")

        # Equilibrium at origin with zero velocity
        return self.pack_state(mass=self.params.m_wet, position=np.zeros(3), velocity=np.zeros(3))

    def energy(self, x: np.ndarray) -> dict:
        """
        Compute kinetic and potential energy.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        dict
            Dictionary with 'kinetic', 'potential', and 'total' energy.
        """
        m = self.get_mass(x)
        r = self.get_position(x)
        v = self.get_velocity(x)
        g = self.params.g_vec

        # Kinetic energy
        KE = 0.5 * m * np.dot(v, v)

        # Potential energy (relative to origin)
        # PE = -m * g · r (since g points down, PE increases with altitude)
        PE = -m * np.dot(g, r)

        return {"kinetic": KE, "potential": PE, "total": KE + PE}


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_rocket3dof(
    m_dry: float = 1.0,
    m_wet: float = 2.0,
    I_sp: float = 225.0,
    T_max: float = 24000.0,
    T_min: float = 0.0,
    g: float = 9.80665,
    enable_drag: bool = False,
) -> Rocket3DoF:
    """
    Create a Rocket3DoF system with specified parameters.

    Parameters
    ----------
    m_dry : float
        Dry mass.
    m_wet : float
        Wet mass (with fuel).
    I_sp : float
        Specific impulse [s].
    T_max : float
        Maximum thrust.
    T_min : float
        Minimum thrust.
    g : float
        Gravity magnitude (applied in -x direction).
    enable_drag : bool
        Whether to include aerodynamic drag.

    Returns
    -------
    Rocket3DoF
        Configured system.
    """
    params = Rocket3DoFParams(
        m_dry=m_dry,
        m_wet=m_wet,
        I_sp=I_sp,
        T_max=T_max,
        T_min=T_min,
        g_vec=np.array([-g, 0.0, 0.0]),
        enable_drag=enable_drag,
    )
    return Rocket3DoF(params)


def create_normalized_rocket3dof() -> Rocket3DoF:
    """
    Create a Rocket3DoF with normalized (non-dimensional) parameters.

    Matches the parameters used in Szmuk et al. (2018) for testing.

    Returns
    -------
    Rocket3DoF
        System with normalized parameters.
    """
    return Rocket3DoF(normalized_params())
