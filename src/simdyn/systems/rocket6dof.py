"""
Rocket 6-DoF (Six Degrees of Freedom) dynamical system.

A rigid-body rocket model for powered descent guidance and control,
matching the formulation in Szmuk & Açikmeşe (2018).

Includes translational and rotational dynamics with:
- Mass depletion
- Quaternion attitude representation
- Thrust vector control (gimbaling)
- Aerodynamic forces and torques
- Full analytical Jacobians

Dynamics
--------
    ṁ = -alpha·‖T_B‖
    ṙ_I = v_I
    v̇_I = (1/m)·C_BI^T·T_B + g_I + (1/m)·F_aero_I
    q̇_BI = (1/2)·Ω(ω_B)·q_BI
    ω̇_B = J^{-1}·(r_T x T_B + r_cp x F_aero_B - ω_B x J·ω_B + τ_aero_B)

where:
    m : mass (scalar)
    r_I : position in inertial frame (3,)
    v_I : velocity in inertial frame (3,)
    q_BI : quaternion from inertial to body frame (4,) [scalar-first]
    ω_B : angular velocity in body frame (3,)
    T_B : thrust vector in body frame (3,)
    C_BI : DCM from inertial to body (body vectors = C_BI @ inertial vectors)

State Vector (n=14)
-------------------
    x = [m, r_I(3), v_I(3), q_BI(4), ω_B(3)]

    Index 0: mass
    Index 1-3: position (inertial frame)
    Index 4-6: velocity (inertial frame)
    Index 7-10: quaternion [q_w, q_x, q_y, q_z] (scalar-first)
    Index 11-13: angular velocity (body frame)

Control Vector (m=3)
--------------------
    u = [T_Bx, T_By, T_Bz] - thrust vector in body frame

Frame Convention (Szmuk et al.)
-------------------------------
    - Inertial frame (I): Up-East-North (UEN), origin at landing site
    - Body frame (B): x-axis along vehicle centerline (thrust direction)
    - Quaternion q_BI transforms inertial vectors to body: v_B = C_BI @ v_I

References
----------
- Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for 6-DoF
  powered descent guidance with compound state-triggered constraints.
- Szmuk, M., Açikmeşe, B., & Berning, A. W. (2016). Successive convexification
  for fuel-optimal powered landing with aerodynamic drag and non-convex constraints.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from simdyn.base import DynamicalSystem
from simdyn.utils.quaternion import (
    omega_matrix,
    quat_conjugate,
    quat_normalize,
    quat_rotate,
    quat_to_dcm,
)
from simdyn.utils.rotations import skew

# =============================================================================
# Constants
# =============================================================================

G0_EARTH = 9.80665  # Standard gravity [m/s²]


# =============================================================================
# Parameter Dataclass
# =============================================================================


@dataclass
class Rocket6DoFParams:
    """
    Parameters for the Rocket 6-DoF system.

    Based on Szmuk et al. (2018) formulation.

    Attributes
    ----------
    m_dry : float
        Dry mass (mass without fuel).
    m_wet : float
        Wet mass (initial mass with fuel).
    J_B : np.ndarray
        Inertia tensor in body frame (3, 3). Diagonal for symmetric rocket.
    I_sp : float
        Specific impulse [s].
    g0 : float
        Reference gravity for Isp calculation.
    g_I : np.ndarray
        Gravity vector in inertial frame (3,).

    # Thrust geometry
    r_T_B : np.ndarray
        Thrust application point in body frame (3,).
        For rocket: typically negative x (at bottom).

    # Aerodynamics
    r_cp_B : np.ndarray
        Center of pressure in body frame (3,).
    enable_aero : bool
        Whether to include aerodynamic forces.
    aero_model : str
        Aerodynamic model: 'spherical' or 'ellipsoidal'.
    rho : float
        Atmospheric density.
    C_A : np.ndarray
        Aerodynamic coefficient matrix (3, 3) for ellipsoidal model.
        For spherical: C_A = c_a * I.
    S_ref : float
        Reference area for aerodynamics.

    # Thrust constraints
    T_min : float
        Minimum thrust magnitude.
    T_max : float
        Maximum thrust magnitude.
    delta_max : float
        Maximum gimbal angle [rad].
    theta_max : float
        Maximum tilt angle from vertical [rad].

    # Other constraints
    gamma_gs : float
        Glide slope angle [rad].
    omega_max : float
        Maximum angular velocity magnitude [rad/s].
    v_max : float
        Maximum velocity magnitude.
    """

    # Mass properties
    m_dry: float = 1.0
    m_wet: float = 2.0
    J_B: np.ndarray = field(default_factory=lambda: np.diag([0.02, 1.0, 1.0]) * 0.168)

    # Propulsion
    I_sp: float = 30.0
    g0: float = 1.0

    # Gravity
    g_I: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))  # noqa: N815

    # Thrust geometry
    r_T_B: np.ndarray = field(default_factory=lambda: np.array([-0.25, 0.0, 0.0]))  # noqa: N815

    # Aerodynamics
    r_cp_B: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.0]))  # noqa: N815
    enable_aero: bool = False
    aero_model: str = "spherical"  # 'spherical' or 'ellipsoidal'
    rho: float = 1.0
    C_A: np.ndarray = field(default_factory=lambda: np.diag([0.2, 1.0, 1.0]) * 0.5)
    S_ref: float = 0.5

    # Thrust constraints
    T_min: float = 1.5
    T_max: float = 6.5
    delta_max: float = np.deg2rad(20.0)  # 20° gimbal
    theta_max: float = np.deg2rad(90.0)  # 90° tilt (pointing constraint)

    # Other constraints
    gamma_gs: float = np.deg2rad(30.0)  # 30° glide slope
    omega_max: float = np.deg2rad(60.0)  # 60°/s max angular rate
    v_max: float = np.inf

    def __post_init__(self):  # noqa: C901, PLR0912
        """Validate and convert parameters."""
        # Validate mass
        if self.m_dry <= 0:
            raise ValueError(f"m_dry must be positive, got {self.m_dry}")
        if self.m_wet < self.m_dry:
            raise ValueError("m_wet must be >= m_dry")

        # Validate propulsion
        if self.I_sp <= 0:
            raise ValueError(f"I_sp must be positive, got {self.I_sp}")
        if self.g0 <= 0:
            raise ValueError(f"g0 must be positive, got {self.g0}")

        # Validate thrust bounds
        if self.T_min < 0:
            raise ValueError("T_min must be non-negative")
        if self.T_max < self.T_min:
            raise ValueError("T_max must be >= T_min")

        # Validate angles
        if not 0 <= self.delta_max <= np.pi / 2:
            raise ValueError("delta_max must be in [0, π/2]")
        if not 0 < self.theta_max <= np.pi:
            raise ValueError("theta_max must be in (0, π]")
        if not 0 < self.gamma_gs < np.pi / 2:
            raise ValueError("gamma_gs must be in (0, π/2)")

        # Convert to numpy arrays
        self.J_B = np.asarray(self.J_B, dtype=np.float64)
        self.g_I = np.asarray(self.g_I, dtype=np.float64)
        self.r_T_B = np.asarray(self.r_T_B, dtype=np.float64)
        self.r_cp_B = np.asarray(self.r_cp_B, dtype=np.float64)
        self.C_A = np.asarray(self.C_A, dtype=np.float64)

        # Validate shapes
        if self.J_B.shape != (3, 3):
            raise ValueError(f"J_B must be (3,3), got {self.J_B.shape}")
        if self.g_I.shape != (3,):
            raise ValueError(f"g_I must be (3,), got {self.g_I.shape}")
        if self.r_T_B.shape != (3,):
            raise ValueError(f"r_T_B must be (3,), got {self.r_T_B.shape}")
        if self.r_cp_B.shape != (3,):
            raise ValueError(f"r_cp_B must be (3,), got {self.r_cp_B.shape}")
        if self.C_A.shape != (3, 3):
            raise ValueError(f"C_A must be (3,3), got {self.C_A.shape}")

        # Precompute inverse inertia
        self._J_B_inv = np.linalg.inv(self.J_B)

    @property
    def alpha(self) -> float:
        """Mass flow rate coefficient: alpha = 1/(g₀·Isp)."""
        return 1.0 / (self.g0 * self.I_sp)

    @property
    def fuel_mass(self) -> float:
        """Available fuel mass."""
        return self.m_wet - self.m_dry

    @property
    def J_B_inv(self) -> np.ndarray:
        """Inverse of inertia tensor."""
        return self._J_B_inv


def default_params() -> Rocket6DoFParams:
    """Create default parameters (Szmuk normalized)."""
    return Rocket6DoFParams()


def szmuk_params() -> Rocket6DoFParams:
    """
    Create parameters matching Szmuk et al. (2018) paper.

    Non-dimensional units with scaling:
        U_L = 10 m (length)
        U_T = 2 s (time)
        U_M = 5000 kg (mass)
    """
    return Rocket6DoFParams(
        m_dry=1.0,
        m_wet=2.0,
        J_B=np.diag([0.02, 1.0, 1.0]) * 0.168,
        I_sp=30.0,
        g0=1.0,
        g_I=np.array([-1.0, 0.0, 0.0]),
        r_T_B=np.array([-0.25, 0.0, 0.0]),
        r_cp_B=np.array([0.05, 0.0, 0.0]),
        enable_aero=False,
        T_min=1.5,
        T_max=6.5,
        delta_max=np.deg2rad(20.0),
        theta_max=np.deg2rad(90.0),
        gamma_gs=np.deg2rad(30.0),
        omega_max=np.deg2rad(60.0),
    )


# =============================================================================
# Main System Class
# =============================================================================


class Rocket6DoF(DynamicalSystem):
    """
    Rocket 6-DoF (rigid body) dynamical system.

    Full translational and rotational dynamics for rocket powered descent,
    matching the Szmuk et al. (2018) formulation.

    Parameters
    ----------
    params : Rocket6DoFParams, optional
        System parameters. If None, uses default (Szmuk) parameters.

    Examples
    --------
    >>> # Create system with Szmuk parameters
    >>> rocket = Rocket6DoF()
    >>>
    >>> # Initial state: hovering at altitude 10
    >>> q0 = np.array([1, 0, 0, 0])  # identity quaternion (upright)
    >>> x0 = rocket.pack_state(
    ...     mass=2.0,
    ...     position=np.array([10.0, 0.0, 0.0]),
    ...     velocity=np.zeros(3),
    ...     quaternion=q0,
    ...     omega=np.zeros(3)
    ... )
    >>>
    >>> # Thrust along body x-axis (up when upright)
    >>> T_hover = rocket.hover_thrust(x0)
    >>> x_dot = rocket.f(x0, T_hover)

    Notes
    -----
    State ordering: [m, r_I(3), v_I(3), q_BI(4), ω_B(3)]

    Quaternion convention:
    - Scalar-first: q = [q_w, q_x, q_y, q_z]
    - q_BI transforms inertial to body: v_B = C(q_BI) @ v_I

    Body frame:
    - x-axis: along rocket centerline (thrust direction when δ=0)
    - y, z: perpendicular, completing right-hand frame
    """

    def __init__(self, params: Optional[Rocket6DoFParams] = None):
        if params is None:
            params = default_params()
        super().__init__(params)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_state(self) -> int:
        """Dimension of state vector [m, r, v, q, ω]."""
        return 14

    @property
    def n_control(self) -> int:
        """Dimension of control vector [T_B]."""
        return 3

    @property
    def n_disturbance(self) -> int:
        """Dimension of disturbance vector."""
        return 14

    @property
    def state_names(self) -> List[str]:
        """Names for each state element."""
        return [
            "m",
            "r_x",
            "r_y",
            "r_z",
            "v_x",
            "v_y",
            "v_z",
            "q_w",
            "q_x",
            "q_y",
            "q_z",
            "omega_x",
            "omega_y",
            "omega_z",
        ]

    @property
    def control_names(self) -> List[str]:
        """Names for each control element."""
        return ["T_Bx", "T_By", "T_Bz"]

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_mass(self, x: np.ndarray) -> float:
        """Extract mass from state vector."""
        return float(np.asarray(x)[0])

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position vector (inertial frame) from state."""
        return np.asarray(x)[1:4]

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity vector (inertial frame) from state."""
        return np.asarray(x)[4:7]

    def get_quaternion(self, x: np.ndarray) -> np.ndarray:
        """Extract quaternion from state."""
        return np.asarray(x)[7:11]

    def get_omega(self, x: np.ndarray) -> np.ndarray:
        """Extract angular velocity (body frame) from state."""
        return np.asarray(x)[11:14]

    def get_dcm(self, x: np.ndarray) -> np.ndarray:
        """
        Get Direction Cosine Matrix from state quaternion.

        Returns C_BI such that v_B = C_BI @ v_I.
        """
        q = self.get_quaternion(x)
        # quat_to_dcm gives C such that v_rotated = C @ v
        # For q_BI, this gives C_BI
        return quat_to_dcm(q)

    def get_altitude(self, x: np.ndarray) -> float:
        """Get altitude (x-component of position in UEN frame)."""
        return float(np.asarray(x)[1])

    def get_speed(self, x: np.ndarray) -> float:
        """Get speed (velocity magnitude)."""
        v = self.get_velocity(x)
        return float(np.linalg.norm(v))

    def pack_state(
        self, mass: float, position: np.ndarray, velocity: np.ndarray, quaternion: np.ndarray, omega: np.ndarray
    ) -> np.ndarray:
        """
        Pack components into state vector.

        Parameters
        ----------
        mass : float
            Current mass.
        position : np.ndarray, shape (3,)
            Position in inertial frame.
        velocity : np.ndarray, shape (3,)
            Velocity in inertial frame.
        quaternion : np.ndarray, shape (4,)
            Attitude quaternion [q_w, q_x, q_y, q_z].
        omega : np.ndarray, shape (3,)
            Angular velocity in body frame.

        Returns
        -------
        np.ndarray, shape (14,)
            State vector.
        """
        position = np.asarray(position)
        velocity = np.asarray(velocity)
        quaternion = np.asarray(quaternion)
        omega = np.asarray(omega)

        return np.concatenate([[mass], position, velocity, quaternion, omega])

    # =========================================================================
    # Thrust and Attitude Utilities
    # =========================================================================

    def get_thrust_magnitude(self, u: np.ndarray) -> float:
        """Get thrust magnitude from control vector."""
        return float(np.linalg.norm(u))

    def get_thrust_direction_body(self, u: np.ndarray) -> np.ndarray:
        """
        Get unit thrust direction in body frame.

        Returns [1,0,0] (body x-axis) if thrust is zero.
        """
        u = np.asarray(u)
        T_mag = np.linalg.norm(u)
        if T_mag < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return u / T_mag

    def get_gimbal_angle(self, u: np.ndarray) -> float:
        """
        Compute gimbal angle (thrust deflection from body x-axis).

        δ = arccos(T_B · e_x / ‖T_B‖)

        Parameters
        ----------
        u : np.ndarray, shape (3,)
            Thrust vector in body frame.

        Returns
        -------
        float
            Gimbal angle in radians [0, π].
        """
        u = np.asarray(u)
        T_mag = np.linalg.norm(u)
        if T_mag < 1e-10:
            return 0.0

        # cos(δ) = T_x / ‖T‖
        cos_delta = u[0] / T_mag
        cos_delta = np.clip(cos_delta, -1.0, 1.0)
        return np.arccos(cos_delta)

    def get_tilt_angle(self, x: np.ndarray) -> float:
        """
        Compute tilt angle (body x-axis deflection from inertial up).

        θ = arccos(e_x^B · e_x^I)

        where e_x^B is body x-axis expressed in inertial frame.

        Parameters
        ----------
        x : np.ndarray, shape (14,)
            State vector.

        Returns
        -------
        float
            Tilt angle in radians [0, π].
        """
        q = self.get_quaternion(x)

        # Body x-axis in body frame
        e_x_B = np.array([1.0, 0.0, 0.0])

        # Transform to inertial frame
        # v_I = C_BI^T @ v_B = C_IB @ v_B
        # Using quaternion: v_I = q_IB ⊗ v_B ⊗ q_IB*
        # where q_IB = q_BI* (conjugate)
        q_IB = quat_conjugate(q)
        e_x_I = quat_rotate(q_IB, e_x_B)

        # Inertial up direction
        e_up_I = np.array([1.0, 0.0, 0.0])

        # Tilt angle
        cos_theta = np.dot(e_x_I, e_up_I)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    # =========================================================================
    # Aerodynamic Model
    # =========================================================================

    def compute_aero_force_body(self, v_I: np.ndarray, C_BI: np.ndarray) -> np.ndarray:
        """
        Compute aerodynamic force in body frame.

        For spherical model:
            F_aero_B = -0.5 * rho * S * c_a * ‖v_B‖ * v_B

        For ellipsoidal model (Szmuk):
            F_aero_B = -0.5 * rho * S * C_A @ v_B * ‖v_B‖

        Parameters
        ----------
        v_I : np.ndarray, shape (3,)
            Velocity in inertial frame.
        C_BI : np.ndarray, shape (3, 3)
            DCM from inertial to body.

        Returns
        -------
        np.ndarray, shape (3,)
            Aerodynamic force in body frame.
        """
        if not self.params.enable_aero:
            return np.zeros(3)

        # Velocity in body frame
        v_B = C_BI @ v_I
        v_mag = np.linalg.norm(v_B)

        if v_mag < 1e-10:
            return np.zeros(3)

        p = self.params

        if p.aero_model == "spherical":
            # Scalar drag coefficient (use average of C_A diagonal)
            c_a = np.trace(p.C_A) / 3.0
            F_aero_B = -0.5 * p.rho * p.S_ref * c_a * v_mag * v_B
        else:  # ellipsoidal
            # F = -0.5 * rho * S * C_A @ v_B * ‖v_B‖
            F_aero_B = -0.5 * p.rho * p.S_ref * (p.C_A @ v_B) * v_mag

        return F_aero_B

    def compute_aero_torque_body(self, F_aero_B: np.ndarray) -> np.ndarray:
        """
        Compute aerodynamic torque in body frame.

        τ_aero_B = r_cp x F_aero_B

        Parameters
        ----------
        F_aero_B : np.ndarray, shape (3,)
            Aerodynamic force in body frame.

        Returns
        -------
        np.ndarray, shape (3,)
            Aerodynamic torque in body frame.
        """
        if not self.params.enable_aero:
            return np.zeros(3)

        return np.cross(self.params.r_cp_B, F_aero_B)

    # =========================================================================
    # Dynamics
    # =========================================================================

    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Dynamics:
            ṁ = -alpha·‖T_B‖
            ṙ_I = v_I
            v̇_I = (1/m)·C_BI^T·T_B + g_I + (1/m)·F_aero_I
            q̇_BI = (1/2)·Ω(ω_B)·q_BI
            ω̇_B = J^{-1}·(r_T x T_B + τ_aero - ω_B x J·ω_B)

        Parameters
        ----------
        x : np.ndarray, shape (14,)
            State vector [m, r_I, v_I, q_BI, ω_B].
        u : np.ndarray, shape (3,)
            Control vector [T_B] (thrust in body frame).
        w : np.ndarray, shape (14,), optional
            Disturbance vector.

        Returns
        -------
        np.ndarray, shape (14,)
            State derivative.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        w = np.zeros(14) if w is None else np.asarray(w, dtype=np.float64)

        # Extract state
        m = x[0]
        _ = x[1:4]
        v_I = x[4:7]
        q = x[7:11]
        omega_B = x[11:14]

        # Thrust
        T_B = u
        T_mag = np.linalg.norm(T_B)

        # Parameters
        p = self.params
        alpha = p.alpha
        g_I = p.g_I
        r_T_B = p.r_T_B
        J_B = p.J_B
        J_B_inv = p.J_B_inv

        # DCM: C_BI transforms inertial to body
        C_BI = quat_to_dcm(q)
        C_IB = C_BI.T  # Transforms body to inertial

        # === Mass rate ===
        m_dot = -alpha * T_mag + w[0]

        # === Position rate ===
        r_dot = v_I + w[1:4]

        # === Velocity rate ===
        # Thrust acceleration (transform to inertial)
        a_thrust_I = C_IB @ T_B / m

        # Aerodynamic force
        F_aero_B = self.compute_aero_force_body(v_I, C_BI)
        F_aero_I = C_IB @ F_aero_B
        a_aero_I = F_aero_I / m

        v_dot = a_thrust_I + g_I + a_aero_I + w[4:7]

        # === Quaternion rate ===
        # q̇ = (1/2) * Ω(ω) * q
        Omega = omega_matrix(omega_B)
        q_dot = 0.5 * Omega @ q + w[7:11]

        # === Angular velocity rate ===
        # τ = r_T x T_B + τ_aero - ω x J·ω
        tau_thrust = np.cross(r_T_B, T_B)
        tau_aero = self.compute_aero_torque_body(F_aero_B)
        tau_gyro = -np.cross(omega_B, J_B @ omega_B)

        tau_total = tau_thrust + tau_aero + tau_gyro
        omega_dot = J_B_inv @ tau_total + w[11:14]

        return np.concatenate([[m_dot], r_dot, v_dot, q_dot, omega_dot])

    # =========================================================================
    # Jacobians
    # =========================================================================

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian: ∂f/∂x.

        Structure (14x14):
            [∂ṁ/∂x    ]   [0  0  0  0  0 ]
            [∂ṙ/∂x    ] = [0  0  I  0  0 ]
            [∂v̇/∂x    ]   [*  0  *  *  0 ]
            [∂q̇/∂x    ]   [0  0  0  *  * ]
            [∂ω̇/∂x    ]   [0  0  *  0  * ]

        Parameters
        ----------
        x : np.ndarray, shape (14,)
            State vector.
        u : np.ndarray, shape (3,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (14, 14)
            State Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        # Extract state
        m = x[0]
        v_I = x[4:7]
        q = x[7:11]
        omega_B = x[11:14]

        T_B = u

        p = self.params
        J_B = p.J_B
        J_B_inv = p.J_B_inv

        # DCM and derivatives
        C_BI = quat_to_dcm(q)
        C_IB = C_BI.T

        A = np.zeros((14, 14))

        # --- ∂ṙ/∂v = I (rows 1-3, cols 4-6) ---
        A[1:4, 4:7] = np.eye(3)

        # --- ∂v̇/∂m = -(1/m²) * C_IB @ T_B (row 4-6, col 0) ---
        A[4:7, 0] = -(1.0 / m**2) * (C_IB @ T_B)

        # Add aero contribution to ∂v̇/∂m
        if p.enable_aero:
            F_aero_B = self.compute_aero_force_body(v_I, C_BI)
            F_aero_I = C_IB @ F_aero_B
            A[4:7, 0] += -(1.0 / m**2) * F_aero_I

        # --- ∂v̇/∂v (rows 4-6, cols 4-6) - from aero ---
        if p.enable_aero:
            A[4:7, 4:7] = self._compute_dv_dot_dv(m, v_I, C_BI, C_IB)

        # --- ∂v̇/∂q (rows 4-6, cols 7-10) ---
        A[4:7, 7:11] = self._compute_dv_dot_dq(m, v_I, q, T_B)

        # --- ∂q̇/∂q (rows 7-10, cols 7-10) ---
        # q̇ = (1/2) * Ω(ω) * q
        # ∂q̇/∂q = (1/2) * Ω(ω)
        A[7:11, 7:11] = 0.5 * omega_matrix(omega_B)

        # --- ∂q̇/∂ω (rows 7-10, cols 11-13) ---
        A[7:11, 11:14] = self._compute_dq_dot_domega(q)

        # --- ∂ω̇/∂v (rows 11-13, cols 4-6) - from aero torque ---
        if p.enable_aero:
            A[11:14, 4:7] = self._compute_domega_dot_dv(v_I, C_BI, J_B_inv)

        # --- ∂ω̇/∂ω (rows 11-13, cols 11-13) ---
        A[11:14, 11:14] = self._compute_domega_dot_domega(omega_B, J_B, J_B_inv)

        return A

    def _compute_dv_dot_dv(self, m: float, v_I: np.ndarray, C_BI: np.ndarray, C_IB: np.ndarray) -> np.ndarray:
        """Compute ∂v̇/∂v contribution from aerodynamics."""
        p = self.params
        v_B = C_BI @ v_I
        v_mag = np.linalg.norm(v_B)

        if v_mag < 1e-10:
            return np.zeros((3, 3))

        v_hat_B = v_B / v_mag

        if p.aero_model == "spherical":
            c_a = np.trace(p.C_A) / 3.0
            # ∂F_B/∂v_B = -0.5*rho*S*c_a * (v_mag*I + v_B ⊗ v_hat_B)
            dF_B_dv_B = -0.5 * p.rho * p.S_ref * c_a * (v_mag * np.eye(3) + np.outer(v_B, v_hat_B))
        else:
            # ∂F_B/∂v_B = -0.5*rho*S * (C_A*v_mag + C_A@v_B ⊗ v_hat_B)
            dF_B_dv_B = -0.5 * p.rho * p.S_ref * (p.C_A * v_mag + np.outer(p.C_A @ v_B, v_hat_B))

        # Chain rule: ∂v̇/∂v_I = (1/m) * C_IB @ ∂F_B/∂v_B @ C_BI
        return (1.0 / m) * C_IB @ dF_B_dv_B @ C_BI

    def _compute_dv_dot_dq(self, m: float, v_I: np.ndarray, q: np.ndarray, T_B: np.ndarray) -> np.ndarray:
        """
        Compute ∂v̇/∂q.

        v̇ includes (1/m) * C_IB @ T_B, which depends on q.
        """
        # v̇ = (1/m) * C_IB(q) @ T_B + ...
        # ∂v̇/∂q = (1/m) * ∂(C_IB @ T_B)/∂q

        # For quaternion q = [w, x, y, z], C_IB = C(q)^T where C = quat_to_dcm
        # Use finite difference as analytical form is complex
        # Actually, we can derive it using the fact that:
        # C_IB @ v = R(q^*) @ v for rotation

        # Numerical approximation for robustness
        eps = 1e-7
        dv_dot_dq = np.zeros((3, 4))

        C_BI = quat_to_dcm(q)
        C_IB = C_BI.T

        # Base term
        v_dot_base = (1.0 / m) * C_IB @ T_B

        # Aero contribution
        if self.params.enable_aero:
            F_aero_B = self.compute_aero_force_body(v_I, C_BI)
            v_dot_base += (1.0 / m) * C_IB @ F_aero_B

        for i in range(4):
            q_plus = q.copy()
            q_plus[i] += eps
            q_plus = quat_normalize(q_plus)

            C_BI_plus = quat_to_dcm(q_plus)
            C_IB_plus = C_BI_plus.T

            v_dot_plus = (1.0 / m) * C_IB_plus @ T_B

            if self.params.enable_aero:
                F_aero_B_plus = self.compute_aero_force_body(v_I, C_BI_plus)
                v_dot_plus += (1.0 / m) * C_IB_plus @ F_aero_B_plus

            dv_dot_dq[:, i] = (v_dot_plus - v_dot_base) / eps

        return dv_dot_dq

    def _compute_dq_dot_domega(self, q: np.ndarray) -> np.ndarray:
        """
        Compute ∂q̇/∂ω.

        q̇ = (1/2) * Ω(ω) * q
        ∂q̇/∂ω = (1/2) * [∂Ω/∂ωx * q, ∂Ω/∂ωy * q, ∂Ω/∂ωz * q]
        """
        # Ω(ω) = [0, -ω^T; ω, -skew(ω)]
        # ∂Ω/∂ωi gives specific matrices

        qw, qx, qy, qz = q

        # ∂q̇/∂ωx
        dOmega_dox = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=np.float64)

        # ∂q̇/∂ωy
        dOmega_doy = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        # ∂q̇/∂ωz
        dOmega_doz = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], dtype=np.float64)

        dq_dot_domega = np.zeros((4, 3))
        dq_dot_domega[:, 0] = 0.5 * dOmega_dox @ q
        dq_dot_domega[:, 1] = 0.5 * dOmega_doy @ q
        dq_dot_domega[:, 2] = 0.5 * dOmega_doz @ q

        return dq_dot_domega

    def _compute_domega_dot_dv(self, v_I: np.ndarray, C_BI: np.ndarray, J_B_inv: np.ndarray) -> np.ndarray:
        """Compute ∂ω̇/∂v from aerodynamic torque."""
        p = self.params
        v_B = C_BI @ v_I
        v_mag = np.linalg.norm(v_B)

        if v_mag < 1e-10:
            return np.zeros((3, 3))

        v_hat_B = v_B / v_mag
        r_cp = p.r_cp_B

        # τ_aero = r_cp x F_aero
        # ∂τ_aero/∂v_I = skew(r_cp) @ ∂F_aero_B/∂v_B @ C_BI

        if p.aero_model == "spherical":
            c_a = np.trace(p.C_A) / 3.0
            dF_B_dv_B = -0.5 * p.rho * p.S_ref * c_a * (v_mag * np.eye(3) + np.outer(v_B, v_hat_B))
        else:
            dF_B_dv_B = -0.5 * p.rho * p.S_ref * (p.C_A * v_mag + np.outer(p.C_A @ v_B, v_hat_B))

        dtau_dv_B = skew(r_cp) @ dF_B_dv_B
        dtau_dv_I = dtau_dv_B @ C_BI

        return J_B_inv @ dtau_dv_I

    def _compute_domega_dot_domega(self, omega_B: np.ndarray, J_B: np.ndarray, J_B_inv: np.ndarray) -> np.ndarray:
        """
        Compute ∂ω̇/∂ω.

        From gyroscopic term: -J^{-1} @ (ω x Jω)
        """
        # τ_gyro = -ω x Jω
        # ∂τ_gyro/∂ω = -skew(ω) @ J + skew(Jω)
        #            = skew(Jω) - skew(ω) @ J

        J_omega = J_B @ omega_B

        dtau_gyro_domega = skew(J_omega) - skew(omega_B) @ J_B

        return J_B_inv @ dtau_gyro_domega

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Control Jacobian: ∂f/∂u.

        Structure (14x3):
            [∂ṁ/∂T  ]   [-alpha·T/‖T‖]
            [∂ṙ/∂T  ] = [0       ]
            [∂v̇/∂T  ]   [C_IB/m  ]
            [∂q̇/∂T  ]   [0       ]
            [∂ω̇/∂T  ]   [J^{-1}·skew(r_T)]

        Parameters
        ----------
        x : np.ndarray, shape (14,)
            State vector.
        u : np.ndarray, shape (3,)
            Control vector.

        Returns
        -------
        np.ndarray, shape (14, 3)
            Control Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        m = x[0]
        q = x[7:11]
        T_B = u
        T_mag = np.linalg.norm(T_B)

        p = self.params
        r_T_B = p.r_T_B
        J_B_inv = p.J_B_inv

        C_BI = quat_to_dcm(q)
        C_IB = C_BI.T

        B = np.zeros((14, 3))

        # --- ∂ṁ/∂T = -alpha · T/‖T‖ ---
        if T_mag > 1e-10:
            B[0, :] = -p.alpha * T_B / T_mag

        # --- ∂v̇/∂T = C_IB / m ---
        B[4:7, :] = C_IB / m

        # --- ∂ω̇/∂T = J^{-1} @ skew(r_T) ---
        # τ_thrust = r_T x T = -T x r_T = -skew(T) @ r_T = skew(r_T) @ T
        # ∂τ/∂T = skew(r_T)
        B[11:14, :] = J_B_inv @ skew(r_T_B)

        return B

    # =========================================================================
    # Constraints
    # =========================================================================

    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds."""
        p = self.params

        lb = np.array(
            [
                p.m_dry,  # mass
                -np.inf,
                -np.inf,
                -np.inf,  # position
                -p.v_max,
                -p.v_max,
                -p.v_max,  # velocity
                -1.0,
                -1.0,
                -1.0,
                -1.0,  # quaternion
                -p.omega_max,
                -p.omega_max,
                -p.omega_max,  # angular velocity
            ]
        )

        ub = np.array(
            [
                p.m_wet,  # mass
                np.inf,
                np.inf,
                np.inf,  # position
                p.v_max,
                p.v_max,
                p.v_max,  # velocity
                1.0,
                1.0,
                1.0,
                1.0,  # quaternion
                p.omega_max,
                p.omega_max,
                p.omega_max,  # angular velocity
            ]
        )

        return lb, ub

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control bounds (component-wise, actual constraint is on magnitude)."""
        T_max = self.params.T_max
        lb = np.array([-T_max, -T_max, -T_max])
        ub = np.array([T_max, T_max, T_max])
        return lb, ub

    def thrust_constraint(self, u: np.ndarray) -> dict:
        """Evaluate thrust magnitude constraints."""
        T_mag = np.linalg.norm(u)
        p = self.params
        return {
            "thrust_min": p.T_min - T_mag,
            "thrust_max": T_mag - p.T_max,
        }

    def gimbal_constraint(self, u: np.ndarray) -> float:
        """
        Evaluate gimbal angle constraint.

        cos(δ) ≥ cos(δ_max)  =>  T_x / ‖T‖ ≥ cos(δ_max)

        Returns
        -------
        float
            Constraint value (negative = satisfied).
        """
        delta = self.get_gimbal_angle(u)
        return delta - self.params.delta_max

    def tilt_constraint(self, x: np.ndarray) -> float:
        """
        Evaluate tilt angle constraint.

        θ ≤ θ_max

        Returns
        -------
        float
            Constraint value (negative = satisfied).
        """
        theta = self.get_tilt_angle(x)
        return theta - self.params.theta_max

    def glide_slope_constraint(self, x: np.ndarray) -> float:
        """
        Evaluate glide slope constraint.

        ‖r_yz‖ ≤ r_x · tan(gamma_gs)

        Returns
        -------
        float
            Constraint value (negative = satisfied).
        """
        r = self.get_position(x)
        r_x = r[0]  # altitude
        r_yz = r[1:3]  # horizontal

        gamma = self.params.gamma_gs
        return np.linalg.norm(r_yz) - r_x * np.tan(gamma)

    def angular_rate_constraint(self, x: np.ndarray) -> float:
        """
        Evaluate angular velocity magnitude constraint.

        ‖ω‖ ≤ ω_max

        Returns
        -------
        float
            Constraint value (negative = satisfied).
        """
        omega = self.get_omega(x)
        return np.linalg.norm(omega) - self.params.omega_max

    def is_thrust_valid(self, u: np.ndarray) -> bool:
        """Check if thrust satisfies magnitude constraints."""
        constraints = self.thrust_constraint(u)
        return all(v <= 1e-10 for v in constraints.values())

    def is_gimbal_valid(self, u: np.ndarray) -> bool:
        """Check if gimbal angle is within bounds."""
        return self.gimbal_constraint(u) <= 1e-10

    def is_tilt_valid(self, x: np.ndarray) -> bool:
        """Check if tilt angle is within bounds."""
        return self.tilt_constraint(x) <= 1e-10

    def is_glide_slope_satisfied(self, x: np.ndarray) -> bool:
        """Check if state satisfies glide slope constraint."""
        return self.glide_slope_constraint(x) <= 1e-10

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def fuel_remaining(self, x: np.ndarray) -> float:
        """Compute remaining fuel mass."""
        return self.get_mass(x) - self.params.m_dry

    def fuel_fraction(self, x: np.ndarray) -> float:
        """Compute remaining fuel as fraction of initial fuel."""
        return self.fuel_remaining(x) / self.params.fuel_mass

    def hover_thrust(self, x: np.ndarray) -> np.ndarray:
        """
        Compute thrust required to hover.

        Returns thrust vector in body frame.
        """
        m = self.get_mass(x)
        q = self.get_quaternion(x)
        g_I = self.params.g_I

        # Required force in inertial frame to counteract gravity
        F_required_I = -m * g_I

        # Transform to body frame
        C_BI = quat_to_dcm(q)
        T_hover_B = C_BI @ F_required_I

        return T_hover_B

    def normalize_quaternion(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the quaternion in the state vector.

        Parameters
        ----------
        x : np.ndarray, shape (14,)
            State vector.

        Returns
        -------
        np.ndarray, shape (14,)
            State with normalized quaternion.
        """
        x = np.asarray(x, dtype=np.float64).copy()
        q = x[7:11]
        x[7:11] = quat_normalize(q)
        return x

    def energy(self, x: np.ndarray) -> dict:
        """Compute kinetic and potential energy."""
        m = self.get_mass(x)
        r = self.get_position(x)
        v = self.get_velocity(x)
        omega = self.get_omega(x)
        g = self.params.g_I
        J = self.params.J_B

        # Translational kinetic energy
        KE_trans = 0.5 * m * np.dot(v, v)

        # Rotational kinetic energy
        KE_rot = 0.5 * np.dot(omega, J @ omega)

        # Potential energy
        PE = -m * np.dot(g, r)

        return {
            "kinetic_translational": KE_trans,
            "kinetic_rotational": KE_rot,
            "kinetic_total": KE_trans + KE_rot,
            "potential": PE,
            "total": KE_trans + KE_rot + PE,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_rocket6dof(
    m_dry: float = 1.0,
    m_wet: float = 2.0,
    I_sp: float = 30.0,
    T_max: float = 6.5,
    T_min: float = 1.5,
    enable_aero: bool = False,
) -> Rocket6DoF:
    """
    Create a Rocket6DoF with specified parameters.

    Uses Szmuk-style normalized parameters as base.
    """
    params = Rocket6DoFParams(
        m_dry=m_dry,
        m_wet=m_wet,
        I_sp=I_sp,
        T_max=T_max,
        T_min=T_min,
        enable_aero=enable_aero,
    )
    return Rocket6DoF(params)


def create_szmuk_rocket6dof(enable_aero: bool = False) -> Rocket6DoF:
    """
    Create a Rocket6DoF with exact Szmuk et al. (2018) parameters.
    """
    params = szmuk_params()
    params.enable_aero = enable_aero
    return Rocket6DoF(params)
