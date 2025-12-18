# simdyn API Reference

Complete API documentation for the simdyn library.

## Table of Contents

1. [Base Class](#base-class)
2. [Systems](#systems)
3. [Integrators](#integrators)
4. [Utilities](#utilities)
5. [Factory Functions](#factory-functions)

---

## Base Class

### `DynamicalSystem`

Abstract base class for all dynamical systems.

```python
from simdyn import DynamicalSystem
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_state` | `int` | Dimension of state vector |
| `n_control` | `int` | Dimension of control vector |
| `n_disturbance` | `int` | Dimension of disturbance vector |
| `state_names` | `List[str]` | Human-readable names for states |
| `control_names` | `List[str]` | Human-readable names for controls |
| `params` | `dataclass` | System parameters |

#### Core Methods

##### `f(x, u, w=None) → np.ndarray`

Continuous-time dynamics.

**Parameters:**
- `x`: State vector, shape `(n_state,)`
- `u`: Control vector, shape `(n_control,)`
- `w`: Disturbance vector, shape `(n_disturbance,)`, optional

**Returns:**
- State derivative `ẋ`, shape `(n_state,)`

**Example:**
```python
x = np.array([0.1, 0.0])
u = np.array([0.5])
x_dot = system.f(x, u)
```

##### `f_discrete(x, u, dt, w=None, method='rk4') → np.ndarray`

Discrete-time dynamics.

**Parameters:**
- `x`: Current state, shape `(n_state,)`
- `u`: Control input, shape `(n_control,)`
- `dt`: Time step (scalar)
- `w`: Disturbance, shape `(n_disturbance,)`, optional
- `method`: Integration method (`'euler'` or `'rk4'`)

**Returns:**
- Next state `x_{k+1}`, shape `(n_state,)`

##### `A(x, u) → np.ndarray`

State Jacobian matrix ∂f/∂x.

**Parameters:**
- `x`: State vector
- `u`: Control vector

**Returns:**
- Jacobian matrix, shape `(n_state, n_state)`

##### `B(x, u) → np.ndarray`

Control Jacobian matrix ∂f/∂u.

**Parameters:**
- `x`: State vector
- `u`: Control vector

**Returns:**
- Jacobian matrix, shape `(n_state, n_control)`

##### `linearize(x0, u0) → Tuple[np.ndarray, np.ndarray, np.ndarray]`

Linearize dynamics at operating point.

**Parameters:**
- `x0`: Operating point state
- `u0`: Operating point control

**Returns:**
- `A`: State matrix at operating point
- `B`: Control matrix at operating point
- `c`: Affine term `f(x0, u0)`

##### `simulate(x0, controller, t_span, dt, disturbance_fn=None, method='rk4') → Tuple`

Run closed-loop simulation.

**Parameters:**
- `x0`: Initial state
- `controller`: Function `(t, x) → u`
- `t_span`: Tuple `(t_start, t_end)`
- `dt`: Time step
- `disturbance_fn`: Optional function `(t, x) → w`
- `method`: Integration method

**Returns:**
- `t`: Time array, shape `(N,)`
- `x`: State trajectory, shape `(N, n_state)`
- `u`: Control history, shape `(N-1, n_control)`

**Example:**
```python
def my_controller(t, x):
    return -K @ x

t, x, u = system.simulate(x0, my_controller, (0, 10), dt=0.01)
```

##### `verify_jacobians(x, u, eps=1e-7, tol=1e-5) → Tuple[bool, dict]`

Verify analytical Jacobians against numerical differentiation.

**Parameters:**
- `x`: Test state
- `u`: Test control
- `eps`: Finite difference step size
- `tol`: Error tolerance

**Returns:**
- `passed`: Boolean, True if verification passed
- `errors`: Dictionary with error details

##### `get_state_bounds() → Tuple[np.ndarray, np.ndarray]`

Get state bounds.

**Returns:**
- `lb`: Lower bounds, shape `(n_state,)`
- `ub`: Upper bounds, shape `(n_state,)`

##### `get_control_bounds() → Tuple[np.ndarray, np.ndarray]`

Get control bounds.

**Returns:**
- `lb`: Lower bounds, shape `(n_control,)`
- `ub`: Upper bounds, shape `(n_control,)`

---

## Systems

### `DoubleIntegrator`

Linear point-mass system.

```python
from simdyn import DoubleIntegrator, DoubleIntegratorParams
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 1 | Spatial dimension (1, 2, or 3) |
| `mass` | float | 1.0 | Point mass |
| `damping` | float | 0.0 | Velocity damping coefficient |
| `u_max` | float | inf | Maximum control magnitude |
| `x_max` | float | inf | Maximum position |
| `v_max` | float | inf | Maximum velocity |

#### Convenience Classes

```python
DoubleIntegrator1D()  # 1D: n=2, m=1
DoubleIntegrator2D()  # 2D: n=4, m=2
DoubleIntegrator3D()  # 3D: n=6, m=3
```

---

### `Unicycle`

Nonholonomic 2D mobile robot.

```python
from simdyn import Unicycle, UnicycleParams
```

#### State: `[x, y, θ]`
#### Control: `[v, ω]` (linear velocity, angular velocity)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `v_max` | float | inf | Maximum linear velocity |
| `v_min` | float | -inf | Minimum linear velocity |
| `omega_max` | float | inf | Maximum angular velocity |

#### Methods

```python
# State accessors
x, y = unicycle.get_position(state)
theta = unicycle.get_heading(state)
state = unicycle.pack_state(x, y, theta)

# Utilities
dist = unicycle.distance_to_point(state, target)
heading = unicycle.heading_to_point(state, target)
error = unicycle.heading_error(state, target)
```

---

### `Pendulum`

Simple pendulum (point mass on massless rod).

```python
from simdyn import Pendulum, PendulumParams
```

#### State: `[θ, ω]`
#### Control: `[τ]` (torque at pivot)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m` | float | 1.0 | Point mass |
| `l` | float | 1.0 | Pendulum length |
| `g` | float | 9.81 | Gravitational acceleration |
| `b` | float | 0.0 | Damping coefficient |
| `tau_max` | float | inf | Maximum torque |

#### Derived Parameters

```python
params.inertia           # I = m·l²
params.natural_frequency # ωn = √(g/l)
params.period            # T = 2π/ωn
params.damping_ratio     # ζ = b/(2·I·ωn)
```

#### Methods

```python
# Energy
E = pendulum.energy(x)  # Returns {'kinetic', 'potential', 'total'}
E_top = pendulum.energy_at_top()
E_bottom = pendulum.energy_at_bottom()

# Equilibrium
x_eq = pendulum.equilibrium(u)

# Linearization
A, B = pendulum.linearize_at_bottom()
A, B = pendulum.linearize_at_top()

# Utilities
is_up = pendulum.is_upright(x, tol=0.1)
is_down = pendulum.is_at_bottom(x, tol=0.1)
pos = pendulum.get_position(x)  # Cartesian position of bob
```

---

### `CartPole`

Inverted pendulum on cart (underactuated).

```python
from simdyn import CartPole, CartPoleParams
```

#### State: `[x, ẋ, θ, θ̇]`
#### Control: `[F]` (force on cart)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `M` | float | 1.0 | Cart mass |
| `m` | float | 0.1 | Pole mass |
| `l` | float | 0.5 | Pole half-length |
| `g` | float | 9.81 | Gravitational acceleration |
| `b_c` | float | 0.0 | Cart friction |
| `b_p` | float | 0.0 | Pole friction |
| `x_max` | float | 2.4 | Track half-length |
| `F_max` | float | 10.0 | Maximum force |

#### Methods

```python
# State accessors
cart_x = cartpole.get_cart_position(x)
cart_v = cartpole.get_cart_velocity(x)
theta = cartpole.get_pole_angle(x)
omega = cartpole.get_pole_angular_velocity(x)
x = cartpole.pack_state(cart_x, cart_v, theta, omega)

# Pole tip
pos = cartpole.get_pole_tip_position(x)
vel = cartpole.get_pole_tip_velocity(x)

# Energy
E = cartpole.energy(x)
E_up = cartpole.energy_upright()
E_down = cartpole.energy_down()

# Linearization
A, B = cartpole.linearize_upright()
A, B = cartpole.linearize_down()

# Status
cartpole.is_balanced(x, tol=0.1)
cartpole.is_fallen(x, tol=0.5)
cartpole.is_within_track(x)
```

---

### `Rocket3DoF`

Point-mass rocket for powered descent.

```python
from simdyn import Rocket3DoF, Rocket3DoFParams
```

#### State: `[m, r_x, r_y, r_z, v_x, v_y, v_z]` (n=7)
#### Control: `[T_x, T_y, T_z]` - Thrust in inertial frame (m=3)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m_dry` | float | 1.0 | Dry mass |
| `m_wet` | float | 2.0 | Wet mass (with fuel) |
| `I_sp` | float | 30.0 | Specific impulse |
| `g0` | float | 1.0 | Reference gravity |
| `g_I` | array | [-1,0,0] | Gravity vector |
| `T_min` | float | 0.0 | Minimum thrust |
| `T_max` | float | 6.5 | Maximum thrust |
| `enable_drag` | bool | False | Enable aerodynamic drag |
| `gamma_gs` | float | 30° | Glide slope angle |

#### Methods

```python
# State accessors
m = rocket.get_mass(x)
r = rocket.get_position(x)
v = rocket.get_velocity(x)
alt = rocket.get_altitude(x)
spd = rocket.get_speed(x)
x = rocket.pack_state(mass, position, velocity)

# Thrust utilities
T_mag = rocket.get_thrust_magnitude(u)
T_dir = rocket.get_thrust_direction(u)

# Fuel
fuel = rocket.fuel_remaining(x)
frac = rocket.fuel_fraction(x)

# Constraints
rocket.thrust_constraint(u)
rocket.glide_slope_constraint(x)
rocket.is_thrust_valid(u)
rocket.is_glide_slope_satisfied(x)

# Utilities
T_hover = rocket.hover_thrust(x)
tof = rocket.time_of_flight_estimate(x)
E = rocket.energy(x)
```

---

### `Rocket6DoF`

Rigid-body rocket with attitude dynamics.

```python
from simdyn import Rocket6DoF, Rocket6DoFParams
```

#### State: `[m, r_I(3), v_I(3), q_BI(4), ω_B(3)]` (n=14)
#### Control: `[T_Bx, T_By, T_Bz]` - Thrust in body frame (m=3)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m_dry` | float | 1.0 | Dry mass |
| `m_wet` | float | 2.0 | Wet mass |
| `J_B` | array | diag([.02,1,1])*.168 | Inertia tensor |
| `I_sp` | float | 30.0 | Specific impulse |
| `g_I` | array | [-1,0,0] | Gravity vector |
| `r_T_B` | array | [-.25,0,0] | Thrust application point |
| `r_cp_B` | array | [.05,0,0] | Center of pressure |
| `enable_aero` | bool | False | Enable aerodynamics |
| `T_min` | float | 1.5 | Minimum thrust |
| `T_max` | float | 6.5 | Maximum thrust |
| `delta_max` | float | 20° | Maximum gimbal angle |
| `theta_max` | float | 90° | Maximum tilt angle |
| `gamma_gs` | float | 30° | Glide slope angle |
| `omega_max` | float | 60°/s | Maximum angular rate |

#### Methods

```python
# State accessors
m = rocket.get_mass(x)
r = rocket.get_position(x)
v = rocket.get_velocity(x)
q = rocket.get_quaternion(x)
omega = rocket.get_omega(x)
C = rocket.get_dcm(x)
x = rocket.pack_state(mass, position, velocity, quaternion, omega)

# Attitude utilities
delta = rocket.get_gimbal_angle(u)
theta = rocket.get_tilt_angle(x)
T_mag = rocket.get_thrust_magnitude(u)
T_dir = rocket.get_thrust_direction_body(u)

# Aerodynamics
F_aero = rocket.compute_aero_force_body(v_I, C_BI)
tau_aero = rocket.compute_aero_torque_body(F_aero_B)

# Constraints
rocket.thrust_constraint(u)
rocket.gimbal_constraint(u)
rocket.tilt_constraint(x)
rocket.glide_slope_constraint(x)
rocket.angular_rate_constraint(x)

# Utilities
T_hover = rocket.hover_thrust(x)
x_norm = rocket.normalize_quaternion(x)
E = rocket.energy(x)
```

---

## Integrators

### `euler_step(f, x, u, dt, w=None)`

Forward Euler integration step.

```python
from simdyn.integrators import euler_step

x_next = euler_step(dynamics_fn, x, u, dt)
```

### `rk4_step(f, x, u, dt, w=None)`

4th-order Runge-Kutta integration step.

```python
from simdyn.integrators import rk4_step

x_next = rk4_step(dynamics_fn, x, u, dt)
```

---

## Utilities

### Rotation Utilities

```python
from simdyn.utils.rotations import (
    skew,           # Skew-symmetric matrix from vector
    vee,            # Vector from skew-symmetric matrix
    rot_x,          # Rotation matrix about x-axis
    rot_y,          # Rotation matrix about y-axis
    rot_z,          # Rotation matrix about z-axis
    euler_to_dcm,   # Euler angles to DCM
    dcm_to_euler,   # DCM to Euler angles
    wrap_angle,     # Wrap angle to [-π, π]
)
```

### Quaternion Utilities

```python
from simdyn.utils.quaternion import (
    quat_identity,         # Returns [1, 0, 0, 0]
    quat_normalize,        # Normalize to unit quaternion
    quat_conjugate,        # q* = [w, -x, -y, -z]
    quat_multiply,         # Hamilton product q1 ⊗ q2
    quat_inverse,          # q^{-1} = q* / |q|²
    quat_to_dcm,          # Convert to 3×3 rotation matrix
    quat_from_dcm,        # Convert from rotation matrix
    quat_to_euler,        # Convert to [roll, pitch, yaw]
    quat_from_euler,      # Convert from Euler angles
    quat_to_axis_angle,   # Convert to (axis, angle)
    quat_from_axis_angle, # Create from axis and angle
    quat_rotate,          # Rotate vector by quaternion
    quat_rotate_inverse,  # Inverse rotation
    omega_matrix,         # Ω(ω) for kinematics
    quat_error,           # Error quaternion q2 * q1^{-1}
    quat_angle,           # Extract rotation angle
)
```

---

## Factory Functions

### Pendulum

```python
create_pendulum(m=1.0, l=1.0, g=9.81, b=0.0, tau_max=inf)
create_damped_pendulum(damping_ratio=0.1, m=1.0, l=1.0, g=9.81)
create_normalized_pendulum()  # m=l=g=1
```

### CartPole

```python
create_cartpole(M=1.0, m=0.1, l=0.5, g=9.81, b_c=0, b_p=0, F_max=10)
create_gym_cartpole()        # OpenAI Gym compatible
create_normalized_cartpole() # Normalized parameters
```

### Unicycle

```python
create_unicycle(v_max=inf, omega_max=inf)
forward_only_unicycle(v_max=1.0)  # v ≥ 0 only
```

### Rocket3DoF

```python
create_rocket3dof(m_dry=1.0, m_wet=2.0, I_sp=30.0, ...)
create_normalized_rocket3dof()  # Szmuk normalized
```

### Rocket6DoF

```python
create_rocket6dof(m_dry=1.0, m_wet=2.0, I_sp=30.0, ...)
create_szmuk_rocket6dof(enable_aero=False)  # Szmuk (2018) params
```