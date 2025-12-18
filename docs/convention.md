# simdyn Conventions

This document describes all conventions used in simdyn, including coordinate frames, quaternions, units, and state vector layouts.

## Table of Contents

1. [Coordinate Frames](#coordinate-frames)
2. [Quaternion Convention](#quaternion-convention)
3. [Units](#units)
4. [State Vector Conventions](#state-vector-conventions)
5. [Dynamics Conventions](#dynamics-conventions)

---

## Coordinate Frames

### Inertial Frames

simdyn uses different inertial frame conventions depending on the application domain:

#### UEN (Up-East-North) - Aerospace

Used for rocket systems (Rocket3DoF, Rocket6DoF).

```
    x (Up)
    ↑
    │
    │
    └────→ y (East)
   ╱
  ↙
 z (North)
```

- **x-axis**: Points up (opposite to gravity)
- **y-axis**: Points East
- **z-axis**: Points North
- **Origin**: Typically at landing site

This convention follows Szmuk et al. (2018) for powered descent guidance.

#### NED (North-East-Down) - Aviation

Alternative for aircraft and some quadrotor applications.

```
    x (North)
    ↑
    │
    │
    └────→ y (East)
         ╲
          ↘
           z (Down)
```

#### 2D Plane

Used for ground vehicles (Unicycle, Bicycle).

```
    y
    ↑
    │
    │
    └────→ x
```

- **x-axis**: Forward/horizontal
- **y-axis**: Left/vertical
- **θ = 0**: Vehicle facing positive x direction

### Body Frames

#### Rocket Body Frame

```
    x_B (thrust axis)
    ↑
    │
    │
    └────→ y_B
   ╱
  ↙
 z_B
```

- **x-axis**: Along vehicle centerline (thrust direction when gimbal = 0)
- **y, z-axes**: Complete right-handed frame
- **Origin**: Center of mass

#### Quadrotor Body Frame

```
    x_B (forward)
    ↑
    │     
    │ ⊙ z_B (up)
    │
    └────→ y_B (right)
```

### Frame Transformations

The Direction Cosine Matrix (DCM) `C_BI` transforms vectors from inertial to body frame:

```
v_B = C_BI @ v_I
```

To transform from body to inertial:

```
v_I = C_BI.T @ v_B = C_IB @ v_B
```

---

## Quaternion Convention

### Scalar-First Format

simdyn uses **scalar-first** quaternions:

```
q = [q_w, q_x, q_y, q_z] = [cos(θ/2), sin(θ/2)·n]
```

where `θ` is the rotation angle and `n` is the unit rotation axis.

### Identity Quaternion

```python
q_identity = [1, 0, 0, 0]  # No rotation
```

### Quaternion Semantics

The quaternion `q_BI` represents the rotation from inertial frame to body frame:

- A vector in inertial frame `v_I` is transformed to body frame via:
  ```
  v_B = q_BI ⊗ v_I ⊗ q_BI*
  ```
  
- Equivalently using the DCM:
  ```
  v_B = C(q_BI) @ v_I
  ```

### Quaternion Operations

```python
from simdyn.utils.quaternion import (
    quat_multiply,      # Hamilton product
    quat_conjugate,     # q* = [q_w, -q_x, -q_y, -q_z]
    quat_normalize,     # Unit quaternion
    quat_to_dcm,        # Convert to 3×3 rotation matrix
    quat_from_dcm,      # Convert from rotation matrix
    quat_to_euler,      # Convert to Euler angles
    quat_from_euler,    # Convert from Euler angles
    quat_rotate,        # Rotate vector by quaternion
    quat_from_axis_angle,  # Create from axis-angle
    omega_matrix,       # Ω(ω) matrix for kinematics
)
```

### Quaternion Kinematics

The quaternion rate equation:

```
q̇ = (1/2) · Ω(ω) · q
```

where `Ω(ω)` is the 4×4 matrix:

```
       ⎡ 0   -ωx  -ωy  -ωz ⎤
Ω(ω) = ⎢ ωx   0    ωz  -ωy ⎥
       ⎢ ωy  -ωz   0    ωx ⎥
       ⎣ ωz   ωy  -ωx   0  ⎦
```

---

## Units

### SI Units (Default)

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | meters | m |
| Time | seconds | s |
| Mass | kilograms | kg |
| Angle | radians | rad |
| Force | Newtons | N |
| Torque | Newton-meters | N·m |
| Velocity | meters/second | m/s |
| Angular velocity | radians/second | rad/s |
| Acceleration | meters/second² | m/s² |

### Normalized Units

For algorithm testing, normalized (non-dimensional) parameters are available:

```python
# Pendulum: m=l=g=1 gives ωn=1, T=2π
pendulum = ds.create_normalized_pendulum()

# Rocket: Szmuk normalized parameters
rocket = ds.create_normalized_rocket3dof()
```

The Szmuk normalization uses:
- `U_L = 10 m` (length)
- `U_T = 2 s` (time)
- `U_M = 5000 kg` (mass)

---

## State Vector Conventions

### Double Integrator (1D)

```
x = [position, velocity]
    [   p    ,    v    ]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | p | Position | m |
| 1 | v | Velocity | m/s |

### Unicycle

```
x = [x, y, θ]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | x-position | m |
| 1 | y | y-position | m |
| 2 | θ | Heading angle | rad |

- θ = 0: Facing positive x direction
- θ > 0: Counter-clockwise from x-axis

### Pendulum

```
x = [θ, ω]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | θ | Angle from down | rad |
| 1 | ω | Angular velocity | rad/s |

- θ = 0: Hanging down (stable)
- θ = π: Pointing up (unstable)

### Cart-Pole

```
x = [x, ẋ, θ, θ̇]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x | Cart position | m |
| 1 | ẋ | Cart velocity | m/s |
| 2 | θ | Pole angle from up | rad |
| 3 | θ̇ | Pole angular velocity | rad/s |

- θ = 0: Pole pointing up (unstable)
- θ = π: Pole pointing down (stable)

### Rocket 3-DoF

```
x = [m, r_x, r_y, r_z, v_x, v_y, v_z]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | m | Mass | kg |
| 1-3 | r_I | Position (inertial) | m |
| 4-6 | v_I | Velocity (inertial) | m/s |

Control: `u = [T_x, T_y, T_z]` - Thrust vector in inertial frame

### Rocket 6-DoF

```
x = [m, r_x, r_y, r_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, ω_x, ω_y, ω_z]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | m | Mass | kg |
| 1-3 | r_I | Position (inertial) | m |
| 4-6 | v_I | Velocity (inertial) | m/s |
| 7-10 | q_BI | Attitude quaternion | - |
| 11-13 | ω_B | Angular velocity (body) | rad/s |

Control: `u = [T_Bx, T_By, T_Bz]` - Thrust vector in body frame

---

## Dynamics Conventions

### Continuous-Time Dynamics

All systems implement:

```python
x_dot = f(x, u, w)
```

where:
- `x`: State vector (n,)
- `u`: Control vector (m,)
- `w`: Disturbance vector (n_disturbance,), optional

### Discrete-Time Dynamics

```python
x_next = f_discrete(x, u, dt, w, method='rk4')
```

Available integration methods:
- `'euler'`: Forward Euler
- `'rk4'`: 4th-order Runge-Kutta (default)

### Jacobian Definitions

State Jacobian:
```
A(x, u) = ∂f/∂x
```

Control Jacobian:
```
B(x, u) = ∂f/∂u
```

Linearization about (x₀, u₀):
```
ẋ ≈ A(x - x₀) + B(u - u₀) + c
```

where `c = f(x₀, u₀)`.

### Disturbance Conventions

Disturbances enter additively:

```
ẋ = f(x, u) + G·w
```

For most systems, `G = I` (identity), so disturbance dimension equals state dimension.

### Constraint Conventions

Constraints are expressed as:
- **State bounds**: `x_lb ≤ x ≤ x_ub`
- **Control bounds**: `u_lb ≤ u ≤ u_ub`
- **Nonlinear constraints**: `g(x, u) ≤ 0` (negative = satisfied)

Example:
```python
# Check constraint satisfaction
thrust_constraint = rocket.thrust_constraint(u)
# Returns {'thrust_min': ..., 'thrust_max': ...}
# Negative values mean constraint is satisfied
```

---

## Sign Conventions Summary

| System | Angle Convention | Positive Control |
|--------|------------------|------------------|
| Pendulum | θ=0 down, θ=π up | τ>0 counter-clockwise |
| CartPole | θ=0 up, θ=π down | F>0 pushes cart right |
| Unicycle | θ=0 facing +x | ω>0 counter-clockwise |
| Rocket | UEN frame, x=up | Thrust in specified frame |