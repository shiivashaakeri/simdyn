# Rocket 6-DoF System Documentation

Rigid-body rocket model for powered descent guidance and control, matching the Szmuk et al. (2018) formulation.

## Overview

The Rocket6DoF system models a rocket with full translational and rotational dynamics. It includes:

- Mass depletion from fuel consumption
- Quaternion-based attitude representation
- Thrust vector control (gimbaling)
- Aerodynamic forces and torques (optional)
- Comprehensive constraint handling

## Mathematical Model

### State Vector (n=14)

```
x = [m, r_I, v_I, q_BI, ω_B]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | m | Mass | kg |
| 1-3 | r_I | Position (inertial frame) | m |
| 4-6 | v_I | Velocity (inertial frame) | m/s |
| 7-10 | q_BI | Attitude quaternion [w,x,y,z] | - |
| 11-13 | ω_B | Angular velocity (body frame) | rad/s |

### Control Vector (m=3)

```
u = [T_Bx, T_By, T_Bz]
```

Thrust vector in body frame. When gimbal angle is zero, thrust is along body x-axis.

### Dynamics Equations

#### Mass Depletion
```
ṁ = -α·‖T_B‖
```
where α = 1/(g₀·I_sp) is the mass flow rate coefficient.

#### Translational Dynamics
```
ṙ_I = v_I
v̇_I = (1/m)·C_IB·T_B + g_I + (1/m)·F_aero_I
```

#### Rotational Dynamics
```
q̇_BI = (1/2)·Ω(ω_B)·q_BI
ω̇_B = J⁻¹·(τ_thrust + τ_aero + τ_gyro)
```

where:
- τ_thrust = r_T × T_B (thrust torque)
- τ_aero = r_cp × F_aero_B (aerodynamic torque)
- τ_gyro = -ω × J·ω (gyroscopic torque)

## Coordinate Frames

### Inertial Frame (I)

Up-East-North (UEN) convention:
- **x-axis**: Up (opposite to gravity)
- **y-axis**: East
- **z-axis**: North
- **Origin**: Landing site

### Body Frame (B)

- **x-axis**: Along rocket centerline (thrust direction when δ=0)
- **y-axis**: Out the side
- **z-axis**: Completes right-hand frame
- **Origin**: Center of mass

### Frame Transformation

The quaternion q_BI transforms vectors from inertial to body frame:
```
v_B = C(q_BI) · v_I
```

## Parameters

### Szmuk et al. (2018) Parameters

```python
params = Rocket6DoFParams(
    m_dry=1.0,           # Dry mass
    m_wet=2.0,           # Wet mass (with fuel)
    J_B=diag([0.02, 1.0, 1.0]) * 0.168,  # Inertia tensor
    I_sp=30.0,           # Specific impulse
    g0=1.0,              # Reference gravity
    g_I=[-1, 0, 0],      # Gravity vector
    r_T_B=[-0.25, 0, 0], # Thrust application point
    r_cp_B=[0.05, 0, 0], # Center of pressure
    T_min=1.5,           # Minimum thrust
    T_max=6.5,           # Maximum thrust
    delta_max=20°,       # Maximum gimbal angle
    theta_max=90°,       # Maximum tilt angle
    gamma_gs=30°,        # Glide slope angle
    omega_max=60°/s,     # Maximum angular rate
)
```

### Non-Dimensional Units

The Szmuk parameters use scaled units:
- Length: U_L = 10 m
- Time: U_T = 2 s
- Mass: U_M = 5000 kg

## Constraints

### Thrust Magnitude
```
T_min ≤ ‖T_B‖ ≤ T_max
```

### Gimbal Angle
```
cos(δ) = T_Bx / ‖T_B‖ ≥ cos(δ_max)
```
The gimbal angle is the angle between thrust vector and body x-axis.

### Tilt Angle
```
cos(θ) = e_x^B · e_x^I ≥ cos(θ_max)
```
The tilt angle measures how far the rocket has tilted from vertical.

### Glide Slope
```
‖r_yz‖ ≤ r_x · tan(γ_gs)
```
Keeps the rocket within a cone centered on the landing site.

### Angular Rate
```
‖ω_B‖ ≤ ω_max
```

## Aerodynamic Model

When `enable_aero=True`, aerodynamic forces are included.

### Spherical Model
```
F_aero_B = -0.5 · ρ · S · c_a · ‖v_B‖ · v_B
```

### Ellipsoidal Model (Szmuk)
```
F_aero_B = -0.5 · ρ · S · C_A · v_B · ‖v_B‖
```

The ellipsoidal model uses a 3×3 coefficient matrix C_A to account for different drag in axial vs lateral directions.

## Usage Examples

### Basic Simulation

```python
import numpy as np
from simdyn import create_szmuk_rocket6dof
from simdyn.utils.quaternion import quat_identity

# Create system
rocket = create_szmuk_rocket6dof()

# Initial state: hovering at altitude 10
x0 = rocket.pack_state(
    mass=2.0,
    position=np.array([10.0, 0.0, 0.0]),
    velocity=np.zeros(3),
    quaternion=quat_identity(),
    omega=np.zeros(3)
)

# Hover controller
def hover_controller(t, x):
    return rocket.hover_thrust(x)

# Simulate
t, x, u = rocket.simulate(x0, hover_controller, (0, 5), dt=0.01)
```

### Checking Constraints

```python
# Check all constraints
thrust_ok = rocket.is_thrust_valid(u)
gimbal_ok = rocket.is_gimbal_valid(u)
tilt_ok = rocket.is_tilt_valid(x)
glide_ok = rocket.is_glide_slope_satisfied(x)

# Get constraint values (negative = satisfied)
thrust_viol = rocket.thrust_constraint(u)
gimbal_viol = rocket.gimbal_constraint(u)
tilt_viol = rocket.tilt_constraint(x)
glide_viol = rocket.glide_slope_constraint(x)
omega_viol = rocket.angular_rate_constraint(x)
```

### Landing Trajectory

```python
import numpy as np
from simdyn import create_szmuk_rocket6dof
from simdyn.utils.quaternion import quat_identity

rocket = create_szmuk_rocket6dof()

# Initial: high altitude, falling, tilted
q0 = quat_from_axis_angle(np.array([0, 1, 0]), 0.2)  # slight tilt
x0 = rocket.pack_state(
    mass=2.0,
    position=np.array([20.0, 5.0, 0.0]),
    velocity=np.array([-5.0, 2.0, 0.0]),
    quaternion=q0,
    omega=np.zeros(3)
)

def landing_controller(t, x):
    r = rocket.get_position(x)
    v = rocket.get_velocity(x)
    m = rocket.get_mass(x)
    
    # Target: soft landing at origin
    r_target = np.zeros(3)
    v_target = np.zeros(3)
    
    # PD control in inertial frame
    Kp, Kd = 0.5, 1.0
    F_des_I = m * (Kp * (r_target - r) + Kd * (v_target - v) - rocket.params.g_I)
    
    # Transform to body frame
    C_BI = rocket.get_dcm(x)
    T_B = C_BI @ F_des_I
    
    # Clamp thrust
    T_mag = np.linalg.norm(T_B)
    T_mag = np.clip(T_mag, rocket.params.T_min, rocket.params.T_max)
    if np.linalg.norm(T_B) > 1e-6:
        T_B = T_B / np.linalg.norm(T_B) * T_mag
    
    return T_B

t, x, u = rocket.simulate(x0, landing_controller, (0, 15), dt=0.01)
```

## Jacobians

### State Jacobian A (14×14)

The analytical state Jacobian includes:
- Position-velocity coupling (∂ṙ/∂v = I)
- Thrust acceleration dependence on mass
- Quaternion kinematics
- Gyroscopic coupling

### Control Jacobian B (14×3)

```
      ⎡ -α·T/‖T‖  ⎤  (mass rate)
      ⎢    0      ⎥  (position)
B =   ⎢  C_IB/m   ⎥  (velocity)
      ⎢    0      ⎥  (quaternion)
      ⎣ J⁻¹·[r_T×]⎦  (angular velocity)
```

## References

1. Szmuk, M., & Açıkmeşe, B. (2018). Successive convexification for 6-DoF powered descent guidance with compound state-triggered constraints.

2. Szmuk, M., Açıkmeşe, B., & Berning, A. W. (2016). Successive convexification for fuel-optimal powered landing with aerodynamic drag and non-convex constraints.