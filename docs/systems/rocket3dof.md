# Rocket 3-DoF System Documentation

Point-mass rocket model for powered descent guidance. Captures translational dynamics with mass depletion.

## Overview

The Rocket3DoF system models a rocket as a point mass with variable mass due to fuel consumption. It is simpler than the 6-DoF model and is useful for:

- Trajectory optimization without attitude considerations
- Algorithm prototyping
- Understanding fundamental powered descent dynamics

## Mathematical Model

### State Vector (n=7)

```
x = [m, r_x, r_y, r_z, v_x, v_y, v_z]
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | m | Mass | kg |
| 1-3 | r_I | Position (inertial frame) | m |
| 4-6 | v_I | Velocity (inertial frame) | m/s |

### Control Vector (m=3)

```
u = [T_x, T_y, T_z]
```

Thrust vector in inertial frame.

### Dynamics Equations

#### Mass Depletion
```
ṁ = -α·‖T‖
```
where α = 1/(g₀·I_sp) is the mass flow rate coefficient.

#### Translational Dynamics
```
ṙ = v
v̇ = T/m + g + F_drag/m
```

#### Drag Model (Optional)
```
F_drag = -0.5 · ρ · C_D · A_ref · ‖v‖ · v
```

## Coordinate Frame

### Inertial Frame (UEN)

- **x-axis**: Up (opposite to gravity)
- **y-axis**: East
- **z-axis**: North
- **Origin**: Landing site

Gravity vector: g_I = [-g, 0, 0] where g is gravitational acceleration.

## Parameters

```python
Rocket3DoFParams(
    # Mass
    m_dry=1.0,           # Dry mass [kg]
    m_wet=2.0,           # Wet mass [kg]
    
    # Propulsion
    I_sp=30.0,           # Specific impulse [s]
    g0=1.0,              # Reference gravity [m/s²]
    T_min=0.0,           # Minimum thrust [N]
    T_max=6.5,           # Maximum thrust [N]
    
    # Environment
    g_I=[-1, 0, 0],      # Gravity vector [m/s²]
    
    # Drag (optional)
    enable_drag=False,
    rho=1.0,             # Atmospheric density [kg/m³]
    C_D=0.5,             # Drag coefficient
    A_ref=0.5,           # Reference area [m²]
    
    # Constraints
    gamma_gs=30°,        # Glide slope angle [rad]
    v_max=inf,           # Maximum velocity [m/s]
)
```

### Derived Parameters

```python
params.alpha       # Mass flow rate: 1/(g0·I_sp)
params.fuel_mass   # Available fuel: m_wet - m_dry
```

## Constraints

### Thrust Magnitude
```
T_min ≤ ‖T‖ ≤ T_max
```

### Glide Slope
```
‖r_yz‖ ≤ r_x · tan(γ_gs)
```

The glide slope constraint keeps the rocket within a cone centered at the landing site. This ensures the rocket can always "see" the landing pad.

## Jacobians

### State Jacobian A (7×7)

```
      ⎡ 0   0   0   0   0   0   0 ⎤
      ⎢ 0   0   0   0   1   0   0 ⎥
      ⎢ 0   0   0   0   0   1   0 ⎥
A =   ⎢ 0   0   0   0   0   0   1 ⎥
      ⎢ *   0   0   0   *   *   * ⎥
      ⎢ *   0   0   0   *   *   * ⎥
      ⎣ *   0   0   0   *   *   * ⎦
```

Key elements:
- ∂ṙ/∂v = I₃ (position-velocity coupling)
- ∂v̇/∂m = -T/m² (thrust acceleration)
- ∂v̇/∂v from drag (when enabled)

### Control Jacobian B (7×3)

```
      ⎡ -α·T_x/‖T‖  -α·T_y/‖T‖  -α·T_z/‖T‖ ⎤
      ⎢     0            0           0      ⎥
      ⎢     0            0           0      ⎥
B =   ⎢     0            0           0      ⎥
      ⎢    1/m           0           0      ⎥
      ⎢     0           1/m          0      ⎥
      ⎣     0            0          1/m     ⎦
```

## Usage Examples

### Basic Simulation

```python
import numpy as np
from simdyn import create_normalized_rocket3dof

# Create rocket with normalized parameters
rocket = create_normalized_rocket3dof()

# Initial state: mass=2, altitude=10, at rest
x0 = rocket.pack_state(
    mass=2.0,
    position=np.array([10.0, 0.0, 0.0]),
    velocity=np.zeros(3)
)

# Hover controller
def hover_controller(t, x):
    return rocket.hover_thrust(x)

# Simulate
t, x, u = rocket.simulate(x0, hover_controller, (0, 5), dt=0.01)

print(f"Fuel used: {1 - rocket.fuel_fraction(x[-1]):.1%}")
```

### Gravity Turn Landing

```python
import numpy as np
from simdyn import create_normalized_rocket3dof

rocket = create_normalized_rocket3dof()

# Initial: high altitude, falling
x0 = rocket.pack_state(
    mass=2.0,
    position=np.array([15.0, 3.0, 0.0]),
    velocity=np.array([-3.0, 1.0, 0.0])
)

def gravity_turn_controller(t, x):
    m = rocket.get_mass(x)
    v = rocket.get_velocity(x)
    g = rocket.params.g_I
    
    # Thrust opposite to velocity (gravity turn)
    speed = np.linalg.norm(v)
    if speed > 0.1:
        T_dir = -v / speed
    else:
        T_dir = -np.array(g) / np.linalg.norm(g)
    
    # Modulate thrust based on altitude
    alt = rocket.get_altitude(x)
    T_mag = rocket.params.T_max if alt > 5 else rocket.params.T_min + 1.0
    
    return T_dir * T_mag

t, x, u = rocket.simulate(x0, gravity_turn_controller, (0, 10), dt=0.01)
```

### With Aerodynamic Drag

```python
from simdyn import Rocket3DoF, Rocket3DoFParams

params = Rocket3DoFParams(
    m_wet=2.0,
    m_dry=1.0,
    enable_drag=True,
    rho=1.0,
    C_D=0.5,
    A_ref=0.1,
)

rocket = Rocket3DoF(params)
```

## Energy Analysis

```python
E = rocket.energy(x)
# Returns:
# {
#   'kinetic': 0.5 * m * v²,
#   'potential': m * g * h,
#   'total': KE + PE
# }
```

## References

1. Szmuk, M., & Açıkmeşe, B. (2018). Successive convexification for 6-DoF powered descent guidance.

2. Blackmore, L., Açıkmeşe, B., & Scharf, D. P. (2010). Minimum-landing-error powered-descent guidance for Mars landing using convex optimization.