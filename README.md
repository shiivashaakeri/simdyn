# 🚀 SimDyn

**A Modular Dynamics Library for Control Systems Research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-692%20passed-brightgreen.svg)]()

SimDyn is a unified Python library providing standardized implementations of dynamical systems commonly used in robotics and aerospace control research. The library emphasizes:

- **Consistency**: All systems share a common interface
- **Completeness**: Each system provides dynamics, Jacobians, and constraints
- **Flexibility**: Support for disturbances, parameter variations, and different discretization schemes
- **Research-ready**: Designed for control synthesis, trajectory optimization, and data-driven methods

## Installation

```bash
# Install from PyPI
pip install simdyn

# Or install with visualization support
pip install simdyn[viz]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/shiivashaakeri/simdyn.git
cd simdyn

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- NumPy
- SciPy

### Optional Dependencies

- matplotlib (visualization)

## Quick Start

```python
import numpy as np
import simdyn as ds

# Create a simple pendulum
pendulum = ds.create_normalized_pendulum()

# Initial state: displaced from bottom
x0 = np.array([0.5, 0.0])  # [angle, angular_velocity]

# Define a controller (or use zero for free dynamics)
controller = lambda t, x: np.array([0.0])

# Simulate for 10 seconds
t, x, u = pendulum.simulate(x0, controller, t_span=(0, 10), dt=0.01)

# Check energy conservation
E_initial = pendulum.energy(x[0])['total']
E_final = pendulum.energy(x[-1])['total']
print(f"Energy drift: {abs(E_final - E_initial):.6f}")
```

## Available Systems

| System | State Dim | Control Dim | Type | Description |
|--------|-----------|-------------|------|-------------|
| **DoubleIntegrator** | 2-6 | 1-3 | Linear | Point mass in 1D/2D/3D |
| **Unicycle** | 3 | 2 | Nonholonomic | 2D mobile robot |
| **Pendulum** | 2 | 1 | Nonlinear | Classic swing-up benchmark |
| **CartPole** | 4 | 1 | Underactuated | Inverted pendulum on cart |
| **Rocket3DoF** | 7 | 3 | Nonlinear | Point-mass powered descent |
| **Rocket6DoF** | 14 | 3 | Nonlinear | Rigid-body rocket with attitude |

## Core Features

### Unified Interface

All systems inherit from `DynamicalSystem` and provide:

```python
# Continuous-time dynamics
x_dot = system.f(x, u, w)  # w is optional disturbance

# Discrete-time dynamics
x_next = system.f_discrete(x, u, dt, w, method='rk4')

# Analytical Jacobians
A = system.A(x, u)  # ∂f/∂x
B = system.B(x, u)  # ∂f/∂u

# Linearization at operating point
A, B, c = system.linearize(x0, u0)

# Constraint checking
lb, ub = system.get_state_bounds()
lb, ub = system.get_control_bounds()

# Closed-loop simulation
t, x, u = system.simulate(x0, controller, t_span, dt)
```

### Disturbance Handling

All dynamics accept an optional disturbance input:

```python
# Process noise
w = np.random.randn(system.n_disturbance) * 0.01
x_dot = system.f(x, u, w)

# Simulate with disturbance function
def disturbance_fn(t, x):
    return np.random.randn(system.n_disturbance) * 0.01

t, x, u = system.simulate(x0, controller, t_span, dt, disturbance_fn=disturbance_fn)
```

### Jacobian Verification

Verify analytical Jacobians against numerical differentiation:

```python
passed, errors = system.verify_jacobians(x, u, eps=1e-7, tol=1e-5)
if not passed:
    print(f"Jacobian errors: {errors}")
```

### Parameterized Systems (Digital Twins)

Create system variants with different parameters:

```python
# True system
params_true = ds.Rocket3DoFParams(m_wet=2.0, I_sp=30.0)
rocket_true = ds.Rocket3DoF(params_true)

# Model with parameter mismatch
params_model = ds.Rocket3DoFParams(m_wet=1.9, I_sp=28.0)  # 5% error
rocket_model = ds.Rocket3DoF(params_model)

# Collect data from true system, design control with model
```

## Examples

### Rocket Landing

```python
import numpy as np
import simdyn as ds

# Create 3-DoF rocket with normalized parameters
rocket = ds.create_normalized_rocket3dof()

# Initial state: mass=2, altitude=10, falling
x0 = rocket.pack_state(
    mass=2.0,
    position=np.array([10.0, 2.0, 0.0]),
    velocity=np.array([-2.0, 1.0, 0.0])
)

# Simple gravity-turn controller
def landing_controller(t, x):
    pos = rocket.get_position(x)
    vel = rocket.get_velocity(x)
    mass = rocket.get_mass(x)
    g = rocket.params.g_I
    
    # Thrust to cancel gravity + velocity damping
    T = -mass * g - 2.0 * vel
    T_mag = np.clip(np.linalg.norm(T), rocket.params.T_min, rocket.params.T_max)
    
    if np.linalg.norm(T) > 1e-6:
        return T / np.linalg.norm(T) * T_mag
    return rocket.hover_thrust(x)

# Simulate
t, x, u = rocket.simulate(x0, landing_controller, (0, 8), dt=0.01, method='rk4')

print(f"Final altitude: {rocket.get_altitude(x[-1]):.3f}")
print(f"Final velocity: {rocket.get_speed(x[-1]):.3f}")
print(f"Fuel used: {1 - rocket.fuel_fraction(x[-1]):.1%}")
```

### Cart-Pole Swing-Up

```python
import numpy as np
import simdyn as ds

cartpole = ds.create_cartpole()

# Start with pole down
x0 = np.array([0.0, 0.0, np.pi, 0.0])

# Energy-based swing-up + balancing controller
E_target = cartpole.energy_upright()

def swing_up_controller(t, x):
    cart_x, cart_v, theta, omega = x
    E = cartpole.energy(x)['total']
    
    # Near upright: switch to balancing
    if abs(theta) < 0.3:
        F = 50*theta + 15*omega + cart_x + 2*cart_v
    else:
        # Energy pumping
        F = 5.0 * np.sign(omega * np.cos(theta)) * (E_target - E)
    
    return np.array([np.clip(F, -10, 10)])

t, x, u = cartpole.simulate(x0, swing_up_controller, (0, 10), dt=0.02)
```

### LQR Control Design

```python
import numpy as np
from scipy import linalg
import simdyn as ds

# Create pendulum
pendulum = ds.create_normalized_pendulum()

# Linearize at upright equilibrium
A, B = pendulum.linearize_at_top()

# LQR weights
Q = np.diag([10.0, 1.0])  # state cost
R = np.array([[0.1]])      # control cost

# Solve Riccati equation
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# LQR controller (relative to upright)
x_target = np.array([np.pi, 0.0])

def lqr_controller(t, x):
    x_error = x - x_target
    x_error[0] = np.arctan2(np.sin(x_error[0]), np.cos(x_error[0]))  # wrap angle
    return -K @ x_error

# Simulate from near upright
x0 = np.array([np.pi - 0.1, 0.0])
t, x, u = pendulum.simulate(x0, lqr_controller, (0, 5), dt=0.01)
```

## Conventions

### Coordinate Frames

| System | Inertial Frame | Body Frame |
|--------|---------------|------------|
| Rocket 3-DoF/6-DoF | UEN (x=up) | x=thrust axis |
| Quadrotor | NED or ENU | x=forward |
| Unicycle/Bicycle | 2D XY plane | x=forward |

### Quaternions

- **Convention**: Scalar-first `q = [q_w, q_x, q_y, q_z]`
- **Rotation**: `q_BI` transforms inertial vectors to body frame
- **Identity**: `[1, 0, 0, 0]`

### Units

- Default: SI units (meters, seconds, kilograms, radians)
- Normalized parameters available for algorithm testing

See [docs/conventions.md](docs/conventions.md) for full details.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simdyn

# Run specific test file
pytest tests/systems/test_rocket6dof.py -v
```

## Project Structure

```
simdyn/
├── src/simdyn/
│   ├── __init__.py          # Package exports
│   ├── base.py              # Abstract base class
│   ├── systems/             # System implementations
│   │   ├── double_integrator.py
│   │   ├── unicycle.py
│   │   ├── pendulum.py
│   │   ├── cartpole.py
│   │   ├── rocket3dof.py
│   │   └── rocket6dof.py
│   ├── integrators/         # Numerical integrators
│   │   ├── euler.py
│   │   └── rk4.py
│   └── utils/               # Utilities
│       ├── rotations.py     # SO(3), Euler angles
│       └── quaternion.py    # Quaternion operations
├── tests/                   # Comprehensive test suite
├── examples/                # Example scripts
└── docs/                    # Documentation
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Szmuk, M., & Açıkmeşe, B. (2018). Successive convexification for 6-DoF powered descent guidance with compound state-triggered constraints.
- Tedrake, R. Underactuated Robotics (MIT OCW).
- Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system.

## Acknowledgments

Developed for control systems research at University of Washington.