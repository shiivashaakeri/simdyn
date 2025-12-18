"""
Tests for the Double Integrator dynamical system.

These tests verify:
1. System properties and dimensions
2. Dynamics evaluation
3. Analytical Jacobians
4. Constraint checking
5. Analytical solutions
6. Simulation accuracy
7. Factory functions
"""

import numpy as np
import pytest

from simdyn import DynamicalSystem
from simdyn.systems.double_integrator import (
    DoubleIntegrator,
    DoubleIntegrator1D,
    DoubleIntegrator2D,
    DoubleIntegrator3D,
    DoubleIntegratorParams,
    default_params_1d,
    default_params_2d,
    default_params_3d,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def system_1d():
    """1D double integrator with default parameters."""
    return DoubleIntegrator1D()


@pytest.fixture
def system_2d():
    """2D double integrator with default parameters."""
    return DoubleIntegrator2D()


@pytest.fixture
def system_3d():
    """3D double integrator with default parameters."""
    return DoubleIntegrator3D()


@pytest.fixture
def custom_params():
    """Custom parameters for testing."""
    return DoubleIntegratorParams(
        n_dim=2,
        mass=2.0,
        p_min=np.array([-5.0, -5.0]),
        p_max=np.array([5.0, 5.0]),
        v_min=np.array([-3.0, -3.0]),
        v_max=np.array([3.0, 3.0]),
        u_min=np.array([-1.0, -1.0]),
        u_max=np.array([1.0, 1.0]),
    )


# =============================================================================
# Test: Parameter Validation
# =============================================================================


class TestParameters:
    """Tests for parameter validation and defaults."""

    def test_default_params_1d(self):
        params = default_params_1d()
        assert params.n_dim == 1
        assert params.mass == 1.0

    def test_default_params_2d(self):
        params = default_params_2d()
        assert params.n_dim == 2
        assert params.mass == 1.0

    def test_default_params_3d(self):
        params = default_params_3d()
        assert params.n_dim == 3
        assert params.mass == 1.0

    def test_invalid_n_dim(self):
        with pytest.raises(ValueError):
            DoubleIntegratorParams(n_dim=4)

    def test_invalid_mass(self):
        with pytest.raises(ValueError):
            DoubleIntegratorParams(mass=-1.0)
        with pytest.raises(ValueError):
            DoubleIntegratorParams(mass=0.0)

    def test_custom_bounds(self, custom_params):
        system = DoubleIntegrator(custom_params)
        lb, ub = system.get_state_bounds()
        assert lb[0] == -5.0
        assert ub[0] == 5.0


# =============================================================================
# Test: System Properties
# =============================================================================


class TestProperties:
    """Tests for system properties."""

    def test_inheritance(self, system_2d):
        assert isinstance(system_2d, DynamicalSystem)

    def test_n_state_1d(self, system_1d):
        assert system_1d.n_state == 2

    def test_n_state_2d(self, system_2d):
        assert system_2d.n_state == 4

    def test_n_state_3d(self, system_3d):
        assert system_3d.n_state == 6

    def test_n_control_1d(self, system_1d):
        assert system_1d.n_control == 1

    def test_n_control_2d(self, system_2d):
        assert system_2d.n_control == 2

    def test_n_control_3d(self, system_3d):
        assert system_3d.n_control == 3

    def test_n_disturbance(self, system_2d):
        assert system_2d.n_disturbance == system_2d.n_state

    def test_state_names_1d(self, system_1d):
        assert system_1d.state_names == ["p_x", "v_x"]

    def test_state_names_2d(self, system_2d):
        assert system_2d.state_names == ["p_x", "p_y", "v_x", "v_y"]

    def test_state_names_3d(self, system_3d):
        assert system_3d.state_names == ["p_x", "p_y", "p_z", "v_x", "v_y", "v_z"]

    def test_control_names_2d(self, system_2d):
        assert system_2d.control_names == ["u_x", "u_y"]

    def test_n_dim(self, system_2d):
        assert system_2d.n_dim == 2

    def test_repr(self, system_2d):
        repr_str = repr(system_2d)
        assert "DoubleIntegrator" in repr_str
        assert "n_state=4" in repr_str


# =============================================================================
# Test: State Accessors
# =============================================================================


class TestStateAccessors:
    """Tests for state pack/unpack utilities."""

    def test_get_position_2d(self, system_2d):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        p = system_2d.get_position(x)
        np.testing.assert_array_equal(p, [1.0, 2.0])

    def test_get_velocity_2d(self, system_2d):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        v = system_2d.get_velocity(x)
        np.testing.assert_array_equal(v, [3.0, 4.0])

    def test_pack_state_2d(self, system_2d):
        p = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        x = system_2d.pack_state(p, v)
        np.testing.assert_array_equal(x, [1.0, 2.0, 3.0, 4.0])

    def test_pack_unpack_roundtrip(self, system_3d):
        p_orig = np.array([1.0, 2.0, 3.0])
        v_orig = np.array([4.0, 5.0, 6.0])
        x = system_3d.pack_state(p_orig, v_orig)
        p_back = system_3d.get_position(x)
        v_back = system_3d.get_velocity(x)
        np.testing.assert_array_equal(p_back, p_orig)
        np.testing.assert_array_equal(v_back, v_orig)


# =============================================================================
# Test: Continuous Dynamics
# =============================================================================


class TestDynamics:
    """Tests for continuous-time dynamics f(x, u, w)."""

    def test_dynamics_shape_1d(self, system_1d):
        x = np.array([0.0, 1.0])
        u = np.array([0.5])
        x_dot = system_1d.f(x, u)
        assert x_dot.shape == (2,)

    def test_dynamics_shape_2d(self, system_2d):
        x = np.array([0.0, 0.0, 1.0, 2.0])
        u = np.array([0.5, 0.5])
        x_dot = system_2d.f(x, u)
        assert x_dot.shape == (4,)

    def test_dynamics_velocity_integration(self, system_2d):
        """ṗ = v should hold."""
        x = np.array([0.0, 0.0, 3.0, 4.0])  # velocity is [3, 4]
        u = np.array([0.0, 0.0])
        x_dot = system_2d.f(x, u)
        # Position derivative should equal velocity
        np.testing.assert_array_equal(x_dot[:2], [3.0, 4.0])

    def test_dynamics_acceleration(self, system_2d):
        """v̇ = u/m should hold."""
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u = np.array([2.0, 4.0])
        x_dot = system_2d.f(x, u)
        # Velocity derivative should equal u/m = u (since m=1)
        np.testing.assert_array_equal(x_dot[2:], [2.0, 4.0])

    def test_dynamics_with_mass(self, custom_params):
        """Test dynamics with non-unit mass."""
        system = DoubleIntegrator(custom_params)  # mass = 2.0
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u = np.array([2.0, 4.0])
        x_dot = system.f(x, u)
        # v̇ = u/m = [2, 4]/2 = [1, 2]
        np.testing.assert_array_equal(x_dot[2:], [1.0, 2.0])

    def test_dynamics_with_disturbance(self, system_2d):
        x = np.array([0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0])
        w = np.array([0.1, 0.2, 0.3, 0.4])
        x_dot = system_2d.f(x, u, w)
        # x_dot = Ax + Bu + w = 0 + 0 + w
        np.testing.assert_array_equal(x_dot, w)

    def test_dynamics_zero_disturbance_default(self, system_2d):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        u = np.array([0.5, 0.5])
        x_dot_no_w = system_2d.f(x, u)
        x_dot_zero_w = system_2d.f(x, u, np.zeros(4))
        np.testing.assert_array_equal(x_dot_no_w, x_dot_zero_w)

    def test_dynamics_linear_form(self, system_2d):
        """Verify ẋ = Ax + Bu."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        u = np.array([0.5, 1.0])

        A = system_2d.A(x, u)
        B = system_2d.B(x, u)

        x_dot_direct = system_2d.f(x, u)
        x_dot_matrix = A @ x + B @ u

        np.testing.assert_array_almost_equal(x_dot_direct, x_dot_matrix)


# =============================================================================
# Test: Jacobians
# =============================================================================


class TestJacobians:
    """Tests for Jacobian matrices."""

    def test_A_shape_2d(self, system_2d):
        x = np.zeros(4)
        u = np.zeros(2)
        A = system_2d.A(x, u)
        assert A.shape == (4, 4)

    def test_B_shape_2d(self, system_2d):
        x = np.zeros(4)
        u = np.zeros(2)
        B = system_2d.B(x, u)
        assert B.shape == (4, 2)

    def test_A_structure(self, system_2d):
        """A matrix should be [0 I; 0 0]."""
        x = np.zeros(4)
        u = np.zeros(2)
        A = system_2d.A(x, u)

        expected = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(A, expected)

    def test_B_structure(self, system_2d):
        """B matrix should be [0; I/m]."""
        x = np.zeros(4)
        u = np.zeros(2)
        B = system_2d.B(x, u)

        expected = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )
        np.testing.assert_array_equal(B, expected)

    def test_B_with_mass(self, custom_params):
        """B matrix should scale with 1/m."""
        system = DoubleIntegrator(custom_params)  # mass = 2.0
        B = system.B(np.zeros(4), np.zeros(2))

        expected = np.array(
            [
                [0, 0],
                [0, 0],
                [0.5, 0],
                [0, 0.5],
            ]
        )
        np.testing.assert_array_equal(B, expected)

    def test_A_constant(self, system_2d):
        """A matrix should be constant (independent of state/control)."""
        A1 = system_2d.A(np.zeros(4), np.zeros(2))
        A2 = system_2d.A(np.ones(4), np.ones(2))
        A3 = system_2d.A(np.random.randn(4), np.random.randn(2))

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(A2, A3)

    def test_B_constant(self, system_2d):
        """B matrix should be constant (independent of state/control)."""
        B1 = system_2d.B(np.zeros(4), np.zeros(2))
        B2 = system_2d.B(np.ones(4), np.ones(2))

        np.testing.assert_array_equal(B1, B2)

    def test_jacobian_numerical_verification(self, system_2d):
        """Verify analytical Jacobians against numerical computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        u = np.array([0.5, 1.0])

        passed, errors = system_2d.verify_jacobians(x, u)

        assert passed
        assert errors["A_relative_error"] < 1e-5
        assert errors["B_relative_error"] < 1e-5

    def test_jacobian_verification_all_dims(self, system_1d, system_2d, system_3d):
        """Verify Jacobians for all dimensions."""
        for system in [system_1d, system_2d, system_3d]:
            x = np.random.randn(system.n_state)
            u = np.random.randn(system.n_control)
            passed, _ = system.verify_jacobians(x, u)
            assert passed


# =============================================================================
# Test: Linearization
# =============================================================================


class TestLinearization:
    """Tests for the linearize() method."""

    def test_linearize_returns_correct_shapes(self, system_2d):
        x0 = np.zeros(4)
        u0 = np.zeros(2)

        A, B, G, c = system_2d.linearize(x0, u0)

        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        assert G.shape == (4, 4)
        assert c.shape == (4,)

    def test_linearize_at_origin(self, system_2d):
        """At origin with zero control, c should be zero."""
        x0 = np.zeros(4)
        u0 = np.zeros(2)

        A, B, G, c = system_2d.linearize(x0, u0)

        np.testing.assert_array_almost_equal(c, np.zeros(4))

    def test_linearize_c_equals_f(self, system_2d):
        """Affine term c should equal f(x0, u0)."""
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        u0 = np.array([0.5, 1.0])

        A, B, G, c = system_2d.linearize(x0, u0)
        f_at_x0 = system_2d.f(x0, u0)

        np.testing.assert_array_almost_equal(c, f_at_x0)


# =============================================================================
# Test: Constraints
# =============================================================================


class TestConstraints:
    """Tests for constraint checking."""

    def test_state_bounds_shape(self, system_2d):
        lb, ub = system_2d.get_state_bounds()
        assert lb.shape == (4,)
        assert ub.shape == (4,)

    def test_control_bounds_shape(self, system_2d):
        lb, ub = system_2d.get_control_bounds()
        assert lb.shape == (2,)
        assert ub.shape == (2,)

    def test_bounds_consistency(self, system_2d):
        """Lower bounds should be less than upper bounds."""
        lb_x, ub_x = system_2d.get_state_bounds()
        lb_u, ub_u = system_2d.get_control_bounds()

        assert np.all(lb_x < ub_x)
        assert np.all(lb_u < ub_u)

    def test_is_state_valid_in_bounds(self, system_2d):
        x = np.array([0.0, 0.0, 0.0, 0.0])
        assert system_2d.is_state_valid(x)

    def test_is_state_valid_out_of_bounds(self, system_2d):
        x = np.array([100.0, 0.0, 0.0, 0.0])  # way out of bounds
        assert not system_2d.is_state_valid(x)

    def test_is_control_valid_in_bounds(self, system_2d):
        u = np.array([0.0, 0.0])
        assert system_2d.is_control_valid(u)

    def test_is_control_valid_out_of_bounds(self, system_2d):
        u = np.array([100.0, 0.0])  # way out of bounds
        assert not system_2d.is_control_valid(u)

    def test_unbounded_parameters(self):
        """Test system with no bounds specified."""
        params = DoubleIntegratorParams(n_dim=2)
        system = DoubleIntegrator(params)

        lb_x, ub_x = system.get_state_bounds()
        lb_u, ub_u = system.get_control_bounds()

        assert np.all(lb_x == -np.inf)
        assert np.all(ub_x == np.inf)
        assert np.all(lb_u == -np.inf)
        assert np.all(ub_u == np.inf)


# =============================================================================
# Test: Analytical Solution
# =============================================================================


class TestAnalyticalSolution:
    """Tests for analytical solution."""

    def test_analytical_at_t_zero(self, system_2d):
        """At t=0, solution should equal initial state."""
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        u = np.array([0.5, 1.0])

        x_t = system_2d.analytical_solution(x0, u, t=0.0)

        np.testing.assert_array_almost_equal(x_t, x0)

    def test_analytical_zero_control(self, system_2d):
        """With zero control, motion should be linear in time."""
        x0 = np.array([0.0, 0.0, 1.0, 2.0])  # vel = [1, 2]
        u = np.array([0.0, 0.0])
        t = 1.0

        x_t = system_2d.analytical_solution(x0, u, t)

        # p(t) = p0 + v0*t = [0,0] + [1,2]*1 = [1, 2]
        np.testing.assert_array_almost_equal(x_t[:2], [1.0, 2.0])
        # v(t) = v0 = [1, 2]
        np.testing.assert_array_almost_equal(x_t[2:], [1.0, 2.0])

    def test_analytical_constant_acceleration(self, system_2d):
        """With constant control, position should be quadratic."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])  # at rest at origin
        u = np.array([2.0, 0.0])  # accelerate in x
        t = 1.0

        x_t = system_2d.analytical_solution(x0, u, t)

        # p(t) = 0.5*a*t² = 0.5*[2,0]*1 = [1, 0]
        np.testing.assert_array_almost_equal(x_t[:2], [1.0, 0.0])
        # v(t) = a*t = [2, 0]*1 = [2, 0]
        np.testing.assert_array_almost_equal(x_t[2:], [2.0, 0.0])

    def test_analytical_matches_simulation(self, system_2d):
        """Analytical solution should match numerical simulation."""
        x0 = np.array([1.0, 2.0, 0.5, 1.0])
        u_const = np.array([0.3, -0.2])
        t_final = 2.0
        dt = 0.001

        # Numerical simulation with constant control
        controller = lambda t, x: u_const
        t, x_traj, _ = system_2d.simulate(x0, controller, (0, t_final), dt, method="rk4")

        # Analytical solution
        x_analytical = system_2d.analytical_solution(x0, u_const, t_final)

        np.testing.assert_array_almost_equal(x_traj[-1], x_analytical, decimal=4)


# =============================================================================
# Test: Equilibrium and Energy
# =============================================================================


class TestEquilibriumAndEnergy:
    """Tests for equilibrium and energy computation."""

    def test_equilibrium_zero_control(self, system_2d):
        """Equilibrium with zero control should be at rest."""
        x_eq = system_2d.equilibrium()

        # Position can be anything (we use origin), velocity must be zero
        v = system_2d.get_velocity(x_eq)
        np.testing.assert_array_almost_equal(v, np.zeros(2))

    def test_equilibrium_nonzero_control_raises(self, system_2d):
        """Non-zero control should raise error (no equilibrium)."""
        with pytest.raises(ValueError):
            system_2d.equilibrium(u=np.array([1.0, 0.0]))

    def test_energy_at_rest(self, system_2d):
        """Energy at rest should be zero."""
        x = np.array([1.0, 2.0, 0.0, 0.0])  # at rest
        E = system_2d.energy(x)
        assert E == 0.0

    def test_energy_moving(self, system_2d):
        """Kinetic energy should be 0.5*m*v²."""
        x = np.array([0.0, 0.0, 3.0, 4.0])  # velocity [3, 4], speed = 5
        E = system_2d.energy(x)
        # E = 0.5 * 1 * (3² + 4²) = 0.5 * 25 = 12.5
        assert E == 12.5

    def test_energy_with_mass(self, custom_params):
        """Energy should scale with mass."""
        system = DoubleIntegrator(custom_params)  # mass = 2.0
        x = np.array([0.0, 0.0, 3.0, 4.0])  # velocity [3, 4]
        E = system.energy(x)
        # E = 0.5 * 2 * 25 = 25
        assert E == 25.0


# =============================================================================
# Test: Discrete Matrices
# =============================================================================


class TestDiscreteMatrices:
    """Tests for discrete-time system matrices."""

    def test_discrete_matrices_shape(self, system_2d):
        A_d, B_d = system_2d.discrete_matrices(dt=0.1)

        assert A_d.shape == (4, 4)
        assert B_d.shape == (4, 2)

    def test_discrete_A_structure(self, system_2d):
        """A_d should be [I, dt*I; 0, I]."""
        dt = 0.1
        A_d, _ = system_2d.discrete_matrices(dt)

        expected = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(A_d, expected)

    def test_discrete_B_structure(self, system_2d):
        """B_d should be [0.5*dt²*I/m; dt*I/m]."""
        dt = 0.1
        _, B_d = system_2d.discrete_matrices(dt)

        expected = np.array(
            [
                [0.5 * dt**2, 0],
                [0, 0.5 * dt**2],
                [dt, 0],
                [0, dt],
            ]
        )
        np.testing.assert_array_almost_equal(B_d, expected)

    def test_discrete_matches_exact_propagation(self, system_2d):
        """Discrete matrices should give same result as analytical solution."""
        x0 = np.array([1.0, 2.0, 0.5, 1.0])
        u = np.array([0.3, -0.2])
        dt = 0.1

        # Using discrete matrices
        A_d, B_d = system_2d.discrete_matrices(dt)
        x_discrete = A_d @ x0 + B_d @ u

        # Using analytical solution
        x_analytical = system_2d.analytical_solution(x0, u, dt)

        np.testing.assert_array_almost_equal(x_discrete, x_analytical)


# =============================================================================
# Test: Simulation
# =============================================================================


class TestSimulation:
    """Tests for simulation functionality."""

    def test_simulate_shapes(self, system_2d):
        x0 = np.zeros(4)
        controller = lambda t, x: np.zeros(2)
        t_span = (0.0, 1.0)
        dt = 0.1

        t, x, u = system_2d.simulate(x0, controller, t_span, dt)

        assert t.shape[0] == 11  # 0.0, 0.1, ..., 1.0
        assert x.shape == (11, 4)
        assert u.shape == (10, 2)

    def test_simulate_initial_condition(self, system_2d):
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        controller = lambda t, x: np.zeros(2)

        t, x, u = system_2d.simulate(x0, controller, (0, 1), dt=0.1)

        np.testing.assert_array_equal(x[0], x0)

    def test_simulate_constant_velocity(self, system_2d):
        """Zero control with initial velocity should move linearly."""
        x0 = np.array([0.0, 0.0, 1.0, 0.0])  # vel = [1, 0]
        controller = lambda t, x: np.zeros(2)

        t, x, u = system_2d.simulate(x0, controller, (0, 1), dt=0.01, method="rk4")

        # After 1 second, position should be [1, 0]
        np.testing.assert_array_almost_equal(x[-1, :2], [1.0, 0.0], decimal=3)

    def test_simulate_pd_controller(self, system_2d):
        """PD controller should stabilize the system."""
        x0 = np.array([1.0, 1.0, 0.0, 0.0])  # start displaced

        # PD control: u = -kp*p - kd*v
        kp, kd = 2.0, 3.0

        def controller(t, x):  # noqa: ARG001
            p = x[:2]
            v = x[2:]
            return -kp * p - kd * v

        t, x, u = system_2d.simulate(x0, controller, (0, 10), dt=0.01, method="rk4")

        # Should converge to origin
        np.testing.assert_array_almost_equal(x[-1], np.zeros(4), decimal=2)

    def test_simulate_with_disturbance(self, system_2d):
        """Simulation with disturbance function."""
        x0 = np.zeros(4)
        controller = lambda t, x: np.zeros(2)
        disturbance = lambda t, x: np.array([0.0, 0.0, 1.0, 0.0])  # constant accel

        t, x, u = system_2d.simulate(x0, controller, (0, 1), dt=0.01, disturbance_fn=disturbance, method="rk4")

        # x-velocity should increase, y should remain zero
        assert x[-1, 2] > 0.5  # vx increased
        np.testing.assert_almost_equal(x[-1, 3], 0.0, decimal=3)  # vy stayed 0


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_DoubleIntegrator1D(self):
        system = DoubleIntegrator1D()
        assert system.n_state == 2
        assert system.n_control == 1
        assert system.n_dim == 1

    def test_DoubleIntegrator2D(self):
        system = DoubleIntegrator2D()
        assert system.n_state == 4
        assert system.n_control == 2
        assert system.n_dim == 2

    def test_DoubleIntegrator3D(self):
        system = DoubleIntegrator3D()
        assert system.n_state == 6
        assert system.n_control == 3
        assert system.n_dim == 3

    def test_factory_with_custom_mass(self):
        system = DoubleIntegrator2D(mass=5.0)
        assert system.params.mass == 5.0

    def test_factory_with_custom_bounds(self):
        system = DoubleIntegrator2D(
            p_bounds=(-20.0, 20.0),
            v_bounds=(-10.0, 10.0),
            u_bounds=(-5.0, 5.0),
        )

        lb_x, ub_x = system.get_state_bounds()
        lb_u, ub_u = system.get_control_bounds()

        np.testing.assert_array_equal(lb_x[:2], [-20.0, -20.0])
        np.testing.assert_array_equal(ub_x[:2], [20.0, 20.0])
        np.testing.assert_array_equal(lb_u, [-5.0, -5.0])
        np.testing.assert_array_equal(ub_u, [5.0, 5.0])


# =============================================================================
# Test: Integration Accuracy
# =============================================================================


class TestIntegrationAccuracy:
    """Tests comparing numerical vs analytical solutions."""

    def test_rk4_matches_analytical(self, system_2d):
        """RK4 should closely match analytical solution."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        u_const = np.array([1.0, 0.5])
        t_final = 2.0
        dt = 0.01

        controller = lambda t, x: u_const
        t, x_traj, _ = system_2d.simulate(x0, controller, (0, t_final), dt, method="rk4")

        x_analytical = system_2d.analytical_solution(x0, u_const, t_final)

        np.testing.assert_array_almost_equal(x_traj[-1], x_analytical, decimal=5)

    def test_euler_less_accurate_than_rk4(self, system_2d):
        """Euler should be less accurate than RK4."""
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        u_const = np.array([1.0, 0.5])
        t_final = 2.0
        dt = 0.05  # Larger step to see difference

        controller = lambda t, x: u_const
        _, x_euler, _ = system_2d.simulate(x0, controller, (0, t_final), dt, method="euler")
        _, x_rk4, _ = system_2d.simulate(x0, controller, (0, t_final), dt, method="rk4")

        x_analytical = system_2d.analytical_solution(x0, u_const, t_final)

        error_euler = np.linalg.norm(x_euler[-1] - x_analytical)
        error_rk4 = np.linalg.norm(x_rk4[-1] - x_analytical)

        # For linear system, both should be exact, but RK4 is generally better
        # Actually for double integrator they should both be exact
        assert error_rk4 <= error_euler + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
