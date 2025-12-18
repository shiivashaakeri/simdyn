"""
Tests for the Unicycle dynamical system.

These tests verify:
1. System properties and dimensions
2. Dynamics evaluation
3. Analytical Jacobians (nonlinear - state dependent)
4. Constraint checking
5. Utility methods (heading, distance)
6. Simulation accuracy
7. Trajectory generation
"""

import numpy as np
import pytest

from simdyn import DynamicalSystem
from simdyn.systems.unicycle import (
    Unicycle,
    UnicycleParams,
    create_unicycle,
    default_params,
    forward_only_unicycle,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def system():
    """Default unicycle system."""
    return Unicycle()


@pytest.fixture
def custom_params():
    """Custom parameters for testing."""
    return UnicycleParams(
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        v_min=-1.0,
        v_max=1.0,
        omega_min=-2.0,
        omega_max=2.0,
    )


# =============================================================================
# Test: Parameter Validation
# =============================================================================


class TestParameters:
    """Tests for parameter validation and defaults."""

    def test_default_params(self):
        params = default_params()
        assert params.x_min == -10.0
        assert params.x_max == 10.0
        assert params.v_max == 2.0

    def test_invalid_x_bounds(self):
        with pytest.raises(ValueError):
            UnicycleParams(x_min=10.0, x_max=-10.0)

    def test_invalid_y_bounds(self):
        with pytest.raises(ValueError):
            UnicycleParams(y_min=10.0, y_max=-10.0)

    def test_invalid_v_bounds(self):
        with pytest.raises(ValueError):
            UnicycleParams(v_min=5.0, v_max=-5.0)

    def test_invalid_omega_bounds(self):
        with pytest.raises(ValueError):
            UnicycleParams(omega_min=1.0, omega_max=-1.0)

    def test_custom_params(self, custom_params):
        system = Unicycle(custom_params)
        assert system.params.v_max == 1.0


# =============================================================================
# Test: System Properties
# =============================================================================


class TestProperties:
    """Tests for system properties."""

    def test_inheritance(self, system):
        assert isinstance(system, DynamicalSystem)

    def test_n_state(self, system):
        assert system.n_state == 3

    def test_n_control(self, system):
        assert system.n_control == 2

    def test_n_disturbance(self, system):
        assert system.n_disturbance == 3

    def test_state_names(self, system):
        assert system.state_names == ["x", "y", "theta"]

    def test_control_names(self, system):
        assert system.control_names == ["v", "omega"]

    def test_repr(self, system):
        repr_str = repr(system)
        assert "Unicycle" in repr_str
        assert "n_state=3" in repr_str


# =============================================================================
# Test: State Accessors
# =============================================================================


class TestStateAccessors:
    """Tests for state pack/unpack utilities."""

    def test_get_position(self, system):
        x = np.array([1.0, 2.0, 0.5])
        pos = system.get_position(x)
        np.testing.assert_array_equal(pos, [1.0, 2.0])

    def test_get_heading(self, system):
        x = np.array([1.0, 2.0, 0.5])
        theta = system.get_heading(x)
        assert theta == 0.5

    def test_pack_state(self, system):
        pos = np.array([1.0, 2.0])
        theta = 0.5
        x = system.pack_state(pos, theta)
        np.testing.assert_array_equal(x, [1.0, 2.0, 0.5])

    def test_pack_unpack_roundtrip(self, system):
        pos_orig = np.array([3.0, 4.0])
        theta_orig = np.pi / 4
        x = system.pack_state(pos_orig, theta_orig)
        pos_back = system.get_position(x)
        theta_back = system.get_heading(x)
        np.testing.assert_array_equal(pos_back, pos_orig)
        assert theta_back == theta_orig

    def test_get_direction_vector(self, system):
        # Facing +x direction
        x = np.array([0.0, 0.0, 0.0])
        d = system.get_direction_vector(x)
        np.testing.assert_array_almost_equal(d, [1.0, 0.0])

        # Facing +y direction
        x = np.array([0.0, 0.0, np.pi / 2])
        d = system.get_direction_vector(x)
        np.testing.assert_array_almost_equal(d, [0.0, 1.0])

        # Facing -x direction
        x = np.array([0.0, 0.0, np.pi])
        d = system.get_direction_vector(x)
        np.testing.assert_array_almost_equal(d, [-1.0, 0.0])


# =============================================================================
# Test: Continuous Dynamics
# =============================================================================


class TestDynamics:
    """Tests for continuous-time dynamics f(x, u, w)."""

    def test_dynamics_shape(self, system):
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.5])
        x_dot = system.f(x, u)
        assert x_dot.shape == (3,)

    def test_dynamics_facing_x(self, system):
        """Moving forward while facing +x should increase x."""
        x = np.array([0.0, 0.0, 0.0])  # facing +x
        u = np.array([1.0, 0.0])  # v=1, ω=0
        x_dot = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, [1.0, 0.0, 0.0])

    def test_dynamics_facing_y(self, system):
        """Moving forward while facing +y should increase y."""
        x = np.array([0.0, 0.0, np.pi / 2])  # facing +y
        u = np.array([1.0, 0.0])  # v=1, ω=0
        x_dot = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, [0.0, 1.0, 0.0])

    def test_dynamics_facing_45_deg(self, system):
        """Moving at 45° should split velocity between x and y."""
        x = np.array([0.0, 0.0, np.pi / 4])  # facing 45°
        u = np.array([np.sqrt(2), 0.0])  # v=sqrt(2)
        x_dot = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, [1.0, 1.0, 0.0])

    def test_dynamics_rotation_only(self, system):
        """Pure rotation should only change theta."""
        x = np.array([1.0, 2.0, 0.0])
        u = np.array([0.0, 1.0])  # v=0, ω=1
        x_dot = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, [0.0, 0.0, 1.0])

    def test_dynamics_backward(self, system):
        """Negative velocity should move backward."""
        x = np.array([0.0, 0.0, 0.0])  # facing +x
        u = np.array([-1.0, 0.0])  # v=-1
        x_dot = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, [-1.0, 0.0, 0.0])

    def test_dynamics_with_disturbance(self, system):
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0])
        w = np.array([0.1, 0.2, 0.3])
        x_dot = system.f(x, u, w)

        np.testing.assert_array_almost_equal(x_dot, w)

    def test_dynamics_zero_disturbance_default(self, system):
        x = np.array([1.0, 2.0, 0.5])
        u = np.array([1.0, 0.5])
        x_dot_no_w = system.f(x, u)
        x_dot_zero_w = system.f(x, u, np.zeros(3))
        np.testing.assert_array_equal(x_dot_no_w, x_dot_zero_w)


# =============================================================================
# Test: Jacobians
# =============================================================================


class TestJacobians:
    """Tests for Jacobian matrices (state-dependent for unicycle)."""

    def test_A_shape(self, system):
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])
        A = system.A(x, u)
        assert A.shape == (3, 3)

    def test_B_shape(self, system):
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])
        B = system.B(x, u)
        assert B.shape == (3, 2)

    def test_A_structure_theta_zero(self, system):
        """A matrix at θ=0."""
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([2.0, 0.0])  # v=2
        A = system.A(x, u)

        # A = [0, 0, -v*sin(0)] = [0, 0, 0]
        #     [0, 0,  v*cos(0)] = [0, 0, 2]
        #     [0, 0,  0       ]
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(A, expected)

    def test_A_structure_theta_90(self, system):
        """A matrix at θ=π/2."""
        x = np.array([0.0, 0.0, np.pi / 2])
        u = np.array([2.0, 0.0])  # v=2
        A = system.A(x, u)

        # A = [0, 0, -v*sin(π/2)] = [0, 0, -2]
        #     [0, 0,  v*cos(π/2)] = [0, 0, 0]
        #     [0, 0,  0         ]
        expected = np.array(
            [
                [0.0, 0.0, -2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(A, expected)

    def test_B_structure_theta_zero(self, system):
        """B matrix at θ=0."""
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.5])
        B = system.B(x, u)

        # B = [cos(0), 0] = [1, 0]
        #     [sin(0), 0] = [0, 0]
        #     [0,      1] = [0, 1]
        expected = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(B, expected)

    def test_B_structure_theta_90(self, system):
        """B matrix at θ=π/2."""
        x = np.array([0.0, 0.0, np.pi / 2])
        u = np.array([1.0, 0.5])
        B = system.B(x, u)

        # B = [cos(π/2), 0] = [0, 0]
        #     [sin(π/2), 0] = [1, 0]
        #     [0,        1] = [0, 1]
        expected = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(B, expected)

    def test_A_depends_on_state(self, system):
        """A matrix should change with state (nonlinear system)."""
        u = np.array([1.0, 0.0])

        A1 = system.A(np.array([0.0, 0.0, 0.0]), u)
        A2 = system.A(np.array([0.0, 0.0, np.pi / 4]), u)

        assert not np.allclose(A1, A2)

    def test_A_depends_on_control(self, system):
        """A matrix should change with control (velocity affects it)."""
        x = np.array([0.0, 0.0, np.pi / 4])

        A1 = system.A(x, np.array([1.0, 0.0]))
        A2 = system.A(x, np.array([2.0, 0.0]))

        assert not np.allclose(A1, A2)

    def test_B_depends_on_state(self, system):
        """B matrix should change with state (heading affects it)."""
        u = np.array([1.0, 0.0])

        B1 = system.B(np.array([0.0, 0.0, 0.0]), u)
        B2 = system.B(np.array([0.0, 0.0, np.pi / 2]), u)

        assert not np.allclose(B1, B2)

    def test_jacobian_numerical_verification(self, system):
        """Verify analytical Jacobians against numerical computation."""
        # Test at multiple states
        test_states = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, np.pi / 4]),
            np.array([-1.0, 3.0, np.pi]),
            np.array([0.5, -0.5, -np.pi / 3]),
        ]
        test_controls = [
            np.array([1.0, 0.5]),
            np.array([0.5, -0.3]),
            np.array([-0.5, 1.0]),
        ]

        for x in test_states:
            for u in test_controls:
                passed, errors = system.verify_jacobians(x, u)
                assert passed, f"Jacobian verification failed at x={x}, u={u}, errors={errors}"


# =============================================================================
# Test: Linearization
# =============================================================================


class TestLinearization:
    """Tests for the linearize() method."""

    def test_linearize_returns_correct_shapes(self, system):
        x0 = np.array([0.0, 0.0, 0.0])
        u0 = np.array([1.0, 0.0])

        A, B, G, c = system.linearize(x0, u0)

        assert A.shape == (3, 3)
        assert B.shape == (3, 2)
        assert G.shape == (3, 3)
        assert c.shape == (3,)

    def test_linearize_c_equals_f(self, system):
        """Affine term c should equal f(x0, u0)."""
        x0 = np.array([1.0, 2.0, np.pi / 4])
        u0 = np.array([1.0, 0.5])

        A, B, G, c = system.linearize(x0, u0)
        f_at_x0 = system.f(x0, u0)

        np.testing.assert_array_almost_equal(c, f_at_x0)


# =============================================================================
# Test: Constraints
# =============================================================================


class TestConstraints:
    """Tests for constraint checking."""

    def test_state_bounds_shape(self, system):
        lb, ub = system.get_state_bounds()
        assert lb.shape == (3,)
        assert ub.shape == (3,)

    def test_control_bounds_shape(self, system):
        lb, ub = system.get_control_bounds()
        assert lb.shape == (2,)
        assert ub.shape == (2,)

    def test_theta_unbounded(self, system):
        """Theta should be unbounded (can wrap around)."""
        lb, ub = system.get_state_bounds()
        assert lb[2] == -np.inf
        assert ub[2] == np.inf

    def test_is_state_valid_in_bounds(self, system):
        x = np.array([0.0, 0.0, 0.0])
        assert system.is_state_valid(x)

    def test_is_state_valid_out_of_bounds(self, system):
        x = np.array([100.0, 0.0, 0.0])  # way out of bounds
        assert not system.is_state_valid(x)

    def test_is_state_valid_any_theta(self, system):
        """Any theta value should be valid."""
        x1 = np.array([0.0, 0.0, 100 * np.pi])
        x2 = np.array([0.0, 0.0, -100 * np.pi])
        assert system.is_state_valid(x1)
        assert system.is_state_valid(x2)

    def test_is_control_valid_in_bounds(self, system):
        u = np.array([0.0, 0.0])
        assert system.is_control_valid(u)

    def test_is_control_valid_out_of_bounds(self, system):
        u = np.array([100.0, 0.0])  # way out of bounds
        assert not system.is_control_valid(u)


# =============================================================================
# Test: Utility Methods
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_normalize_state(self, system):
        """Normalize should wrap theta to [-π, π]."""
        x = np.array([1.0, 2.0, 3 * np.pi])
        x_norm = system.normalize_state(x)

        assert x_norm[0] == 1.0
        assert x_norm[1] == 2.0
        assert -np.pi <= x_norm[2] <= np.pi

    def test_distance_to_point(self, system):
        x = np.array([0.0, 0.0, 0.0])
        target = np.array([3.0, 4.0])

        dist = system.distance_to_point(x, target)

        assert dist == 5.0  # 3-4-5 triangle

    def test_heading_to_point_right(self, system):
        """Target to the right should give heading 0."""
        x = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 0.0])

        heading = system.heading_to_point(x, target)

        np.testing.assert_almost_equal(heading, 0.0)

    def test_heading_to_point_up(self, system):
        """Target above should give heading π/2."""
        x = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0])

        heading = system.heading_to_point(x, target)

        np.testing.assert_almost_equal(heading, np.pi / 2)

    def test_heading_to_point_left(self, system):
        """Target to the left should give heading π."""
        x = np.array([0.0, 0.0, 0.0])
        target = np.array([-1.0, 0.0])

        heading = system.heading_to_point(x, target)

        np.testing.assert_almost_equal(np.abs(heading), np.pi)

    def test_heading_error_zero(self, system):
        """Heading error should be zero when facing target."""
        x = np.array([0.0, 0.0, 0.0])  # facing +x
        target = np.array([1.0, 0.0])  # target in +x

        error = system.heading_error(x, target)

        np.testing.assert_almost_equal(error, 0.0)

    def test_heading_error_90_deg(self, system):
        """Heading error should be π/2 when target is 90° off."""
        x = np.array([0.0, 0.0, 0.0])  # facing +x
        target = np.array([0.0, 1.0])  # target in +y

        error = system.heading_error(x, target)

        np.testing.assert_almost_equal(error, np.pi / 2)

    def test_heading_error_negative(self, system):
        """Heading error should be negative when target is clockwise."""
        x = np.array([0.0, 0.0, 0.0])  # facing +x
        target = np.array([0.0, -1.0])  # target in -y

        error = system.heading_error(x, target)

        np.testing.assert_almost_equal(error, -np.pi / 2)


# =============================================================================
# Test: Equilibrium
# =============================================================================


class TestEquilibrium:
    """Tests for equilibrium computation."""

    def test_equilibrium_zero_control(self, system):
        x_eq = system.equilibrium()
        np.testing.assert_array_equal(x_eq, np.zeros(3))

    def test_equilibrium_nonzero_v_raises(self, system):
        with pytest.raises(ValueError):
            system.equilibrium(u=np.array([1.0, 0.0]))

    def test_equilibrium_nonzero_omega_raises(self, system):
        with pytest.raises(ValueError):
            system.equilibrium(u=np.array([0.0, 1.0]))


# =============================================================================
# Test: Simulation
# =============================================================================


class TestSimulation:
    """Tests for simulation functionality."""

    def test_simulate_shapes(self, system):
        x0 = np.array([0.0, 0.0, 0.0])
        controller = lambda t, x: np.array([1.0, 0.0])
        t_span = (0.0, 1.0)
        dt = 0.1

        t, x, u = system.simulate(x0, controller, t_span, dt)

        assert t.shape[0] == 11
        assert x.shape == (11, 3)
        assert u.shape == (10, 2)

    def test_simulate_initial_condition(self, system):
        x0 = np.array([1.0, 2.0, 0.5])
        controller = lambda t, x: np.array([0.0, 0.0])

        t, x, u = system.simulate(x0, controller, (0, 1), dt=0.1)

        np.testing.assert_array_equal(x[0], x0)

    def test_simulate_straight_line(self, system):
        """Moving forward with no rotation should go straight."""
        x0 = np.array([0.0, 0.0, 0.0])  # facing +x
        controller = lambda t, x: np.array([1.0, 0.0])  # v=1, ω=0

        t, x, u = system.simulate(x0, controller, (0, 1), dt=0.01, method="rk4")

        # After 1 second at v=1, should be at (1, 0)
        np.testing.assert_almost_equal(x[-1, 0], 1.0, decimal=2)
        np.testing.assert_almost_equal(x[-1, 1], 0.0, decimal=2)
        np.testing.assert_almost_equal(x[-1, 2], 0.0, decimal=2)

    def test_simulate_pure_rotation(self, system):
        """Rotating in place should only change heading."""
        x0 = np.array([1.0, 2.0, 0.0])
        controller = lambda t, x: np.array([0.0, 1.0])  # v=0, ω=1

        t, x, u = system.simulate(x0, controller, (0, np.pi), dt=0.01, method="rk4")

        # Position should remain the same
        np.testing.assert_almost_equal(x[-1, 0], 1.0, decimal=2)
        np.testing.assert_almost_equal(x[-1, 1], 2.0, decimal=2)
        # Heading should be π
        np.testing.assert_almost_equal(x[-1, 2], np.pi, decimal=2)

    def test_simulate_circular_motion(self, system):
        """Constant v and ω should produce circular motion."""
        x0 = np.array([1.0, 0.0, np.pi / 2])  # start right, facing up
        v = 1.0
        omega = 1.0  # radius = v/ω = 1
        controller = lambda t, x: np.array([v, omega])

        # Full circle takes 2π/ω = 2π seconds
        t, x, u = system.simulate(x0, controller, (0, 2 * np.pi), dt=0.01, method="rk4")

        # Should return to starting position
        np.testing.assert_almost_equal(x[-1, 0], x0[0], decimal=1)
        np.testing.assert_almost_equal(x[-1, 1], x0[1], decimal=1)


# =============================================================================
# Test: Trajectory Generation
# =============================================================================


class TestTrajectoryGeneration:
    """Tests for trajectory generation methods."""

    def test_circular_trajectory_ccw(self, system):
        """Counter-clockwise circular trajectory."""
        radius = 2.0
        speed = 1.0
        x0, controller = system.circular_trajectory(radius, speed)

        # Initial state should be on circle
        assert np.isclose(x0[0], radius)
        assert np.isclose(x0[1], 0.0)
        assert np.isclose(x0[2], np.pi / 2)  # facing up

        # Controller should give constant v and ω
        u = controller(0, x0)
        assert u[0] == speed
        assert np.isclose(u[1], speed / radius)

    def test_circular_trajectory_cw(self, system):
        """Clockwise circular trajectory."""
        radius = 2.0
        speed = 1.0
        x0, controller = system.circular_trajectory(radius, speed, clockwise=True)

        # Should face down
        assert np.isclose(x0[2], -np.pi / 2)

        # ω should be negative
        u = controller(0, x0)
        assert u[1] < 0

    def test_circular_trajectory_with_center(self, system):
        """Circular trajectory around non-origin center."""
        center = np.array([3.0, 4.0])
        radius = 1.0
        x0, _ = system.circular_trajectory(radius, 1.0, center=center)

        # Should start at center + (radius, 0)
        np.testing.assert_almost_equal(x0[0], center[0] + radius)
        np.testing.assert_almost_equal(x0[1], center[1])

    def test_circular_trajectory_simulation(self, system):
        """Simulate circular trajectory and verify return to start."""
        radius = 1.0
        speed = 1.0
        x0, controller = system.circular_trajectory(radius, speed)

        # Full circle
        t, x, u = system.simulate(x0, controller, (0, 2 * np.pi), dt=0.01, method="rk4")

        # Should return near starting position
        np.testing.assert_almost_equal(x[-1, 0], x0[0], decimal=1)
        np.testing.assert_almost_equal(x[-1, 1], x0[1], decimal=1)

    def test_straight_line_controller(self, system):
        """Go-to-point controller should reach target."""
        target = np.array([3.0, 4.0])
        x0 = np.array([0.0, 0.0, 0.0])

        controller = system.straight_line_controller(target)
        t, x, u = system.simulate(x0, controller, (0, 10), dt=0.01, method="rk4")

        # Should get close to target
        final_dist = system.distance_to_point(x[-1], target)
        assert final_dist < 0.5


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_unicycle(self):
        system = create_unicycle()
        assert isinstance(system, Unicycle)

    def test_create_unicycle_custom_bounds(self):
        system = create_unicycle(
            x_bounds=(-20.0, 20.0),
            v_bounds=(0.0, 5.0),
        )
        assert system.params.x_min == -20.0
        assert system.params.x_max == 20.0
        assert system.params.v_min == 0.0
        assert system.params.v_max == 5.0

    def test_forward_only_unicycle(self):
        system = forward_only_unicycle()
        assert system.params.v_min == 0.0
        assert system.params.v_max > 0.0

    def test_forward_only_custom_max(self):
        system = forward_only_unicycle(v_max=5.0, omega_max=2.0)
        assert system.params.v_max == 5.0
        assert system.params.omega_max == 2.0
        assert system.params.omega_min == -2.0


# =============================================================================
# Test: Nonholonomic Property
# =============================================================================


class TestNonholonomicProperty:
    """Tests verifying nonholonomic constraint behavior."""

    def test_cannot_move_sideways(self, system):
        """Unicycle cannot move perpendicular to heading."""
        x0 = np.array([0.0, 0.0, 0.0])  # facing +x

        # No control can make ẏ > 0 while ẋ = 0 and θ̇ = 0
        # The only way to have ẏ ≠ 0 is with v ≠ 0 and θ ≠ 0 or nπ
        for v in np.linspace(-2, 2, 10):
            for omega in np.linspace(-np.pi, np.pi, 10):
                u = np.array([v, omega])
                x_dot = system.f(x0, u)

                # At θ=0, ẏ should always equal 0 regardless of v
                # (because sin(0) = 0)
                np.testing.assert_almost_equal(x_dot[1], 0.0)

    def test_controllability(self, system):
        """Despite nonholonomy, system should be controllable."""
        # Start facing +x, want to reach point at (0, 1)
        x0 = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0])

        # Use go-to-point controller
        controller = system.straight_line_controller(target)
        t, x, u = system.simulate(x0, controller, (0, 10), dt=0.01, method="rk4")

        # Should reach target despite nonholonomic constraint
        final_dist = system.distance_to_point(x[-1], target)
        assert final_dist < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
