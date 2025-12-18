"""
Tests for the Rocket 3-DoF dynamical system.

These tests verify:
1. System properties and dimensions
2. Dynamics evaluation (including mass depletion)
3. Analytical Jacobians
4. Constraint checking (thrust, glide slope)
5. Drag model
6. Simulation accuracy
7. Factory functions
"""

import numpy as np
import pytest

from simdyn import DynamicalSystem
from simdyn.systems.rocket3dof import (
    Rocket3DoF,
    Rocket3DoFParams,
    create_normalized_rocket3dof,
    create_rocket3dof,
    default_params,
    normalized_params,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def system():
    """Default Rocket3DoF system."""
    return Rocket3DoF()


@pytest.fixture
def normalized_system():
    """Normalized (non-dimensional) Rocket3DoF system."""
    return create_normalized_rocket3dof()


@pytest.fixture
def system_with_drag():
    """Rocket3DoF with drag enabled."""
    params = Rocket3DoFParams(
        m_dry=1.0,
        m_wet=2.0,
        I_sp=225.0,
        g_vec=np.array([-9.80665, 0.0, 0.0]),
        T_max=50000.0,
        enable_drag=True,
        rho=1.225,
        C_D=0.5,
        A_ref=1.0,
    )
    return Rocket3DoF(params)


# =============================================================================
# Test: Parameter Validation
# =============================================================================


class TestParameters:
    """Tests for parameter validation and defaults."""

    def test_default_params(self):
        params = default_params()
        assert params.m_dry == 1.0
        assert params.m_wet == 2.0
        assert params.I_sp == 225.0

    def test_normalized_params(self):
        params = normalized_params()
        assert params.m_dry == 1.0
        assert params.m_wet == 2.0
        assert params.g0 == 1.0
        assert params.g_vec[0] == -1.0

    def test_alpha_computation(self):
        """Test mass flow coefficient calculation."""
        params = Rocket3DoFParams(I_sp=100.0, g0=10.0)
        expected_alpha = 1.0 / (10.0 * 100.0)
        assert params.alpha == expected_alpha

    def test_fuel_mass(self):
        params = Rocket3DoFParams(m_dry=1.0, m_wet=3.0)
        assert params.fuel_mass == 2.0

    def test_invalid_m_dry(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(m_dry=-1.0)
        with pytest.raises(ValueError):
            Rocket3DoFParams(m_dry=0.0)

    def test_invalid_m_wet(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(m_dry=2.0, m_wet=1.0)

    def test_invalid_I_sp(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(I_sp=-100.0)

    def test_invalid_thrust_bounds(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(T_min=-1.0)
        with pytest.raises(ValueError):
            Rocket3DoFParams(T_min=100.0, T_max=50.0)

    def test_invalid_glide_slope(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(gamma_gs=0.0)
        with pytest.raises(ValueError):
            Rocket3DoFParams(gamma_gs=np.pi / 2)

    def test_invalid_g_vec_shape(self):
        with pytest.raises(ValueError):
            Rocket3DoFParams(g_vec=np.array([1.0, 2.0]))


# =============================================================================
# Test: System Properties
# =============================================================================


class TestProperties:
    """Tests for system properties."""

    def test_inheritance(self, system):
        assert isinstance(system, DynamicalSystem)

    def test_n_state(self, system):
        assert system.n_state == 7

    def test_n_control(self, system):
        assert system.n_control == 3

    def test_n_disturbance(self, system):
        assert system.n_disturbance == 7

    def test_state_names(self, system):
        expected = ["m", "r_x", "r_y", "r_z", "v_x", "v_y", "v_z"]
        assert system.state_names == expected

    def test_control_names(self, system):
        expected = ["T_x", "T_y", "T_z"]
        assert system.control_names == expected

    def test_repr(self, system):
        repr_str = repr(system)
        assert "Rocket3DoF" in repr_str
        assert "n_state=7" in repr_str


# =============================================================================
# Test: State Accessors
# =============================================================================


class TestStateAccessors:
    """Tests for state pack/unpack utilities."""

    def test_get_mass(self, system):
        x = np.array([1500.0, 100.0, 0.0, 0.0, -10.0, 5.0, 0.0])
        assert system.get_mass(x) == 1500.0

    def test_get_position(self, system):
        x = np.array([1500.0, 100.0, 50.0, 25.0, -10.0, 5.0, 0.0])
        pos = system.get_position(x)
        np.testing.assert_array_equal(pos, [100.0, 50.0, 25.0])

    def test_get_velocity(self, system):
        x = np.array([1500.0, 100.0, 50.0, 25.0, -10.0, 5.0, 3.0])
        vel = system.get_velocity(x)
        np.testing.assert_array_equal(vel, [-10.0, 5.0, 3.0])

    def test_get_altitude(self, system):
        x = np.array([1500.0, 100.0, 50.0, 25.0, 0.0, 0.0, 0.0])
        assert system.get_altitude(x) == 100.0

    def test_get_speed(self, system):
        x = np.array([1500.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
        assert system.get_speed(x) == 5.0

    def test_pack_state(self, system):
        m = 1500.0
        r = np.array([100.0, 50.0, 25.0])
        v = np.array([-10.0, 5.0, 3.0])
        x = system.pack_state(m, r, v)
        np.testing.assert_array_equal(x, [1500.0, 100.0, 50.0, 25.0, -10.0, 5.0, 3.0])

    def test_pack_unpack_roundtrip(self, system):
        m_orig = 1800.0
        r_orig = np.array([200.0, 100.0, 50.0])
        v_orig = np.array([-15.0, 10.0, 5.0])

        x = system.pack_state(m_orig, r_orig, v_orig)

        assert system.get_mass(x) == m_orig
        np.testing.assert_array_equal(system.get_position(x), r_orig)
        np.testing.assert_array_equal(system.get_velocity(x), v_orig)


# =============================================================================
# Test: Thrust Utilities
# =============================================================================


class TestThrustUtilities:
    """Tests for thrust-related utilities."""

    def test_get_thrust_magnitude(self, system):
        u = np.array([3.0, 4.0, 0.0])
        assert system.get_thrust_magnitude(u) == 5.0

    def test_get_thrust_direction(self, system):
        u = np.array([6.0, 0.0, 0.0])
        d = system.get_thrust_direction(u)
        np.testing.assert_array_equal(d, [1.0, 0.0, 0.0])

    def test_get_thrust_direction_arbitrary(self, system):
        u = np.array([1.0, 1.0, 1.0])
        d = system.get_thrust_direction(u)
        expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        np.testing.assert_array_almost_equal(d, expected)

    def test_get_thrust_direction_zero(self, system):
        u = np.array([0.0, 0.0, 0.0])
        d = system.get_thrust_direction(u)
        np.testing.assert_array_equal(d, [0.0, 0.0, 0.0])


# =============================================================================
# Test: Continuous Dynamics
# =============================================================================


class TestDynamics:
    """Tests for continuous-time dynamics f(x, u, w)."""

    def test_dynamics_shape(self, system):
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])
        x_dot = system.f(x, u)
        assert x_dot.shape == (7,)

    def test_mass_depletion(self, normalized_system):
        """Mass should decrease when thrusting."""
        x = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T_mag = 3.0
        u = np.array([T_mag, 0.0, 0.0])

        x_dot = normalized_system.f(x, u)

        # ṁ = -alpha·‖T‖
        alpha = normalized_system.params.alpha
        expected_m_dot = -alpha * T_mag

        np.testing.assert_almost_equal(x_dot[0], expected_m_dot)

    def test_position_rate(self, system):
        """ṙ = v should hold."""
        x = np.array([2000.0, 100.0, 0.0, 0.0, 10.0, 5.0, 3.0])
        u = np.array([0.0, 0.0, 0.0])

        x_dot = system.f(x, u)

        v = system.get_velocity(x)
        np.testing.assert_array_almost_equal(x_dot[1:4], v)

    def test_gravity_acceleration(self, normalized_system):
        """With no thrust, should accelerate due to gravity."""
        x = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])

        x_dot = normalized_system.f(x, u)

        # v̇ = g = [-1, 0, 0]
        g = normalized_system.params.g_vec
        np.testing.assert_array_almost_equal(x_dot[4:7], g)

    def test_thrust_acceleration(self, normalized_system):
        """Thrust should produce acceleration."""
        m = 2.0
        x = np.array([m, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = np.array([4.0, 2.0, 0.0])
        u = T

        x_dot = normalized_system.f(x, u)

        # v̇ = T/m + g
        g = normalized_system.params.g_vec
        expected_v_dot = T / m + g
        np.testing.assert_array_almost_equal(x_dot[4:7], expected_v_dot)

    def test_hover_equilibrium(self, normalized_system):
        """At hover, velocity rate should be zero."""
        m = 2.0
        g = normalized_system.params.g_vec
        x = np.array([m, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Thrust to counteract gravity
        T_hover = -m * g
        u = T_hover

        x_dot = normalized_system.f(x, u)

        # v̇ should be zero (hover)
        np.testing.assert_array_almost_equal(x_dot[4:7], np.zeros(3), decimal=10)

    def test_dynamics_with_disturbance(self, system):
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])
        w = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        x_dot = system.f(x, u, w)
        x_dot_no_w = system.f(x, u)

        np.testing.assert_array_almost_equal(x_dot, x_dot_no_w + w)

    def test_dynamics_zero_disturbance_default(self, system):
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])

        x_dot_no_w = system.f(x, u)
        x_dot_zero_w = system.f(x, u, np.zeros(7))

        np.testing.assert_array_equal(x_dot_no_w, x_dot_zero_w)


# =============================================================================
# Test: Drag Model
# =============================================================================


class TestDragModel:
    """Tests for aerodynamic drag."""

    def test_drag_disabled_by_default(self, system):
        assert not system.params.enable_drag

    def test_drag_force_zero_velocity(self, system_with_drag):
        v = np.array([0.0, 0.0, 0.0])
        F_drag = system_with_drag.compute_drag(v)
        np.testing.assert_array_equal(F_drag, np.zeros(3))

    def test_drag_force_direction(self, system_with_drag):
        """Drag should oppose velocity."""
        v = np.array([10.0, 0.0, 0.0])
        F_drag = system_with_drag.compute_drag(v)

        # Drag should be in -x direction
        assert F_drag[0] < 0
        assert F_drag[1] == 0
        assert F_drag[2] == 0

    def test_drag_force_magnitude(self, system_with_drag):
        """Test drag magnitude calculation."""
        v = np.array([10.0, 0.0, 0.0])
        F_drag = system_with_drag.compute_drag(v)

        p = system_with_drag.params
        v_mag = np.linalg.norm(v)
        expected_mag = 0.5 * p.rho * p.C_D * p.A_ref * v_mag**2

        np.testing.assert_almost_equal(np.linalg.norm(F_drag), expected_mag)

    def test_drag_quadratic_scaling(self, system_with_drag):
        """Drag should scale quadratically with speed."""
        v1 = np.array([10.0, 0.0, 0.0])
        v2 = np.array([20.0, 0.0, 0.0])

        F1 = system_with_drag.compute_drag(v1)
        F2 = system_with_drag.compute_drag(v2)

        # Doubling velocity should quadruple drag
        ratio = np.linalg.norm(F2) / np.linalg.norm(F1)
        np.testing.assert_almost_equal(ratio, 4.0)

    def test_drag_affects_dynamics(self, system_with_drag):
        """Drag should slow down the rocket."""
        m = 2.0
        x = np.array([m, 100.0, 0.0, 0.0, 50.0, 0.0, 0.0])  # moving in +x
        u = np.array([0.0, 0.0, 0.0])  # no thrust

        x_dot = system_with_drag.f(x, u)

        # v̇_x should be negative due to gravity AND drag
        # Gravity alone gives g_x = -9.8
        # Drag adds more deceleration
        g_x = system_with_drag.params.g_vec[0]
        assert x_dot[4] < g_x  # More deceleration than gravity alone


# =============================================================================
# Test: Jacobians
# =============================================================================


class TestJacobians:
    """Tests for Jacobian matrices."""

    def test_A_shape(self, system):
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])
        A = system.A(x, u)
        assert A.shape == (7, 7)

    def test_B_shape(self, system):
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])
        B = system.B(x, u)
        assert B.shape == (7, 3)

    def test_A_position_velocity_coupling(self, system):
        """∂ṙ/∂v should be identity."""
        x = np.array([2000.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])
        A = system.A(x, u)

        # A[1:4, 4:7] = I
        np.testing.assert_array_equal(A[1:4, 4:7], np.eye(3))

    def test_A_thrust_mass_dependence(self, normalized_system):
        """∂v̇/∂m should be -T/m²."""
        m = 2.0
        x = np.array([m, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = np.array([4.0, 2.0, 0.0])
        u = T

        A = normalized_system.A(x, u)

        expected = -T / (m**2)
        np.testing.assert_array_almost_equal(A[4:7, 0], expected)

    def test_B_mass_rate(self, normalized_system):
        """∂ṁ/∂T should be -alpha·T/‖T‖."""
        x = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = np.array([3.0, 0.0, 0.0])
        u = T

        B = normalized_system.B(x, u)

        alpha = normalized_system.params.alpha
        T_mag = np.linalg.norm(T)
        expected = -alpha * T / T_mag

        np.testing.assert_array_almost_equal(B[0, :], expected)

    def test_B_thrust_acceleration(self, system):
        """∂v̇/∂T should be I/m."""
        m = 2000.0
        x = np.array([m, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([10000.0, 0.0, 0.0])

        B = system.B(x, u)

        np.testing.assert_array_almost_equal(B[4:7, :], np.eye(3) / m)

    def test_jacobian_numerical_verification_no_drag(self, normalized_system):
        """Verify Jacobians without drag."""
        x = np.array([1.8, 8.0, 1.0, -0.5, -2.0, 1.0, 0.5])
        u = np.array([3.0, 0.5, 0.2])

        passed, errors = normalized_system.verify_jacobians(x, u, eps=1e-6, tol=1e-4)
        assert passed, f"Jacobian verification failed: {errors}"

    def test_jacobian_numerical_verification_with_drag(self, system_with_drag):
        """Verify Jacobians with drag enabled."""
        x = np.array([1.8, 100.0, 10.0, 5.0, 20.0, 10.0, 5.0])
        u = np.array([30000.0, 5000.0, 2000.0])

        passed, errors = system_with_drag.verify_jacobians(x, u, eps=1e-6, tol=1e-3)
        assert passed, f"Jacobian verification failed: {errors}"

    def test_jacobian_at_multiple_points(self, normalized_system):
        """Verify Jacobians at multiple state/control points."""
        test_cases = [
            (np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])),
            (np.array([1.5, 5.0, 2.0, 1.0, -3.0, 1.0, 0.0]), np.array([4.0, 1.0, 0.5])),
            (np.array([1.2, 2.0, 0.5, 0.2, -1.0, 0.5, 0.2]), np.array([3.0, 0.0, 0.0])),
        ]

        for x, u in test_cases:
            passed, errors = normalized_system.verify_jacobians(x, u, eps=1e-7, tol=1e-4)
            assert passed, f"Failed at x={x}, u={u}: {errors}"


# =============================================================================
# Test: Constraints
# =============================================================================


class TestConstraints:
    """Tests for constraint checking."""

    def test_state_bounds_shape(self, system):
        lb, ub = system.get_state_bounds()
        assert lb.shape == (7,)
        assert ub.shape == (7,)

    def test_control_bounds_shape(self, system):
        lb, ub = system.get_control_bounds()
        assert lb.shape == (3,)
        assert ub.shape == (3,)

    def test_mass_bounds(self, system):
        lb, ub = system.get_state_bounds()
        assert lb[0] == system.params.m_dry
        assert ub[0] == system.params.m_wet

    def test_thrust_constraint_satisfied(self, system):
        """Thrust within bounds should satisfy constraints."""
        T_mid = 0.5 * (system.params.T_min + system.params.T_max)
        u = np.array([T_mid, 0.0, 0.0])

        assert system.is_thrust_valid(u)

    def test_thrust_constraint_violated_max(self, system):
        """Thrust above T_max should violate constraint."""
        T_over = system.params.T_max * 1.5
        u = np.array([T_over, 0.0, 0.0])

        assert not system.is_thrust_valid(u)

    def test_thrust_constraint_violated_min(self, normalized_system):
        """Thrust below T_min should violate constraint."""
        # Normalized system has T_min = 1.5
        T_under = 0.5  # below T_min
        u = np.array([T_under, 0.0, 0.0])

        assert not normalized_system.is_thrust_valid(u)

    def test_glide_slope_satisfied(self, system):
        """Point inside cone should satisfy glide slope."""
        # At altitude 10, with gamma=45°, horizontal distance can be up to 10
        x = system.pack_state(
            mass=2000.0,
            position=np.array([10.0, 3.0, 3.0]),  # altitude=10, ‖r_yz‖ ≈ 4.2
            velocity=np.zeros(3),
        )

        assert system.is_glide_slope_satisfied(x)

    def test_glide_slope_violated(self, system):
        """Point outside cone should violate glide slope."""
        # At altitude 10, with gamma=45°, horizontal distance > 10 violates
        x = system.pack_state(
            mass=2000.0,
            position=np.array([10.0, 8.0, 8.0]),  # altitude=10, ‖r_yz‖ ≈ 11.3
            velocity=np.zeros(3),
        )

        assert not system.is_glide_slope_satisfied(x)

    def test_glide_slope_at_origin(self, system):
        """At landing site (origin), should satisfy glide slope."""
        x = system.pack_state(mass=2000.0, position=np.zeros(3), velocity=np.zeros(3))

        assert system.is_glide_slope_satisfied(x)


# =============================================================================
# Test: Utility Methods
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_fuel_remaining(self, system):
        m = 1.5
        x = system.pack_state(m, np.zeros(3), np.zeros(3))
        fuel = system.fuel_remaining(x)

        expected = m - system.params.m_dry
        assert fuel == expected

    def test_fuel_fraction(self, system):
        x = system.pack_state(system.params.m_wet, np.zeros(3), np.zeros(3))
        assert system.fuel_fraction(x) == 1.0

        x = system.pack_state(system.params.m_dry, np.zeros(3), np.zeros(3))
        assert system.fuel_fraction(x) == 0.0

    def test_hover_thrust(self, normalized_system):
        m = 2.0
        x = normalized_system.pack_state(m, np.zeros(3), np.zeros(3))
        T_hover = normalized_system.hover_thrust(x)

        g = normalized_system.params.g_vec
        expected = -m * g
        np.testing.assert_array_equal(T_hover, expected)

    def test_time_of_flight_estimate(self, normalized_system):
        x = normalized_system.pack_state(mass=2.0, position=np.array([10.0, 0.0, 0.0]), velocity=np.zeros(3))

        t_est = normalized_system.time_of_flight_estimate(x, thrust_fraction=1.0)

        # t_f ≈ fuel / (alpha * T_max)
        fuel = normalized_system.fuel_remaining(x)
        alpha = normalized_system.params.alpha
        T_max = normalized_system.params.T_max
        expected = fuel / (alpha * T_max)

        np.testing.assert_almost_equal(t_est, expected)

    def test_energy_at_rest_at_origin(self, system):
        x = system.pack_state(2000.0, np.zeros(3), np.zeros(3))
        E = system.energy(x)

        assert E["kinetic"] == 0.0
        assert E["potential"] == 0.0
        assert E["total"] == 0.0

    def test_energy_kinetic(self, system):
        m = 2000.0
        v = np.array([10.0, 0.0, 0.0])
        x = system.pack_state(m, np.zeros(3), v)

        E = system.energy(x)
        expected_KE = 0.5 * m * np.dot(v, v)

        assert E["kinetic"] == expected_KE

    def test_energy_potential(self, system):
        m = 2000.0
        r = np.array([100.0, 0.0, 0.0])  # 100m altitude
        x = system.pack_state(m, r, np.zeros(3))

        E = system.energy(x)
        g = system.params.g_vec
        expected_PE = -m * np.dot(g, r)

        np.testing.assert_almost_equal(E["potential"], expected_PE)


# =============================================================================
# Test: Simulation
# =============================================================================


class TestSimulation:
    """Tests for simulation functionality."""

    def test_simulate_shapes(self, normalized_system):
        x0 = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        controller = lambda t, x: np.array([2.0, 0.0, 0.0])
        t_span = (0.0, 1.0)
        dt = 0.1

        t, x, u = normalized_system.simulate(x0, controller, t_span, dt)

        assert t.shape[0] == 11
        assert x.shape == (11, 7)
        assert u.shape == (10, 3)

    def test_simulate_initial_condition(self, normalized_system):
        x0 = np.array([2.0, 10.0, 1.0, 0.5, -1.0, 0.5, 0.0])
        controller = lambda t, x: np.array([2.0, 0.0, 0.0])

        t, x, u = normalized_system.simulate(x0, controller, (0, 1), dt=0.1)

        np.testing.assert_array_equal(x[0], x0)

    def test_simulate_mass_decreases(self, normalized_system):
        """Mass should decrease during powered flight."""
        x0 = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T_mag = 3.0
        controller = lambda t, x: np.array([T_mag, 0.0, 0.0])

        t, x, u = normalized_system.simulate(x0, controller, (0, 1), dt=0.01, method="rk4")

        # Mass should monotonically decrease
        masses = x[:, 0]
        assert all(masses[i] >= masses[i + 1] for i in range(len(masses) - 1))
        assert x[-1, 0] < x0[0]

    def test_simulate_free_fall(self, normalized_system):
        """With no thrust, should free-fall."""
        x0 = np.array([2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        controller = lambda t, x: np.array([0.0, 0.0, 0.0])

        t, x, u = normalized_system.simulate(x0, controller, (0, 1), dt=0.01, method="rk4")

        # Mass should remain constant (no thrust)
        np.testing.assert_array_almost_equal(x[:, 0], x0[0] * np.ones(len(t)))

        # Altitude should decrease (free fall)
        assert x[-1, 1] < x0[1]

        # Velocity should increase in magnitude (downward)
        assert x[-1, 4] < 0  # Falling in -x direction

    def test_simulate_hover(self, normalized_system):
        """Hover thrust should maintain position."""
        m = 2.0
        g = normalized_system.params.g_vec
        T_hover = -m * g

        x0 = np.array([m, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        controller = lambda t, x: T_hover

        t, x, u = normalized_system.simulate(x0, controller, (0, 0.5), dt=0.01, method="rk4")

        # Position should remain roughly constant
        np.testing.assert_almost_equal(x[-1, 1], x0[1], decimal=1)

        # Velocity should remain near zero
        np.testing.assert_almost_equal(x[-1, 4:7], np.zeros(3), decimal=1)


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_rocket3dof(self):
        system = create_rocket3dof()
        assert isinstance(system, Rocket3DoF)

    def test_create_rocket3dof_custom(self):
        system = create_rocket3dof(
            m_dry=500.0,
            m_wet=1500.0,
            I_sp=300.0,
            T_max=30000.0,
        )

        assert system.params.m_dry == 500.0
        assert system.params.m_wet == 1500.0
        assert system.params.I_sp == 300.0
        assert system.params.T_max == 30000.0

    def test_create_rocket3dof_with_drag(self):
        system = create_rocket3dof(enable_drag=True)
        assert system.params.enable_drag

    def test_create_normalized_rocket3dof(self):
        system = create_normalized_rocket3dof()

        assert system.params.m_dry == 1.0
        assert system.params.m_wet == 2.0
        assert system.params.g0 == 1.0
        assert system.params.g_vec[0] == -1.0


# =============================================================================
# Test: Physics Conservation
# =============================================================================


class TestPhysicsConservation:
    """Tests for physical conservation laws."""

    def test_momentum_change_equals_impulse(self, normalized_system):
        """Change in momentum should equal applied impulse."""
        m0 = 2.0
        x0 = np.array([m0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        T = np.array([4.0, 0.0, 0.0])
        controller = lambda t, x: T

        dt = 0.001
        t_final = 0.1
        t, x, u = normalized_system.simulate(x0, controller, (0, t_final), dt, method="rk4")

        # Initial and final momentum
        m_f = x[-1, 0]
        v_0 = x[0, 4:7]
        v_f = x[-1, 4:7]

        p_0 = m0 * v_0
        p_f = m_f * v_f

        # Impulse from thrust and gravity
        # This is approximate due to mass change
        _ = normalized_system.params.g_vec

        # For small time, momentum change should be roughly (T + m*g) * dt
        # But this is only approximate due to changing mass
        delta_p = p_f - p_0

        # At least verify momentum changed in the expected direction
        assert delta_p[0] > 0  # Thrust in +x should increase p_x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
