"""
Tests for numerical integrators.

These tests verify:
1. Euler integration accuracy and convergence
2. RK4 integration accuracy and convergence
3. Integrator registry functionality
4. Edge cases and error handling
"""

import numpy as np
import pytest

from simdyn.integrators import (
    INTEGRATOR_REGISTRY,
    euler_integrate,
    euler_step,
    get_integrator,
    list_integrators,
    rk4_discretize_jacobians,
    rk4_integrate,
    rk4_step,
)

# =============================================================================
# Test Dynamics Functions
# =============================================================================


def linear_decay(x, u, w=None):  # noqa: ARG001
    """ẋ = -x, exponential decay. Solution: x(t) = x0 * exp(-t)."""
    x = np.asarray(x)
    if w is not None:
        return -x + np.asarray(w)
    return -x


def harmonic_oscillator(x, u, w=None):  # noqa: ARG001
    """
    Simple harmonic oscillator: ẍ + x = 0
    State: [position, velocity]
    Solution: x(t) = A*cos(t) + B*sin(t)
    """
    x = np.asarray(x)
    x_dot = np.array([x[1], -x[0]])
    if w is not None:
        x_dot += np.asarray(w)
    return x_dot


def nonlinear_pendulum(x, u, w=None):  # noqa: ARG001
    """
    Nonlinear pendulum: θ̈ = -sin(θ)
    State: [theta, theta_dot]
    """
    x = np.asarray(x)
    x_dot = np.array([x[1], -np.sin(x[0])])
    if w is not None:
        x_dot += np.asarray(w)
    return x_dot


def forced_system(x, u, w=None):
    """ẋ = u (directly controlled)."""
    x = np.asarray(x)
    u = np.asarray(u)
    if w is not None:
        return u + np.asarray(w)
    return u


def double_integrator(x, u, w=None):
    """
    Double integrator: ṗ = v, v̇ = u
    State: [position, velocity]
    Control: [acceleration]
    """
    x = np.asarray(x)
    u = np.asarray(u)
    x_dot = np.array([x[1], u[0]])
    if w is not None:
        x_dot += np.asarray(w)
    return x_dot


# =============================================================================
# Test: Euler Integration
# =============================================================================


class TestEulerStep:
    """Tests for single Euler integration step."""

    def test_euler_step_shape(self):
        """Output should have same shape as input state."""
        x = np.array([1.0, 2.0])
        u = np.array([0.0])
        dt = 0.1

        x_next = euler_step(harmonic_oscillator, x, u, dt)

        assert x_next.shape == x.shape

    def test_euler_step_linear_decay(self):
        """Test Euler on exponential decay."""
        x = np.array([1.0])
        u = np.array([])  # no control
        dt = 0.1

        x_next = euler_step(linear_decay, x, u, dt)

        # x_next = x + dt * (-x) = x * (1 - dt) = 1 * 0.9 = 0.9
        np.testing.assert_almost_equal(x_next[0], 0.9)

    def test_euler_step_zero_dt(self):
        """Zero timestep should return same state."""
        x = np.array([1.0, 2.0])
        u = np.array([0.0])

        x_next = euler_step(harmonic_oscillator, x, u, dt=0.0)

        np.testing.assert_array_equal(x_next, x)

    def test_euler_step_with_disturbance(self):
        """Test Euler with disturbance input."""
        x = np.array([1.0])
        u = np.array([])
        w = np.array([0.5])
        dt = 0.1

        x_next = euler_step(linear_decay, x, u, dt, w)

        # ẋ = -x + w = -1 + 0.5 = -0.5
        # x_next = x + dt * ẋ = 1 + 0.1 * (-0.5) = 0.95
        np.testing.assert_almost_equal(x_next[0], 0.95)

    def test_euler_step_with_control(self):
        """Test Euler with control input."""
        x = np.array([0.0, 0.0])
        u = np.array([1.0])  # acceleration = 1
        dt = 0.1

        x_next = euler_step(double_integrator, x, u, dt)

        # ṗ = v = 0, v̇ = u = 1
        # p_next = 0 + 0.1 * 0 = 0
        # v_next = 0 + 0.1 * 1 = 0.1
        np.testing.assert_almost_equal(x_next[0], 0.0)
        np.testing.assert_almost_equal(x_next[1], 0.1)


class TestEulerIntegrate:
    """Tests for multi-step Euler integration."""

    def test_euler_integrate_shape(self):
        """Output trajectory should have correct shape."""
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((100, 1))
        dt = 0.01

        traj = euler_integrate(harmonic_oscillator, x0, u_seq, dt)

        assert traj.shape == (101, 2)  # N+1 states for N controls

    def test_euler_integrate_initial_condition(self):
        """First state should be initial condition."""
        x0 = np.array([5.0, 3.0])
        u_seq = np.zeros((10, 1))
        dt = 0.1

        traj = euler_integrate(harmonic_oscillator, x0, u_seq, dt)

        np.testing.assert_array_equal(traj[0], x0)

    def test_euler_integrate_exponential_decay(self):
        """Test accuracy on exponential decay."""
        x0 = np.array([1.0])
        N = 100
        u_seq = np.zeros((N, 0))
        dt = 0.01

        traj = euler_integrate(linear_decay, x0, u_seq, dt)

        # Analytical: x(1) = exp(-1) ≈ 0.3679
        # Euler error is O(dt), so with dt=0.01, expect ~1% error
        t_final = N * dt
        x_analytical = np.exp(-t_final)

        np.testing.assert_almost_equal(traj[-1, 0], x_analytical, decimal=2)

    def test_euler_integrate_with_disturbance(self):
        """Test integration with disturbance sequence."""
        x0 = np.array([0.0])
        N = 10
        u_seq = np.zeros((N, 0))
        w_seq = np.ones((N, 1)) * 0.5
        dt = 0.1

        traj = euler_integrate(linear_decay, x0, u_seq, dt, w_seq)

        # With constant disturbance, system has different equilibrium
        assert traj[-1, 0] > 0  # Should drift positive


class TestEulerConvergence:
    """Tests for Euler convergence rate."""

    def test_euler_first_order_convergence(self):
        """Euler should show first-order convergence."""
        x0 = np.array([1.0])
        t_final = 1.0
        x_exact = np.exp(-t_final)

        errors = []
        dts = [0.1, 0.05, 0.025, 0.0125]

        for dt in dts:
            N = int(t_final / dt)
            u_seq = np.zeros((N, 0))
            traj = euler_integrate(linear_decay, x0, u_seq, dt)
            error = np.abs(traj[-1, 0] - x_exact)
            errors.append(error)

        # Check convergence rate: error should halve when dt halves
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            # First-order: ratio should be ~2
            assert 1.8 < ratio < 2.2, f"Convergence ratio {ratio} not ~2"


# =============================================================================
# Test: RK4 Integration
# =============================================================================


class TestRK4Step:
    """Tests for single RK4 integration step."""

    def test_rk4_step_shape(self):
        """Output should have same shape as input state."""
        x = np.array([1.0, 2.0])
        u = np.array([0.0])
        dt = 0.1

        x_next = rk4_step(harmonic_oscillator, x, u, dt)

        assert x_next.shape == x.shape

    def test_rk4_step_linear_decay(self):
        """Test RK4 on exponential decay."""
        x = np.array([1.0])
        u = np.array([])
        dt = 0.1

        x_next = rk4_step(linear_decay, x, u, dt)

        # Analytical: x(0.1) = exp(-0.1) ≈ 0.9048
        x_analytical = np.exp(-0.1)

        np.testing.assert_almost_equal(x_next[0], x_analytical, decimal=5)

    def test_rk4_step_zero_dt(self):
        """Zero timestep should return same state."""
        x = np.array([1.0, 2.0])
        u = np.array([0.0])

        x_next = rk4_step(harmonic_oscillator, x, u, dt=0.0)

        np.testing.assert_array_equal(x_next, x)

    def test_rk4_step_with_disturbance(self):
        """Test RK4 with disturbance input."""
        x = np.array([1.0])
        u = np.array([])
        w = np.array([0.5])
        dt = 0.1

        x_next = rk4_step(linear_decay, x, u, dt, w)

        # Should be close to integrating ẋ = -x + 0.5
        # Different from Euler due to higher accuracy
        assert 0.9 < x_next[0] < 1.0

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be more accurate than Euler for same dt."""
        x = np.array([1.0])
        u = np.array([])
        dt = 0.1

        x_euler = euler_step(linear_decay, x, u, dt)
        x_rk4 = rk4_step(linear_decay, x, u, dt)
        x_exact = np.exp(-dt)

        error_euler = np.abs(x_euler[0] - x_exact)
        error_rk4 = np.abs(x_rk4[0] - x_exact)

        assert error_rk4 < error_euler


class TestRK4Integrate:
    """Tests for multi-step RK4 integration."""

    def test_rk4_integrate_shape(self):
        """Output trajectory should have correct shape."""
        x0 = np.array([1.0, 0.0])
        u_seq = np.zeros((100, 1))
        dt = 0.01

        traj = rk4_integrate(harmonic_oscillator, x0, u_seq, dt)

        assert traj.shape == (101, 2)

    def test_rk4_integrate_initial_condition(self):
        """First state should be initial condition."""
        x0 = np.array([5.0, 3.0])
        u_seq = np.zeros((10, 1))
        dt = 0.1

        traj = rk4_integrate(harmonic_oscillator, x0, u_seq, dt)

        np.testing.assert_array_equal(traj[0], x0)

    def test_rk4_integrate_exponential_decay(self):
        """Test accuracy on exponential decay."""
        x0 = np.array([1.0])
        N = 100
        u_seq = np.zeros((N, 0))
        dt = 0.01

        traj = rk4_integrate(linear_decay, x0, u_seq, dt)

        t_final = N * dt
        x_analytical = np.exp(-t_final)

        # RK4 should be very accurate
        np.testing.assert_almost_equal(traj[-1, 0], x_analytical, decimal=6)

    def test_rk4_harmonic_oscillator_energy(self):
        """Harmonic oscillator should approximately conserve energy."""
        x0 = np.array([1.0, 0.0])  # initial displacement, zero velocity
        N = 1000
        u_seq = np.zeros((N, 1))
        dt = 0.01

        traj = rk4_integrate(harmonic_oscillator, x0, u_seq, dt)

        # Energy = 0.5 * (x^2 + v^2) for unit mass and spring constant
        E0 = 0.5 * (x0[0] ** 2 + x0[1] ** 2)
        E_final = 0.5 * (traj[-1, 0] ** 2 + traj[-1, 1] ** 2)

        # Energy should be conserved to high precision with RK4
        np.testing.assert_almost_equal(E_final, E0, decimal=4)


class TestRK4Convergence:
    """Tests for RK4 convergence rate."""

    def test_rk4_fourth_order_convergence(self):
        """RK4 should show fourth-order convergence."""
        x0 = np.array([1.0])
        t_final = 1.0
        x_exact = np.exp(-t_final)

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            N = int(t_final / dt)
            u_seq = np.zeros((N, 0))
            traj = rk4_integrate(linear_decay, x0, u_seq, dt)
            error = np.abs(traj[-1, 0] - x_exact)
            errors.append(error)

        # Check convergence rate: error should decrease by factor of 16 when dt halves
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            # Fourth-order: ratio should be ~16
            assert 12 < ratio < 20, f"Convergence ratio {ratio} not ~16"


class TestRK4DiscretizeJacobians:
    """Tests for Jacobian discretization."""

    def test_discretize_identity(self):
        """Zero A matrix should give A_d = I."""
        A_c = np.zeros((2, 2))
        B_c = np.array([[0], [1]])
        dt = 0.1

        A_d, B_d = rk4_discretize_jacobians(A_c, B_c, dt)

        np.testing.assert_array_almost_equal(A_d, np.eye(2))

    def test_discretize_double_integrator(self):
        """Test discretization of double integrator."""
        A_c = np.array([[0, 1], [0, 0]])
        B_c = np.array([[0], [1]])
        dt = 0.1

        A_d, B_d = rk4_discretize_jacobians(A_c, B_c, dt)

        # For double integrator, A_d should be [[1, dt], [0, 1]]
        expected_A_d = np.array([[1, dt], [0, 1]])
        np.testing.assert_array_almost_equal(A_d, expected_A_d, decimal=4)

    def test_discretize_shapes(self):
        """Output shapes should match continuous shapes."""
        A_c = np.random.randn(3, 3)
        B_c = np.random.randn(3, 2)
        dt = 0.1

        A_d, B_d = rk4_discretize_jacobians(A_c, B_c, dt)

        assert A_d.shape == A_c.shape
        assert B_d.shape == B_c.shape


# =============================================================================
# Test: Integrator Registry
# =============================================================================


class TestIntegratorRegistry:
    """Tests for integrator registry functionality."""

    def test_list_integrators(self):
        """Should list available integrators."""
        integrators = list_integrators()

        assert "euler" in integrators
        assert "rk4" in integrators
        assert len(integrators) >= 2

    def test_get_integrator_euler(self):
        """Should retrieve Euler integrator."""
        step_fn = get_integrator("euler")

        assert callable(step_fn)
        assert step_fn == euler_step

    def test_get_integrator_rk4(self):
        """Should retrieve RK4 integrator."""
        step_fn = get_integrator("rk4")

        assert callable(step_fn)
        assert step_fn == rk4_step

    def test_get_integrator_invalid(self):
        """Invalid integrator name should raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_integrator("invalid_method")

        assert "invalid_method" in str(excinfo.value)
        assert "euler" in str(excinfo.value)  # Should suggest valid options

    def test_registry_contains_functions(self):
        """Registry should map to callable functions."""
        for name, fn in INTEGRATOR_REGISTRY.items():
            assert callable(fn)

    def test_get_integrator_usage(self):
        """Test using get_integrator for integration."""
        x = np.array([1.0])
        u = np.array([])
        dt = 0.1

        for method in list_integrators():
            step_fn = get_integrator(method)
            x_next = step_fn(linear_decay, x, u, dt)

            assert x_next.shape == x.shape
            assert 0 < x_next[0] < 1  # Should decay


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_timestep(self):
        """Very small timestep should work."""
        x = np.array([1.0])
        u = np.array([])
        dt = 1e-10

        x_euler = euler_step(linear_decay, x, u, dt)
        x_rk4 = rk4_step(linear_decay, x, u, dt)

        np.testing.assert_almost_equal(x_euler[0], 1.0, decimal=8)
        np.testing.assert_almost_equal(x_rk4[0], 1.0, decimal=8)

    def test_large_timestep_stability(self):
        """Large timestep may be unstable for Euler."""
        x = np.array([1.0])
        u = np.array([])
        dt = 1.5  # Euler should be unstable for linear decay when dt > 2

        x_euler = euler_step(linear_decay, x, u, dt)
        x_rk4 = rk4_step(linear_decay, x, u, dt)

        # Euler: x_next = x * (1 - dt) = 1 * (1 - 1.5) = -0.5 (wrong sign!)
        # RK4 should still give reasonable result
        assert x_euler[0] < 0  # Euler overshoots
        assert x_rk4[0] > 0  # RK4 more stable

    def test_handles_list_inputs(self):
        """Should accept lists as inputs."""
        x = [1.0, 2.0]
        u = [0.0]
        dt = 0.1

        x_next = rk4_step(harmonic_oscillator, x, u, dt)

        assert isinstance(x_next, np.ndarray)

    def test_handles_scalar_state(self):
        """Should handle scalar state."""
        x = np.array([5.0])
        u = np.array([])
        dt = 0.1

        x_next = rk4_step(linear_decay, x, u, dt)

        assert x_next.shape == (1,)

    def test_high_dimensional_state(self):
        """Should handle high-dimensional state."""
        n = 100

        def high_dim_decay(x, u, w=None):  # noqa: ARG001
            return -x

        x = np.ones(n)
        u = np.array([])
        dt = 0.1

        x_next = rk4_step(high_dim_decay, x, u, dt)

        assert x_next.shape == (n,)
        assert np.all(x_next < x)  # All components should decay


# =============================================================================
# Test: Comparison with Analytical Solutions
# =============================================================================


class TestAnalyticalComparison:
    """Tests comparing numerical solutions to analytical solutions."""

    def test_exponential_decay_long_integration(self):
        """Long integration of exponential decay."""
        x0 = np.array([10.0])
        t_final = 5.0
        dt = 0.001
        N = int(t_final / dt)
        u_seq = np.zeros((N, 0))

        traj = rk4_integrate(linear_decay, x0, u_seq, dt)

        x_exact = 10.0 * np.exp(-t_final)

        np.testing.assert_almost_equal(traj[-1, 0], x_exact, decimal=5)

    def test_harmonic_oscillator_period(self):
        """Harmonic oscillator should complete one period in 2π."""
        x0 = np.array([1.0, 0.0])
        t_final = 2 * np.pi
        dt = 0.001
        N = int(t_final / dt)
        u_seq = np.zeros((N, 1))

        traj = rk4_integrate(harmonic_oscillator, x0, u_seq, dt)

        # After one period, should return to initial state
        np.testing.assert_almost_equal(traj[-1, 0], x0[0], decimal=3)
        np.testing.assert_almost_equal(traj[-1, 1], x0[1], decimal=3)

    def test_forced_double_integrator(self):
        """Double integrator with constant force."""
        x0 = np.array([0.0, 0.0])  # start at rest
        t_final = 1.0
        dt = 0.001
        N = int(t_final / dt)

        # Constant acceleration of 1
        u_seq = np.ones((N, 1))

        traj = rk4_integrate(double_integrator, x0, u_seq, dt)

        # Analytical: p(t) = 0.5*a*t^2, v(t) = a*t
        p_exact = 0.5 * 1.0 * t_final**2  # 0.5
        v_exact = 1.0 * t_final  # 1.0

        np.testing.assert_almost_equal(traj[-1, 0], p_exact, decimal=4)
        np.testing.assert_almost_equal(traj[-1, 1], v_exact, decimal=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
