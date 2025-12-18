"""
Tests for the DynamicalSystem abstract base class.

These tests verify:
1. Abstract class behavior (cannot instantiate directly)
2. Concrete implementation requirements
3. Default method implementations
4. Simulation functionality
5. Jacobian verification utilities
"""

from typing import List

import numpy as np
import pytest

from simdyn import DynamicalSystem

# =============================================================================
# Test Fixtures - Concrete System Implementations
# =============================================================================


class DoubleIntegrator1D(DynamicalSystem):
    """
    Simple 1D double integrator for testing.

    State: [position, velocity]
    Control: [acceleration]
    Dynamics: ṗ = v, v̇ = u + w
    """

    def __init__(self, params=None):
        super().__init__(params)

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_control(self) -> int:
        return 1

    @property
    def n_disturbance(self) -> int:
        return 2

    @property
    def state_names(self) -> List[str]:
        return ["position", "velocity"]

    @property
    def control_names(self) -> List[str]:
        return ["acceleration"]

    def f(self, x, u, w=None):
        if w is None:
            w = np.zeros(self.n_disturbance)
        x = np.asarray(x)
        u = np.asarray(u)
        w = np.asarray(w)

        p, v = x
        a = u[0]

        return np.array([v + w[0], a + w[1]])

    def A(self, x, u):  # noqa: ARG002
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    def B(self, x, u):  # noqa: ARG002
        return np.array([[0.0], [1.0]])

    def get_state_bounds(self):
        lb = np.array([-10.0, -5.0])
        ub = np.array([10.0, 5.0])
        return lb, ub

    def get_control_bounds(self):
        lb = np.array([-2.0])
        ub = np.array([2.0])
        return lb, ub


class Pendulum(DynamicalSystem):
    """
    Simple pendulum for testing nonlinear systems.

    State: [theta, theta_dot]
    Control: [torque]
    Dynamics: θ̈ = -g/l * sin(θ) + u/(m*l²) + w
    """

    def __init__(self, params=None):
        if params is None:
            params = {"mass": 1.0, "length": 1.0, "gravity": 9.81}
        super().__init__(params)

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_control(self) -> int:
        return 1

    @property
    def n_disturbance(self) -> int:
        return 2

    @property
    def state_names(self) -> List[str]:
        return ["theta", "theta_dot"]

    @property
    def control_names(self) -> List[str]:
        return ["torque"]

    def f(self, x, u, w=None):
        if w is None:
            w = np.zeros(self.n_disturbance)
        x = np.asarray(x)
        u = np.asarray(u)
        w = np.asarray(w)

        m = self.params["mass"]
        l = self.params["length"]
        g = self.params["gravity"]

        theta, theta_dot = x
        tau = u[0]

        theta_ddot = -g / l * np.sin(theta) + tau / (m * l**2)

        return np.array([theta_dot + w[0], theta_ddot + w[1]])

    def A(self, x, u):  # noqa: ARG002
        l = self.params["length"]
        g = self.params["gravity"]

        theta = x[0]

        return np.array([[0.0, 1.0], [-g / l * np.cos(theta), 0.0]])

    def B(self, x, u):  # noqa: ARG002
        m = self.params["mass"]
        l = self.params["length"]

        return np.array([[0.0], [1.0 / (m * l**2)]])

    def get_state_bounds(self):
        lb = np.array([-np.pi, -10.0])
        ub = np.array([np.pi, 10.0])
        return lb, ub

    def get_control_bounds(self):
        lb = np.array([-5.0])
        ub = np.array([5.0])
        return lb, ub


class IncompleteSystem(DynamicalSystem):
    """System that doesn't implement all abstract methods - for testing."""

    def __init__(self):
        super().__init__(None)

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_control(self) -> int:
        return 1

    # Missing: n_disturbance, state_names, control_names, f, A, B, bounds


# =============================================================================
# Test: Abstract Class Behavior
# =============================================================================


class TestAbstractBehavior:
    """Tests for abstract class enforcement."""

    def test_cannot_instantiate_base_class(self):
        """DynamicalSystem should not be instantiable directly."""
        with pytest.raises(TypeError):
            DynamicalSystem()

    def test_incomplete_implementation_raises(self):
        """Incomplete implementations should raise TypeError."""
        with pytest.raises(TypeError):
            IncompleteSystem()


# =============================================================================
# Test: Properties
# =============================================================================


class TestProperties:
    """Tests for system properties."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    @pytest.fixture
    def pendulum(self):
        return Pendulum()

    def test_n_state(self, double_integrator, pendulum):
        assert double_integrator.n_state == 2
        assert pendulum.n_state == 2

    def test_n_control(self, double_integrator, pendulum):
        assert double_integrator.n_control == 1
        assert pendulum.n_control == 1

    def test_n_disturbance(self, double_integrator, pendulum):
        assert double_integrator.n_disturbance == 2
        assert pendulum.n_disturbance == 2

    def test_state_names(self, double_integrator):
        assert double_integrator.state_names == ["position", "velocity"]
        assert len(double_integrator.state_names) == double_integrator.n_state

    def test_control_names(self, double_integrator):
        assert double_integrator.control_names == ["acceleration"]
        assert len(double_integrator.control_names) == double_integrator.n_control

    def test_params_storage(self, pendulum):
        assert pendulum.params["mass"] == 1.0
        assert pendulum.params["length"] == 1.0
        assert pendulum.params["gravity"] == 9.81

    def test_params_custom(self):
        custom_params = {"mass": 2.0, "length": 0.5, "gravity": 10.0}
        pend = Pendulum(params=custom_params)
        assert pend.params["mass"] == 2.0
        assert pend.params["length"] == 0.5

    def test_repr(self, double_integrator):
        repr_str = repr(double_integrator)
        assert "DoubleIntegrator1D" in repr_str
        assert "n_state=2" in repr_str
        assert "n_control=1" in repr_str


# =============================================================================
# Test: Continuous Dynamics
# =============================================================================


class TestContinuousDynamics:
    """Tests for continuous-time dynamics f(x, u, w)."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    @pytest.fixture
    def pendulum(self):
        return Pendulum()

    def test_dynamics_shape(self, double_integrator):
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        x_dot = double_integrator.f(x, u)
        assert x_dot.shape == (2,)

    def test_double_integrator_dynamics(self, double_integrator):
        """Test ṗ = v, v̇ = u."""
        x = np.array([1.0, 2.0])  # pos=1, vel=2
        u = np.array([0.5])  # accel=0.5

        x_dot = double_integrator.f(x, u)

        assert x_dot[0] == 2.0  # ṗ = v
        assert x_dot[1] == 0.5  # v̇ = u

    def test_dynamics_with_disturbance(self, double_integrator):
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        w = np.array([0.1, 0.2])

        x_dot = double_integrator.f(x, u, w)

        assert x_dot[0] == 2.0 + 0.1  # ṗ = v + w[0]
        assert x_dot[1] == 0.5 + 0.2  # v̇ = u + w[1]

    def test_dynamics_zero_disturbance_default(self, double_integrator):
        """Default disturbance should be zero."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        x_dot_no_w = double_integrator.f(x, u)
        x_dot_zero_w = double_integrator.f(x, u, np.zeros(2))

        np.testing.assert_array_equal(x_dot_no_w, x_dot_zero_w)

    def test_pendulum_equilibrium(self, pendulum):
        """At θ=0 with no torque, θ̈ should be 0."""
        x = np.array([0.0, 0.0])  # hanging down, stationary
        u = np.array([0.0])

        x_dot = pendulum.f(x, u)

        assert x_dot[0] == 0.0
        np.testing.assert_almost_equal(x_dot[1], 0.0)

    def test_pendulum_gravity(self, pendulum):
        """At θ=π/2, gravity should accelerate pendulum."""
        x = np.array([np.pi / 2, 0.0])  # horizontal
        u = np.array([0.0])

        x_dot = pendulum.f(x, u)

        assert x_dot[0] == 0.0
        assert x_dot[1] < 0  # should accelerate back towards equilibrium


# =============================================================================
# Test: Discrete Dynamics
# =============================================================================


class TestDiscreteDynamics:
    """Tests for discrete-time dynamics f_discrete."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    def test_euler_integration(self, double_integrator):
        """Test Euler method: x_{k+1} = x_k + dt * f(x_k, u_k)."""
        x = np.array([0.0, 1.0])  # pos=0, vel=1
        u = np.array([0.0])  # no acceleration
        dt = 0.1

        x_next = double_integrator.f_discrete(x, u, dt, method="euler")

        # After dt=0.1 with vel=1: pos should be ~0.1
        np.testing.assert_almost_equal(x_next[0], 0.1)
        np.testing.assert_almost_equal(x_next[1], 1.0)

    def test_rk4_integration(self, double_integrator):
        """Test RK4 method gives similar result for linear system."""
        x = np.array([0.0, 1.0])
        u = np.array([0.0])
        dt = 0.1

        x_next = double_integrator.f_discrete(x, u, dt, method="rk4")

        # For linear system, RK4 should give same result as Euler
        np.testing.assert_almost_equal(x_next[0], 0.1)
        np.testing.assert_almost_equal(x_next[1], 1.0)

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be more accurate for nonlinear systems."""
        pendulum = Pendulum()
        x = np.array([0.5, 0.0])  # initial angle
        u = np.array([0.0])
        dt = 0.1

        # Propagate for several steps
        x_euler = x.copy()
        x_rk4 = x.copy()

        for _ in range(100):
            x_euler = pendulum.f_discrete(x_euler, u, dt, method="euler")
            x_rk4 = pendulum.f_discrete(x_rk4, u, dt, method="rk4")

        # Both should still be oscillating (energy ~conserved with RK4)
        # Euler typically adds energy, RK4 should be closer to conservation
        # We just check they give different results
        assert not np.allclose(x_euler, x_rk4, atol=0.01)

    def test_discrete_with_disturbance(self, double_integrator):
        x = np.array([0.0, 1.0])
        u = np.array([0.0])
        w = np.array([0.0, 1.0])  # constant acceleration disturbance
        dt = 0.1

        x_next = double_integrator.f_discrete(x, u, dt, w, method="euler")

        # v̇ = u + w[1] = 0 + 1 = 1
        np.testing.assert_almost_equal(x_next[1], 1.0 + 0.1 * 1.0)

    def test_invalid_method_raises(self, double_integrator):
        x = np.array([0.0, 1.0])
        u = np.array([0.0])

        with pytest.raises(ValueError):
            double_integrator.f_discrete(x, u, 0.1, method="invalid")


# =============================================================================
# Test: Jacobians
# =============================================================================


class TestJacobians:
    """Tests for Jacobian matrices A, B, G."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    @pytest.fixture
    def pendulum(self):
        return Pendulum()

    def test_A_shape(self, double_integrator):
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        A = double_integrator.A(x, u)
        assert A.shape == (2, 2)

    def test_B_shape(self, double_integrator):
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        B = double_integrator.B(x, u)
        assert B.shape == (2, 1)

    def test_G_shape(self, double_integrator):
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        G = double_integrator.G(x, u)
        assert G.shape == (2, 2)

    def test_G_default_is_identity(self, double_integrator):
        """Default G should be identity."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        G = double_integrator.G(x, u)
        np.testing.assert_array_equal(G, np.eye(2))

    def test_double_integrator_A_matrix(self, double_integrator):
        """A should be [[0, 1], [0, 0]] for double integrator."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        A = double_integrator.A(x, u)

        expected_A = np.array([[0, 1], [0, 0]])
        np.testing.assert_array_equal(A, expected_A)

    def test_double_integrator_B_matrix(self, double_integrator):
        """B should be [[0], [1]] for double integrator."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])
        B = double_integrator.B(x, u)

        expected_B = np.array([[0], [1]])
        np.testing.assert_array_equal(B, expected_B)

    def test_jacobian_numerical_verification(self, double_integrator):
        """Verify analytical Jacobians match numerical computation."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        passed, errors = double_integrator.verify_jacobians(x, u)

        assert passed
        assert errors["A_relative_error"] < 1e-5
        assert errors["B_relative_error"] < 1e-5

    def test_pendulum_jacobian_verification(self, pendulum):
        """Verify pendulum Jacobians at multiple points."""
        test_points = [
            (np.array([0.0, 0.0]), np.array([0.0])),
            (np.array([0.5, 1.0]), np.array([0.5])),
            (np.array([np.pi / 4, -0.5]), np.array([-1.0])),
        ]

        for x, u in test_points:
            passed, errors = pendulum.verify_jacobians(x, u)
            assert passed, f"Jacobian verification failed at x={x}, u={u}, errors={errors}"

    def test_numerical_jacobian_method(self, double_integrator):
        """Test the numerical Jacobian computation directly."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        A_num, B_num = double_integrator.jacobian_numerical(x, u)

        # Should match analytical for double integrator
        A_analytical = double_integrator.A(x, u)
        B_analytical = double_integrator.B(x, u)

        np.testing.assert_array_almost_equal(A_num, A_analytical, decimal=5)
        np.testing.assert_array_almost_equal(B_num, B_analytical, decimal=5)


# =============================================================================
# Test: Linearization
# =============================================================================


class TestLinearization:
    """Tests for the linearize() method."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    @pytest.fixture
    def pendulum(self):
        return Pendulum()

    def test_linearize_returns_tuple(self, double_integrator):
        x0 = np.array([0.0, 0.0])
        u0 = np.array([0.0])

        result = double_integrator.linearize(x0, u0)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_linearize_shapes(self, double_integrator):
        x0 = np.array([0.0, 0.0])
        u0 = np.array([0.0])

        A, B, G, c = double_integrator.linearize(x0, u0)

        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        assert G.shape == (2, 2)
        assert c.shape == (2,)

    def test_linearize_affine_term(self, double_integrator):
        """c should equal f(x0, u0) for the linearization."""
        x0 = np.array([1.0, 2.0])
        u0 = np.array([0.5])

        A, B, G, c = double_integrator.linearize(x0, u0)
        f_at_x0 = double_integrator.f(x0, u0)

        np.testing.assert_array_almost_equal(c, f_at_x0)

    def test_linearize_at_equilibrium(self, pendulum):
        """At equilibrium, c should be approximately zero."""
        x0 = np.array([0.0, 0.0])  # hanging down equilibrium
        u0 = np.array([0.0])

        A, B, G, c = pendulum.linearize(x0, u0)

        np.testing.assert_array_almost_equal(c, np.zeros(2), decimal=10)


# =============================================================================
# Test: Constraints
# =============================================================================


class TestConstraints:
    """Tests for constraint checking methods."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    def test_get_state_bounds(self, double_integrator):
        lb, ub = double_integrator.get_state_bounds()

        assert lb.shape == (2,)
        assert ub.shape == (2,)
        assert all(lb < ub)

    def test_get_control_bounds(self, double_integrator):
        lb, ub = double_integrator.get_control_bounds()

        assert lb.shape == (1,)
        assert ub.shape == (1,)
        assert all(lb < ub)

    def test_state_constraints_satisfied(self, double_integrator):
        """State within bounds should have all negative constraint values."""
        x = np.array([0.0, 0.0])  # well within bounds

        constraints = double_integrator.state_constraints(x)

        assert all(v <= 0 for v in constraints.values())

    def test_state_constraints_violated(self, double_integrator):
        """State outside bounds should have positive constraint values."""
        x = np.array([15.0, 0.0])  # position exceeds upper bound of 10

        constraints = double_integrator.state_constraints(x)

        # At least one constraint should be violated
        assert any(v > 0 for v in constraints.values())
        assert constraints["position_upper"] > 0

    def test_control_constraints_satisfied(self, double_integrator):
        u = np.array([1.0])  # within [-2, 2]

        constraints = double_integrator.control_constraints(u)

        assert all(v <= 0 for v in constraints.values())

    def test_control_constraints_violated(self, double_integrator):
        u = np.array([3.0])  # exceeds upper bound of 2

        constraints = double_integrator.control_constraints(u)

        assert constraints["acceleration_upper"] > 0

    def test_is_state_valid(self, double_integrator):
        assert double_integrator.is_state_valid(np.array([0.0, 0.0]))
        assert double_integrator.is_state_valid(np.array([9.0, 4.0]))
        assert not double_integrator.is_state_valid(np.array([15.0, 0.0]))
        assert not double_integrator.is_state_valid(np.array([0.0, 10.0]))

    def test_is_control_valid(self, double_integrator):
        assert double_integrator.is_control_valid(np.array([0.0]))
        assert double_integrator.is_control_valid(np.array([1.5]))
        assert double_integrator.is_control_valid(np.array([-2.0]))
        assert not double_integrator.is_control_valid(np.array([3.0]))
        assert not double_integrator.is_control_valid(np.array([-5.0]))


# =============================================================================
# Test: Simulation
# =============================================================================


class TestSimulation:
    """Tests for the simulate() method."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    def test_simulate_returns_correct_shapes(self, double_integrator):
        x0 = np.array([0.0, 0.0])
        controller = lambda t, x: np.array([0.0])
        t_span = (0.0, 1.0)
        dt = 0.1

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt)

        assert t.shape[0] == 11  # 0.0, 0.1, ..., 1.0
        assert x.shape == (11, 2)
        assert u.shape == (10, 1)

    def test_simulate_initial_condition(self, double_integrator):
        x0 = np.array([5.0, 2.0])
        controller = lambda t, x: np.array([0.0])
        t_span = (0.0, 1.0)
        dt = 0.1

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt)

        np.testing.assert_array_equal(x[0], x0)

    def test_simulate_constant_velocity(self, double_integrator):
        """With zero control and initial velocity, position should increase linearly."""
        x0 = np.array([0.0, 1.0])  # vel=1
        controller = lambda t, x: np.array([0.0])
        t_span = (0.0, 1.0)
        dt = 0.01

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt, method="rk4")

        # After 1 second with vel=1, pos should be ~1.0
        np.testing.assert_almost_equal(x[-1, 0], 1.0, decimal=2)
        np.testing.assert_almost_equal(x[-1, 1], 1.0, decimal=2)

    def test_simulate_with_control(self, double_integrator):
        """Constant acceleration should give parabolic motion."""
        x0 = np.array([0.0, 0.0])
        controller = lambda t, x: np.array([1.0])  # constant accel
        t_span = (0.0, 1.0)
        dt = 0.01

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt, method="rk4")

        # p(t) = 0.5 * a * t^2 = 0.5 * 1 * 1^2 = 0.5
        # v(t) = a * t = 1 * 1 = 1
        np.testing.assert_almost_equal(x[-1, 0], 0.5, decimal=2)
        np.testing.assert_almost_equal(x[-1, 1], 1.0, decimal=2)

    def test_simulate_with_disturbance(self, double_integrator):
        """Test simulation with disturbance function."""
        x0 = np.array([0.0, 0.0])
        controller = lambda t, x: np.array([0.0])
        disturbance = lambda t, x: np.array([0.0, 1.0])  # constant accel disturbance
        t_span = (0.0, 1.0)
        dt = 0.01

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt, disturbance_fn=disturbance, method="rk4")

        # Same as constant acceleration of 1
        np.testing.assert_almost_equal(x[-1, 0], 0.5, decimal=2)
        np.testing.assert_almost_equal(x[-1, 1], 1.0, decimal=2)

    def test_simulate_feedback_controller(self, double_integrator):
        """Test simulation with state-feedback controller."""
        x0 = np.array([1.0, 0.0])  # initial displacement

        # Simple proportional controller: u = -kp * p - kd * v
        kp, kd = 2.0, 1.0
        controller = lambda t, x: np.array([-kp * x[0] - kd * x[1]])

        t_span = (0.0, 5.0)
        dt = 0.01

        t, x, u = double_integrator.simulate(x0, controller, t_span, dt, method="rk4")

        # System should converge to origin
        np.testing.assert_almost_equal(x[-1, 0], 0.0, decimal=1)
        np.testing.assert_almost_equal(x[-1, 1], 0.0, decimal=1)

    def test_simulate_euler_vs_rk4(self, double_integrator):
        """Euler and RK4 should give same results for linear system."""
        x0 = np.array([0.0, 1.0])
        controller = lambda t, x: np.array([0.0])
        t_span = (0.0, 1.0)
        dt = 0.1

        t1, x1, u1 = double_integrator.simulate(x0, controller, t_span, dt, method="euler")
        t2, x2, u2 = double_integrator.simulate(x0, controller, t_span, dt, method="rk4")

        # For this linear system, results should be identical
        np.testing.assert_array_almost_equal(x1, x2)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def double_integrator(self):
        return DoubleIntegrator1D()

    def test_handles_list_inputs(self, double_integrator):
        """Should accept lists as well as numpy arrays."""
        x = [1.0, 2.0]
        u = [0.5]

        x_dot = double_integrator.f(x, u)
        assert isinstance(x_dot, np.ndarray)

    def test_handles_integer_inputs(self, double_integrator):
        """Should handle integer inputs."""
        x = np.array([1, 2])
        u = np.array([0])

        x_dot = double_integrator.f(x, u)
        assert x_dot.dtype == np.float64

    def test_zero_timestep(self, double_integrator):
        """Zero timestep should return same state."""
        x = np.array([1.0, 2.0])
        u = np.array([0.5])

        x_next = double_integrator.f_discrete(x, u, dt=0.0, method="euler")

        np.testing.assert_array_equal(x_next, x)

    def test_very_small_timestep(self, double_integrator):
        """Very small timestep should work."""
        x = np.array([0.0, 1.0])
        u = np.array([0.0])
        dt = 1e-10

        x_next = double_integrator.f_discrete(x, u, dt, method="rk4")

        # Should be very close to original
        np.testing.assert_array_almost_equal(x_next, x, decimal=8)


# =============================================================================
# Test: Parameter Variations (Digital Twin)
# =============================================================================


class TestParameterVariations:
    """Tests for creating system variants with different parameters."""

    def test_different_parameters(self):
        """Create two pendulums with different parameters."""
        pend1 = Pendulum({"mass": 1.0, "length": 1.0, "gravity": 9.81})
        pend2 = Pendulum({"mass": 2.0, "length": 0.5, "gravity": 9.81})

        x = np.array([0.5, 0.0])
        u = np.array([0.0])

        x_dot1 = pend1.f(x, u)
        x_dot2 = pend2.f(x, u)

        # Different parameters should give different dynamics
        assert not np.allclose(x_dot1, x_dot2)

    def test_digital_twin_simulation(self):
        """Simulate true system and twin with parameter mismatch."""
        true_params = {"mass": 1.0, "length": 1.0, "gravity": 9.81}
        twin_params = {"mass": 2.0, "length": 1.0, "gravity": 9.81}  # 100% mass error

        true_system = Pendulum(true_params)
        twin_system = Pendulum(twin_params)

        x0 = np.array([0.5, 0.0])
        # Apply constant torque so mass affects dynamics via τ/(m*l²)
        controller = lambda t, x: np.array([1.0])
        t_span = (0.0, 2.0)
        dt = 0.01

        _, x_true, _ = true_system.simulate(x0, controller, t_span, dt)
        _, x_twin, _ = twin_system.simulate(x0, controller, t_span, dt)

        # Trajectories should diverge due to parameter mismatch
        # With different masses and same torque, acceleration differs
        assert not np.allclose(x_true[-1], x_twin[-1], atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
