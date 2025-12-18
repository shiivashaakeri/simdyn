"""
Abstract base class for dynamical systems.

All dynamical systems in simdyn inherit from this class and must implement
the required abstract methods and properties.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from simdyn.integrators import get_integrator


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.

    This class defines the interface that all dynamical systems must implement.
    It provides common functionality for simulation, linearization, and constraint
    checking while requiring subclasses to define system-specific dynamics and Jacobians.

    Attributes
    ----------
    params : object
        System parameters (mass, inertia, etc.). Structure depends on the specific system.

    Notes
    -----
    All dynamics functions accept an optional disturbance input `w` which defaults to zero.
    This is critical for data-driven control research where process noise must be modeled.
    """

    def __init__(self, params=None):
        """
        Initialize the dynamical system.

        Parameters
        ----------
        params : object, optional
            System parameters. If None, subclasses should use default parameters.
        """
        self.params = params

    # =========================================================================
    # Abstract Properties - Subclasses MUST define these
    # =========================================================================

    @property
    @abstractmethod
    def n_state(self) -> int:
        """Dimension of the state vector x."""
        pass

    @property
    @abstractmethod
    def n_control(self) -> int:
        """Dimension of the control vector u."""
        pass

    @property
    @abstractmethod
    def n_disturbance(self) -> int:
        """Dimension of the disturbance vector w."""
        pass

    @property
    @abstractmethod
    def state_names(self) -> List[str]:
        """Human-readable names for each state element."""
        pass

    @property
    @abstractmethod
    def control_names(self) -> List[str]:
        """Human-readable names for each control element."""
        pass

    # =========================================================================
    # Abstract Methods - Subclasses MUST implement these
    # =========================================================================

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u, w).

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            Current state vector.
        u : np.ndarray, shape (n_control,)
            Control input vector.
        w : np.ndarray, shape (n_disturbance,), optional
            Disturbance vector. Defaults to zeros if not provided.

        Returns
        -------
        np.ndarray, shape (n_state,)
            State derivative ẋ.
        """
        pass

    @abstractmethod
    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State Jacobian matrix: ∂f/∂x.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector at which to evaluate the Jacobian.
        u : np.ndarray, shape (n_control,)
            Control vector at which to evaluate the Jacobian.

        Returns
        -------
        np.ndarray, shape (n_state, n_state)
            State Jacobian matrix.
        """
        pass

    @abstractmethod
    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Control Jacobian matrix: ∂f/∂u.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector at which to evaluate the Jacobian.
        u : np.ndarray, shape (n_control,)
            Control vector at which to evaluate the Jacobian.

        Returns
        -------
        np.ndarray, shape (n_state, n_control)
            Control Jacobian matrix.
        """
        pass

    @abstractmethod
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lower and upper bounds on the state vector.

        Returns
        -------
        lb : np.ndarray, shape (n_state,)
            Lower bounds for each state element. Use -np.inf for unbounded.
        ub : np.ndarray, shape (n_state,)
            Upper bounds for each state element. Use np.inf for unbounded.
        """
        pass

    @abstractmethod
    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lower and upper bounds on the control vector.

        Returns
        -------
        lb : np.ndarray, shape (n_control,)
            Lower bounds for each control element. Use -np.inf for unbounded.
        ub : np.ndarray, shape (n_control,)
            Upper bounds for each control element. Use np.inf for unbounded.
        """
        pass

    # =========================================================================
    # Concrete Methods - Default implementations (can be overridden)
    # =========================================================================

    def G(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Disturbance Jacobian matrix: ∂f/∂w.

        Default implementation returns an identity matrix, meaning disturbance
        enters additively on all state derivatives. Override for different
        disturbance structures.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector at which to evaluate the Jacobian.
        u : np.ndarray, shape (n_control,)
            Control vector at which to evaluate the Jacobian.

        Returns
        -------
        np.ndarray, shape (n_state, n_disturbance)
            Disturbance Jacobian matrix.
        """
        return np.eye(self.n_state, self.n_disturbance)

    def f_discrete(
        self, x: np.ndarray, u: np.ndarray, dt: float, w: Optional[np.ndarray] = None, method: str = "rk4"
    ) -> np.ndarray:
        """
        Discrete-time dynamics: x_{k+1} = f_d(x_k, u_k, w_k).

        Integrates the continuous-time dynamics over one time step.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            Current state vector.
        u : np.ndarray, shape (n_control,)
            Control input vector (held constant over the time step).
        dt : float
            Time step duration.
        w : np.ndarray, shape (n_disturbance,), optional
            Disturbance vector. Defaults to zeros if not provided.
        method : str, optional
            Integration method. Options: 'euler', 'rk4'. Default is 'rk4'.

        Returns
        -------
        np.ndarray, shape (n_state,)
            Next state x_{k+1}.

        Raises
        ------
        ValueError
            If an unknown integration method is specified.
        """
        # Default disturbance to zeros
        if w is None:
            w = np.zeros(self.n_disturbance)

        # Ensure inputs are numpy arrays
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)

        # Get the integrator step function
        step_fn = get_integrator(method)

        # Integrate using the selected method
        x_next = step_fn(self.f, x, u, dt, w)

        return x_next

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the dynamics around an operating point.

        Computes the linearization: ẋ ≈ A(x - x0) + B(u - u0) + Gw + c
        where c = f(x0, u0) is the dynamics at the operating point.

        Parameters
        ----------
        x0 : np.ndarray, shape (n_state,)
            State operating point.
        u0 : np.ndarray, shape (n_control,)
            Control operating point.

        Returns
        -------
        A : np.ndarray, shape (n_state, n_state)
            State Jacobian at operating point.
        B : np.ndarray, shape (n_state, n_control)
            Control Jacobian at operating point.
        G : np.ndarray, shape (n_state, n_disturbance)
            Disturbance Jacobian at operating point.
        c : np.ndarray, shape (n_state,)
            Affine term (dynamics at operating point).
        """
        x0 = np.asarray(x0, dtype=np.float64)
        u0 = np.asarray(u0, dtype=np.float64)

        A = self.A(x0, u0)
        B = self.B(x0, u0)
        G = self.G(x0, u0)
        c = self.f(x0, u0)

        return A, B, G, c

    def state_constraints(self, x: np.ndarray) -> Dict[str, float]:
        """
        Evaluate state constraint violations.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector to check.

        Returns
        -------
        dict
            Dictionary with constraint names and values.
            Negative values indicate satisfied constraints.
            Format: {'state_i_lower': x_i - lb_i, 'state_i_upper': x_i - ub_i}
        """
        x = np.asarray(x, dtype=np.float64)
        lb, ub = self.get_state_bounds()

        constraints = {}
        for i, name in enumerate(self.state_names):
            # Lower bound: x_i >= lb_i  =>  lb_i - x_i <= 0
            if lb[i] > -np.inf:
                constraints[f"{name}_lower"] = lb[i] - x[i]
            # Upper bound: x_i <= ub_i  =>  x_i - ub_i <= 0
            if ub[i] < np.inf:
                constraints[f"{name}_upper"] = x[i] - ub[i]

        return constraints

    def control_constraints(self, u: np.ndarray) -> Dict[str, float]:
        """
        Evaluate control constraint violations.

        Parameters
        ----------
        u : np.ndarray, shape (n_control,)
            Control vector to check.

        Returns
        -------
        dict
            Dictionary with constraint names and values.
            Negative values indicate satisfied constraints.
        """
        u = np.asarray(u, dtype=np.float64)
        lb, ub = self.get_control_bounds()

        constraints = {}
        for i, name in enumerate(self.control_names):
            # Lower bound: u_i >= lb_i  =>  lb_i - u_i <= 0
            if lb[i] > -np.inf:
                constraints[f"{name}_lower"] = lb[i] - u[i]
            # Upper bound: u_i <= ub_i  =>  u_i - ub_i <= 0
            if ub[i] < np.inf:
                constraints[f"{name}_upper"] = u[i] - ub[i]

        return constraints

    def is_state_valid(self, x: np.ndarray) -> bool:
        """
        Check if a state satisfies all constraints.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State vector to check.

        Returns
        -------
        bool
            True if all state constraints are satisfied.
        """
        constraints = self.state_constraints(x)
        return all(v <= 0 for v in constraints.values())

    def is_control_valid(self, u: np.ndarray) -> bool:
        """
        Check if a control input satisfies all constraints.

        Parameters
        ----------
        u : np.ndarray, shape (n_control,)
            Control vector to check.

        Returns
        -------
        bool
            True if all control constraints are satisfied.
        """
        constraints = self.control_constraints(u)
        return all(v <= 0 for v in constraints.values())

    def simulate(
        self,
        x0: np.ndarray,
        controller: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        dt: float,
        disturbance_fn: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        method: str = "rk4",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a closed-loop simulation.

        Parameters
        ----------
        x0 : np.ndarray, shape (n_state,)
            Initial state.
        controller : callable
            Control law function with signature u = controller(t, x).
        t_span : tuple of float
            Time span (t_start, t_end).
        dt : float
            Time step for integration.
        disturbance_fn : callable, optional
            Disturbance function with signature w = disturbance_fn(t, x).
            If None, disturbance is zero.
        method : str, optional
            Integration method ('euler' or 'rk4'). Default is 'rk4'.

        Returns
        -------
        t : np.ndarray, shape (n_steps,)
            Time points.
        x : np.ndarray, shape (n_steps, n_state)
            State trajectory.
        u : np.ndarray, shape (n_steps-1, n_control)
            Control inputs applied at each step.
        """
        t_start, t_end = t_span

        # Generate time array
        t = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t)

        # Pre-allocate arrays
        x = np.zeros((n_steps, self.n_state))
        u = np.zeros((n_steps - 1, self.n_control))

        # Set initial state
        x[0] = np.asarray(x0, dtype=np.float64)

        # Simulation loop
        for k in range(n_steps - 1):
            # Get control input
            u_k = controller(t[k], x[k])
            u[k] = u_k

            # Get disturbance
            w_k = disturbance_fn(t[k], x[k]) if disturbance_fn is not None else None

            # Propagate dynamics
            x[k + 1] = self.f_discrete(x[k], u_k, dt, w_k, method=method)

        return t, x, u

    def jacobian_numerical(self, x: np.ndarray, u: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians numerically using central differences.

        Useful for verifying analytical Jacobian implementations.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State at which to compute Jacobians.
        u : np.ndarray, shape (n_control,)
            Control at which to compute Jacobians.
        eps : float, optional
            Perturbation size for finite differences.

        Returns
        -------
        A_num : np.ndarray, shape (n_state, n_state)
            Numerical state Jacobian.
        B_num : np.ndarray, shape (n_state, n_control)
            Numerical control Jacobian.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)

        # State Jacobian
        A_num = np.zeros((self.n_state, self.n_state))
        for j in range(self.n_state):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            A_num[:, j] = (self.f(x_plus, u) - self.f(x_minus, u)) / (2 * eps)

        # Control Jacobian
        B_num = np.zeros((self.n_state, self.n_control))
        for j in range(self.n_control):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[j] += eps
            u_minus[j] -= eps
            B_num[:, j] = (self.f(x, u_plus) - self.f(x, u_minus)) / (2 * eps)

        return A_num, B_num

    def verify_jacobians(
        self, x: np.ndarray, u: np.ndarray, eps: float = 1e-6, tol: float = 1e-5
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Verify analytical Jacobians against numerical computation.

        Parameters
        ----------
        x : np.ndarray, shape (n_state,)
            State at which to verify.
        u : np.ndarray, shape (n_control,)
            Control at which to verify.
        eps : float, optional
            Perturbation size for numerical Jacobians.
        tol : float, optional
            Tolerance for relative error.

        Returns
        -------
        passed : bool
            True if both Jacobians are within tolerance.
        errors : dict
            Dictionary with relative errors for A and B matrices.
        """
        A_analytical = self.A(x, u)
        B_analytical = self.B(x, u)
        A_numerical, B_numerical = self.jacobian_numerical(x, u, eps)

        # Compute relative errors
        A_norm = np.linalg.norm(A_analytical)
        B_norm = np.linalg.norm(B_analytical)

        if A_norm > 0:
            A_error = np.linalg.norm(A_analytical - A_numerical) / A_norm
        else:
            A_error = np.linalg.norm(A_analytical - A_numerical)

        if B_norm > 0:
            B_error = np.linalg.norm(B_analytical - B_numerical) / B_norm
        else:
            B_error = np.linalg.norm(B_analytical - B_numerical)

        passed = (A_error < tol) and (B_error < tol)
        errors = {"A_relative_error": A_error, "B_relative_error": B_error}

        return passed, errors

    def __repr__(self) -> str:
        """String representation of the system."""
        return (
            f"{self.__class__.__name__}("
            f"n_state={self.n_state}, "
            f"n_control={self.n_control}, "
            f"n_disturbance={self.n_disturbance})"
        )
