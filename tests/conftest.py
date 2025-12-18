"""
Pytest configuration and shared fixtures for DynaSim tests.
"""

import numpy as np
import pytest

# =============================================================================
# Random Seed Fixture
# =============================================================================


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def random_state(rng):
    """Generate a random state vector."""

    def _random_state(n):
        return rng.standard_normal(n)

    return _random_state


@pytest.fixture
def random_control(rng):
    """Generate a random control vector."""

    def _random_control(m):
        return rng.standard_normal(m)

    return _random_control


# =============================================================================
# Tolerance Fixtures
# =============================================================================


@pytest.fixture
def atol():
    """Absolute tolerance for floating point comparisons."""
    return 1e-10


@pytest.fixture
def rtol():
    """Relative tolerance for floating point comparisons."""
    return 1e-6


# =============================================================================
# Common Test Utilities
# =============================================================================


def assert_valid_rotation_matrix(C, tol=1e-10):
    """Assert that C is a valid rotation matrix."""
    assert C.shape == (3, 3), f"Expected shape (3,3), got {C.shape}"

    # Check orthogonality: C @ C.T = I
    identity_check = C @ C.T
    np.testing.assert_allclose(identity_check, np.eye(3), atol=tol, err_msg="Matrix is not orthogonal")

    # Check determinant = +1
    det = np.linalg.det(C)
    np.testing.assert_allclose(det, 1.0, atol=tol, err_msg=f"Determinant is {det}, expected 1.0")


def assert_unit_quaternion(q, tol=1e-10):
    """Assert that q is a unit quaternion."""
    assert q.shape == (4,), f"Expected shape (4,), got {q.shape}"

    norm = np.linalg.norm(q)
    np.testing.assert_allclose(norm, 1.0, atol=tol, err_msg=f"Quaternion norm is {norm}, expected 1.0")


# Make utilities available to tests
@pytest.fixture
def assert_rotation():
    """Fixture providing rotation matrix assertion."""
    return assert_valid_rotation_matrix


@pytest.fixture
def assert_unit_quat():
    """Fixture providing unit quaternion assertion."""
    return assert_unit_quaternion


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
