"""
Smoke tests to verify package imports and basic functionality.

These tests run quickly and verify that the package is correctly installed.
"""

import numpy as np
import pytest


class TestPackageImports:
    """Test that all package components can be imported."""

    def test_import_package(self):
        """Package should be importable."""
        import simdyn  # noqa: PLC0415

        assert hasattr(simdyn, "__version__")

    def test_import_base_class(self):
        """Base class should be importable."""
        from simdyn import DynamicalSystem  # noqa: PLC0415

        assert DynamicalSystem is not None

    def test_import_integrators(self):
        """Integrators should be importable."""
        from simdyn import euler_step, get_integrator, rk4_step  # noqa: PLC0415

        assert callable(euler_step)
        assert callable(rk4_step)
        assert callable(get_integrator)

    def test_import_quaternion_utils(self):
        """Quaternion utilities should be importable."""
        from simdyn import (  # noqa: PLC0415
            quat_multiply,
            quat_to_dcm,
        )

        assert callable(quat_multiply)
        assert callable(quat_to_dcm)

    def test_import_rotation_utils(self):
        """Rotation utilities should be importable."""
        from simdyn import (  # noqa: PLC0415
            euler_to_dcm,
            rotx,
        )

        assert callable(rotx)
        assert callable(euler_to_dcm)


class TestBasicFunctionality:
    """Quick tests for basic functionality."""

    def test_version_string(self):
        """Version should be a valid string."""
        import simdyn  # noqa: PLC0415

        assert isinstance(simdyn.__version__, str)
        assert len(simdyn.__version__) > 0

    def test_euler_step_runs(self):
        """Euler step should run without error."""
        from simdyn import euler_step  # noqa: PLC0415

        def simple_dynamics(x, u, w=None):  # noqa: ARG001
            return -x

        x = np.array([1.0])
        u = np.array([])
        dt = 0.1

        x_next = euler_step(simple_dynamics, x, u, dt)
        assert x_next.shape == (1,)

    def test_rk4_step_runs(self):
        """RK4 step should run without error."""
        from simdyn import rk4_step  # noqa: PLC0415

        def simple_dynamics(x, u, w=None):  # noqa: ARG001
            return -x

        x = np.array([1.0])
        u = np.array([])
        dt = 0.1

        x_next = rk4_step(simple_dynamics, x, u, dt)
        assert x_next.shape == (1,)

    def test_quat_identity_correct(self):
        """Identity quaternion should be [1,0,0,0]."""
        from simdyn import quat_identity  # noqa: PLC0415

        q = quat_identity()
        np.testing.assert_array_equal(q, [1, 0, 0, 0])

    def test_rotz_90_degrees(self):
        """90° rotation about z should work correctly."""
        from simdyn import rotz  # noqa: PLC0415

        R = rotz(np.pi / 2)
        v = np.array([1, 0, 0])
        result = R @ v

        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_integrator_registry(self):
        """Integrator registry should work."""
        from simdyn import get_integrator, list_integrators  # noqa: PLC0415

        integrators = list_integrators()
        assert "euler" in integrators
        assert "rk4" in integrators

        euler = get_integrator("euler")
        rk4 = get_integrator("rk4")

        assert callable(euler)
        assert callable(rk4)

    def test_skew_matrix(self):
        """Skew matrix should produce cross product."""
        from simdyn import skew  # noqa: PLC0415

        v = np.array([1, 2, 3])
        u = np.array([4, 5, 6])

        S = skew(v)
        result = S @ u
        expected = np.cross(v, u)

        np.testing.assert_array_almost_equal(result, expected)


class TestErrorHandling:
    """Test that errors are raised appropriately."""

    def test_invalid_integrator_raises(self):
        """Invalid integrator should raise ValueError."""
        from simdyn import get_integrator  # noqa: PLC0415

        with pytest.raises(ValueError):
            get_integrator("invalid")

    def test_abstract_class_not_instantiable(self):
        """DynamicalSystem should not be instantiable."""
        from simdyn import DynamicalSystem  # noqa: PLC0415

        with pytest.raises(TypeError):
            DynamicalSystem()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
