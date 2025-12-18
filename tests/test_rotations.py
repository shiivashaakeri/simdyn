"""
Tests for rotation utilities (quaternions, DCM, Euler angles).

These tests verify:
1. Quaternion operations and properties
2. DCM operations and validity
3. Euler angle conversions
4. Consistency between representations
5. Edge cases (gimbal lock, singularities)
"""

import numpy as np
import pytest

from simdyn.utils.quaternion import (
    dcm_to_quat,
    omega_matrix,
    quat_angle,
    quat_axis,
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_from_axis_angle,
    quat_identity,
    quat_integrate,
    quat_integrate_exact,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_random,
    quat_rotate,
    quat_rotate_inverse,
    quat_slerp,
    quat_to_dcm,
)
from simdyn.utils.rotations import (
    angle_difference,
    angular_velocity_to_euler_rates,
    dcm_angle,
    dcm_axis,
    dcm_from_axis_angle,
    dcm_is_valid,
    dcm_normalize,
    dcm_to_euler,
    euler_rates_to_angular_velocity,
    euler_to_dcm,
    euler_to_quat,
    quat_to_euler,
    rotx,
    roty,
    rotz,
    skew,
    unskew,
    wrap_angle,
    wrap_angle_positive,
)

# =============================================================================
# Test: Quaternion Basic Operations
# =============================================================================


class TestQuaternionBasics:
    """Tests for basic quaternion operations."""

    def test_quat_identity(self):
        """Identity quaternion should be [1, 0, 0, 0]."""
        q = quat_identity()
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(q, expected)

    def test_quat_normalize(self):
        """Normalized quaternion should have unit length."""
        q = np.array([1.0, 1.0, 1.0, 1.0])
        q_norm = quat_normalize(q)

        assert np.isclose(np.linalg.norm(q_norm), 1.0)

    def test_quat_normalize_already_unit(self):
        """Already unit quaternion should be unchanged."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_norm = quat_normalize(q)

        np.testing.assert_array_almost_equal(q_norm, q)

    def test_quat_normalize_zero(self):
        """Zero quaternion should return identity."""
        q = np.array([0.0, 0.0, 0.0, 0.0])
        q_norm = quat_normalize(q)

        np.testing.assert_array_equal(q_norm, quat_identity())

    def test_quat_conjugate(self):
        """Conjugate should negate vector part."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = quat_conjugate(q)

        expected = np.array([0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_equal(q_conj, expected)

    def test_quat_inverse_unit(self):
        """Inverse of unit quaternion equals conjugate."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_inv = quat_inverse(q)
        q_conj = quat_conjugate(q)

        np.testing.assert_array_almost_equal(q_inv, q_conj)

    def test_quat_inverse_property(self):
        """q * q^-1 should equal identity."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_inv = quat_inverse(q)

        result = quat_multiply(q, q_inv)

        np.testing.assert_array_almost_equal(result, quat_identity())


class TestQuaternionMultiplication:
    """Tests for quaternion multiplication."""

    def test_multiply_identity_left(self):
        """Identity * q should equal q."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        identity = quat_identity()

        result = quat_multiply(identity, q)

        np.testing.assert_array_almost_equal(result, q)

    def test_multiply_identity_right(self):
        """q * identity should equal q."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        identity = quat_identity()

        result = quat_multiply(q, identity)

        np.testing.assert_array_almost_equal(result, q)

    def test_multiply_conjugate(self):
        """q * q* should equal identity for unit quaternions."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q_conj = quat_conjugate(q)

        result = quat_multiply(q, q_conj)

        np.testing.assert_array_almost_equal(result, quat_identity())

    def test_multiply_not_commutative(self):
        """Quaternion multiplication is not commutative."""
        q1 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        q2 = quat_from_axis_angle(np.array([0, 1, 0]), np.pi / 4)

        result1 = quat_multiply(q1, q2)
        result2 = quat_multiply(q2, q1)

        assert not np.allclose(result1, result2)

    def test_multiply_associative(self):
        """Quaternion multiplication should be associative."""
        q1 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        q2 = quat_from_axis_angle(np.array([0, 1, 0]), np.pi / 3)
        q3 = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 6)

        result1 = quat_multiply(quat_multiply(q1, q2), q3)
        result2 = quat_multiply(q1, quat_multiply(q2, q3))

        np.testing.assert_array_almost_equal(result1, result2)


class TestQuaternionAxisAngle:
    """Tests for axis-angle representation."""

    def test_from_axis_angle_identity(self):
        """Zero angle should give identity."""
        axis = np.array([1, 0, 0])
        angle = 0.0

        q = quat_from_axis_angle(axis, angle)

        np.testing.assert_array_almost_equal(q, quat_identity())

    def test_from_axis_angle_90_deg_x(self):
        """90° about x-axis."""
        axis = np.array([1, 0, 0])
        angle = np.pi / 2

        q = quat_from_axis_angle(axis, angle)

        # q = [cos(45°), sin(45°), 0, 0]
        expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        np.testing.assert_array_almost_equal(q, expected)

    def test_from_axis_angle_180_deg(self):
        """180° rotation."""
        axis = np.array([0, 0, 1])
        angle = np.pi

        q = quat_from_axis_angle(axis, angle)

        # q = [cos(90°), 0, 0, sin(90°)] = [0, 0, 0, 1]
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_almost_equal(q, expected)

    def test_quat_angle_extraction(self):
        """Extract angle from quaternion."""
        axis = np.array([1, 0, 0])
        angle = np.pi / 3

        q = quat_from_axis_angle(axis, angle)
        extracted_angle = quat_angle(q)

        np.testing.assert_almost_equal(extracted_angle, angle)

    def test_quat_axis_extraction(self):
        """Extract axis from quaternion."""
        axis = np.array([1, 0, 0])
        angle = np.pi / 3

        q = quat_from_axis_angle(axis, angle)
        extracted_axis = quat_axis(q)

        np.testing.assert_array_almost_equal(extracted_axis, axis)

    def test_quat_axis_arbitrary(self):
        """Extract axis for arbitrary rotation."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 4

        q = quat_from_axis_angle(axis, angle)
        extracted_axis = quat_axis(q)

        np.testing.assert_array_almost_equal(extracted_axis, axis)


# =============================================================================
# Test: Quaternion-DCM Conversions
# =============================================================================


class TestQuaternionDCMConversion:
    """Tests for quaternion to/from DCM conversions."""

    def test_identity_quat_to_dcm(self):
        """Identity quaternion should give identity DCM."""
        q = quat_identity()
        C = quat_to_dcm(q)

        np.testing.assert_array_almost_equal(C, np.eye(3))

    def test_identity_dcm_to_quat(self):
        """Identity DCM should give identity quaternion."""
        C = np.eye(3)
        q = dcm_to_quat(C)

        np.testing.assert_array_almost_equal(q, quat_identity())

    def test_quat_dcm_roundtrip(self):
        """quat -> DCM -> quat should preserve rotation."""
        q_orig = quat_from_axis_angle(np.array([1, 1, 1]) / np.sqrt(3), np.pi / 3)

        C = quat_to_dcm(q_orig)
        q_back = dcm_to_quat(C)

        # May differ by sign (q and -q are same rotation)
        if q_back[0] * q_orig[0] < 0:
            q_back = -q_back

        np.testing.assert_array_almost_equal(q_back, q_orig)

    def test_dcm_quat_roundtrip(self):
        """DCM -> quat -> DCM should preserve rotation."""
        C_orig = rotz(np.pi / 4) @ roty(np.pi / 6) @ rotx(np.pi / 3)

        q = dcm_to_quat(C_orig)
        C_back = quat_to_dcm(q)

        np.testing.assert_array_almost_equal(C_back, C_orig)

    def test_90_deg_rotations(self):
        """Test 90° rotations about each axis."""
        test_cases = [
            (np.array([1, 0, 0]), np.pi / 2, rotx(np.pi / 2)),
            (np.array([0, 1, 0]), np.pi / 2, roty(np.pi / 2)),
            (np.array([0, 0, 1]), np.pi / 2, rotz(np.pi / 2)),
        ]

        for axis, angle, expected_dcm in test_cases:
            q = quat_from_axis_angle(axis, angle)
            C = quat_to_dcm(q)
            np.testing.assert_array_almost_equal(C, expected_dcm)


class TestQuaternionRotation:
    """Tests for rotating vectors with quaternions."""

    def test_rotate_identity(self):
        """Identity rotation should preserve vector."""
        q = quat_identity()
        v = np.array([1.0, 2.0, 3.0])

        v_rot = quat_rotate(q, v)

        np.testing.assert_array_almost_equal(v_rot, v)

    def test_rotate_90_deg_z(self):
        """90° about z should map x to y."""
        q = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])

        v_rot = quat_rotate(q, v)

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(v_rot, expected)

    def test_rotate_180_deg(self):
        """180° about z should map x to -x."""
        q = quat_from_axis_angle(np.array([0, 0, 1]), np.pi)
        v = np.array([1.0, 0.0, 0.0])

        v_rot = quat_rotate(q, v)

        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(v_rot, expected)

    def test_rotate_inverse(self):
        """Rotate then rotate_inverse should return original."""
        q = quat_from_axis_angle(np.array([1, 1, 1]) / np.sqrt(3), np.pi / 4)
        v = np.array([1.0, 2.0, 3.0])

        v_rot = quat_rotate(q, v)
        v_back = quat_rotate_inverse(q, v_rot)

        np.testing.assert_array_almost_equal(v_back, v)

    def test_rotation_matches_dcm(self):
        """Quaternion rotation should match DCM rotation."""
        q = quat_from_axis_angle(np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3]), np.pi / 5)
        C = quat_to_dcm(q)
        v = np.array([1.0, 2.0, 3.0])

        v_quat = quat_rotate(q, v)
        v_dcm = C @ v

        np.testing.assert_array_almost_equal(v_quat, v_dcm)


# =============================================================================
# Test: Quaternion Kinematics
# =============================================================================


class TestQuaternionKinematics:
    """Tests for quaternion kinematics and integration."""

    def test_omega_matrix_shape(self):
        """Omega matrix should be 4x4."""
        omega = np.array([0.1, 0.2, 0.3])
        Omega = omega_matrix(omega)

        assert Omega.shape == (4, 4)

    def test_omega_matrix_skew_symmetric(self):
        """Omega matrix should be skew-symmetric."""
        omega = np.array([0.1, 0.2, 0.3])
        Omega = omega_matrix(omega)

        np.testing.assert_array_almost_equal(Omega, -Omega.T)

    def test_quat_derivative_zero_omega(self):
        """Zero angular velocity should give zero derivative."""
        q = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        omega = np.array([0.0, 0.0, 0.0])

        q_dot = quat_derivative(q, omega)

        np.testing.assert_array_almost_equal(q_dot, np.zeros(4))

    def test_quat_integrate_preserves_norm(self):
        """Integration should preserve unit norm."""
        q = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.01

        q_next = quat_integrate(q, omega, dt)

        np.testing.assert_almost_equal(np.linalg.norm(q_next), 1.0)

    def test_quat_integrate_exact_constant_omega(self):
        """Exact integration for constant angular velocity."""
        q0 = quat_identity()
        omega = np.array([0.0, 0.0, 1.0])  # Rotate about z
        dt = np.pi / 2  # Quarter turn

        q_next = quat_integrate_exact(q0, omega, dt)

        # Should be 90° about z
        expected = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)

        # May differ by sign
        if q_next[0] * expected[0] < 0:
            q_next = -q_next

        np.testing.assert_array_almost_equal(q_next, expected)

    def test_quat_integrate_full_rotation(self):
        """Full 360° rotation should return to start."""
        q0 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 6)
        omega = np.array([0.0, 0.0, 2 * np.pi])  # 1 rev/sec about z

        # Integrate for 1 second
        q = q0.copy()
        dt = 0.001
        for _ in range(1000):
            q = quat_integrate_exact(q, omega, dt)

        # Should return to original orientation
        C0 = quat_to_dcm(q0)
        C = quat_to_dcm(q)

        np.testing.assert_array_almost_equal(C, C0, decimal=2)


class TestQuaternionSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_endpoints(self):
        """SLERP at t=0 and t=1 should give endpoints."""
        q1 = quat_from_axis_angle(np.array([1, 0, 0]), 0)
        q2 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 2)

        result_0 = quat_slerp(q1, q2, 0.0)
        result_1 = quat_slerp(q1, q2, 1.0)

        np.testing.assert_array_almost_equal(result_0, q1)
        np.testing.assert_array_almost_equal(result_1, q2)

    def test_slerp_midpoint(self):
        """SLERP at t=0.5 should give midpoint rotation."""
        q1 = quat_identity()
        q2 = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)

        result = quat_slerp(q1, q2, 0.5)
        angle = quat_angle(result)

        np.testing.assert_almost_equal(angle, np.pi / 4)

    def test_slerp_preserves_unit_norm(self):
        """SLERP should always return unit quaternion."""
        q1 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 6)
        q2 = quat_from_axis_angle(np.array([0, 1, 0]), np.pi / 3)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = quat_slerp(q1, q2, t)
            np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)


class TestQuaternionError:
    """Tests for attitude error computation."""

    def test_error_identity(self):
        """Error between same quaternions should be identity."""
        q = quat_from_axis_angle(np.array([1, 1, 1]) / np.sqrt(3), np.pi / 4)

        q_err = quat_error(q, q)
        angle = quat_angle(q_err)

        np.testing.assert_almost_equal(angle, 0.0)

    def test_error_inverse(self):
        """Error from q to identity should be q."""
        q = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        identity = quat_identity()

        q_err = quat_error(q, identity)

        # May differ by sign
        if q_err[0] * q[0] < 0:
            q_err = -q_err

        np.testing.assert_array_almost_equal(q_err, q)


class TestQuaternionRandom:
    """Tests for random quaternion generation."""

    def test_random_is_unit(self):
        """Random quaternion should have unit norm."""
        for _ in range(10):
            q = quat_random()
            np.testing.assert_almost_equal(np.linalg.norm(q), 1.0)

    def test_random_reproducible(self):
        """Random with seed should be reproducible."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        q1 = quat_random(rng1)
        q2 = quat_random(rng2)

        np.testing.assert_array_equal(q1, q2)


# =============================================================================
# Test: Elementary Rotation Matrices
# =============================================================================


class TestElementaryRotations:
    """Tests for rotx, roty, rotz."""

    def test_rotx_zero(self):
        """Zero rotation should give identity."""
        np.testing.assert_array_almost_equal(rotx(0), np.eye(3))

    def test_roty_zero(self):
        np.testing.assert_array_almost_equal(roty(0), np.eye(3))

    def test_rotz_zero(self):
        np.testing.assert_array_almost_equal(rotz(0), np.eye(3))

    def test_rotx_90(self):
        """90° about x should map y to z."""
        R = rotx(np.pi / 2)
        v = np.array([0, 1, 0])
        result = R @ v

        np.testing.assert_array_almost_equal(result, np.array([0, 0, 1]))

    def test_roty_90(self):
        """90° about y should map z to x."""
        R = roty(np.pi / 2)
        v = np.array([0, 0, 1])
        result = R @ v

        np.testing.assert_array_almost_equal(result, np.array([1, 0, 0]))

    def test_rotz_90(self):
        """90° about z should map x to y."""
        R = rotz(np.pi / 2)
        v = np.array([1, 0, 0])
        result = R @ v

        np.testing.assert_array_almost_equal(result, np.array([0, 1, 0]))

    def test_rotation_orthogonal(self):
        """Rotation matrices should be orthogonal."""
        for angle in [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]:
            for R in [rotx(angle), roty(angle), rotz(angle)]:
                np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
                np.testing.assert_almost_equal(np.linalg.det(R), 1.0)


# =============================================================================
# Test: Skew-Symmetric Matrix
# =============================================================================


class TestSkewMatrix:
    """Tests for skew-symmetric matrix operations."""

    def test_skew_shape(self):
        """Skew matrix should be 3x3."""
        v = np.array([1, 2, 3])
        S = skew(v)
        assert S.shape == (3, 3)

    def test_skew_antisymmetric(self):
        """Skew matrix should be antisymmetric."""
        v = np.array([1, 2, 3])
        S = skew(v)
        np.testing.assert_array_almost_equal(S, -S.T)

    def test_skew_cross_product(self):
        """S @ u should equal v x u."""
        v = np.array([1, 2, 3])
        u = np.array([4, 5, 6])
        S = skew(v)

        result = S @ u
        expected = np.cross(v, u)

        np.testing.assert_array_almost_equal(result, expected)

    def test_unskew(self):
        """Unskew should recover original vector."""
        v = np.array([1, 2, 3])
        S = skew(v)
        v_back = unskew(S)

        np.testing.assert_array_almost_equal(v_back, v)


# =============================================================================
# Test: Euler Angles
# =============================================================================


class TestEulerAngles:
    """Tests for Euler angle conversions."""

    def test_euler_to_dcm_identity(self):
        """Zero Euler angles should give identity DCM."""
        C = euler_to_dcm(0, 0, 0)
        np.testing.assert_array_almost_equal(C, np.eye(3))

    def test_dcm_to_euler_identity(self):
        """Identity DCM should give zero Euler angles."""
        phi, theta, psi = dcm_to_euler(np.eye(3))
        np.testing.assert_almost_equal(phi, 0)
        np.testing.assert_almost_equal(theta, 0)
        np.testing.assert_almost_equal(psi, 0)

    def test_euler_dcm_roundtrip(self):
        """euler -> DCM -> euler should preserve angles."""
        phi, theta, psi = np.pi / 6, np.pi / 5, np.pi / 4

        C = euler_to_dcm(phi, theta, psi)
        phi_back, theta_back, psi_back = dcm_to_euler(C)

        np.testing.assert_almost_equal(phi_back, phi)
        np.testing.assert_almost_equal(theta_back, theta)
        np.testing.assert_almost_equal(psi_back, psi)

    def test_euler_to_quat_identity(self):
        """Zero Euler angles should give identity quaternion."""
        q = euler_to_quat(0, 0, 0)
        np.testing.assert_array_almost_equal(q, quat_identity())

    def test_quat_to_euler_identity(self):
        """Identity quaternion should give zero Euler angles."""
        phi, theta, psi = quat_to_euler(quat_identity())
        np.testing.assert_almost_equal(phi, 0)
        np.testing.assert_almost_equal(theta, 0)
        np.testing.assert_almost_equal(psi, 0)

    def test_euler_quat_roundtrip(self):
        """euler -> quat -> euler should preserve angles."""
        phi, theta, psi = np.pi / 6, np.pi / 5, np.pi / 4

        q = euler_to_quat(phi, theta, psi)
        phi_back, theta_back, psi_back = quat_to_euler(q)

        np.testing.assert_almost_equal(phi_back, phi)
        np.testing.assert_almost_equal(theta_back, theta)
        np.testing.assert_almost_equal(psi_back, psi)

    def test_euler_quat_dcm_consistency(self):
        """Euler -> quat -> DCM should match Euler -> DCM."""
        phi, theta, psi = np.pi / 6, np.pi / 5, np.pi / 4

        C_direct = euler_to_dcm(phi, theta, psi)
        q = euler_to_quat(phi, theta, psi)
        C_via_quat = quat_to_dcm(q)

        np.testing.assert_array_almost_equal(C_via_quat, C_direct)


class TestGimbalLock:
    """Tests for gimbal lock handling."""

    def test_gimbal_lock_positive_90(self):
        """Handle theta = +90° gimbal lock."""
        C = euler_to_dcm(0, np.pi / 2, 0)  # Pitch = 90°
        phi, theta, psi = dcm_to_euler(C)

        # Theta should be recovered
        np.testing.assert_almost_equal(theta, np.pi / 2)

        # phi + psi combination should be zero (they're coupled)
        C_back = euler_to_dcm(phi, theta, psi)
        np.testing.assert_array_almost_equal(C_back, C)

    def test_gimbal_lock_negative_90(self):
        """Handle theta = -90° gimbal lock."""
        C = euler_to_dcm(0, -np.pi / 2, 0)  # Pitch = -90°
        phi, theta, psi = dcm_to_euler(C)

        np.testing.assert_almost_equal(theta, -np.pi / 2)

        C_back = euler_to_dcm(phi, theta, psi)
        np.testing.assert_array_almost_equal(C_back, C)


# =============================================================================
# Test: DCM Utilities
# =============================================================================


class TestDCMUtilities:
    """Tests for DCM utility functions."""

    def test_dcm_is_valid_identity(self):
        """Identity should be valid."""
        assert dcm_is_valid(np.eye(3))

    def test_dcm_is_valid_rotation(self):
        """Rotation matrix should be valid."""
        C = rotz(np.pi / 4) @ roty(np.pi / 6)
        assert dcm_is_valid(C)

    def test_dcm_is_valid_not_orthogonal(self):
        """Non-orthogonal matrix should be invalid."""
        C = np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
        assert not dcm_is_valid(C)

    def test_dcm_is_valid_reflection(self):
        """Reflection (det=-1) should be invalid."""
        C = np.diag([1, 1, -1])
        assert not dcm_is_valid(C)

    def test_dcm_normalize(self):
        """Normalize should produce valid DCM."""
        C_noisy = np.eye(3) + 0.01 * np.random.randn(3, 3)
        C_clean = dcm_normalize(C_noisy)

        assert dcm_is_valid(C_clean)

    def test_dcm_from_axis_angle_identity(self):
        """Zero angle should give identity."""
        C = dcm_from_axis_angle(np.array([1, 0, 0]), 0)
        np.testing.assert_array_almost_equal(C, np.eye(3))

    def test_dcm_from_axis_angle_90_z(self):
        """90° about z from axis-angle."""
        C = dcm_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        np.testing.assert_array_almost_equal(C, rotz(np.pi / 2))

    def test_dcm_angle_extraction(self):
        """Extract rotation angle from DCM."""
        angle = np.pi / 5
        C = dcm_from_axis_angle(np.array([1, 0, 0]), angle)
        extracted = dcm_angle(C)

        np.testing.assert_almost_equal(extracted, angle)

    def test_dcm_axis_extraction(self):
        """Extract rotation axis from DCM."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        C = dcm_from_axis_angle(axis, np.pi / 4)
        extracted = dcm_axis(C)

        np.testing.assert_array_almost_equal(extracted, axis)


# =============================================================================
# Test: Angular Velocity Conversions
# =============================================================================


class TestAngularVelocity:
    """Tests for angular velocity conversions."""

    def test_omega_to_euler_rates_zero(self):
        """Zero omega should give zero rates."""
        omega = np.array([0, 0, 0])
        rates = angular_velocity_to_euler_rates(omega, 0, 0)

        np.testing.assert_array_almost_equal(rates, np.zeros(3))

    def test_euler_rates_to_omega_zero(self):
        """Zero rates should give zero omega."""
        rates = np.array([0, 0, 0])
        omega = euler_rates_to_angular_velocity(rates, 0, 0)

        np.testing.assert_array_almost_equal(omega, np.zeros(3))

    def test_omega_euler_rates_roundtrip(self):
        """omega -> rates -> omega should preserve."""
        omega = np.array([0.1, 0.2, 0.3])
        phi, theta = 0.1, 0.2

        rates = angular_velocity_to_euler_rates(omega, phi, theta)
        omega_back = euler_rates_to_angular_velocity(rates, phi, theta)

        np.testing.assert_array_almost_equal(omega_back, omega)

    def test_gimbal_lock_raises(self):
        """Should raise at gimbal lock."""
        omega = np.array([0.1, 0.2, 0.3])

        with pytest.raises(ValueError):
            angular_velocity_to_euler_rates(omega, 0, np.pi / 2)


# =============================================================================
# Test: Angle Utilities
# =============================================================================


class TestAngleUtilities:
    """Tests for angle wrapping utilities."""

    def test_wrap_angle_in_range(self):
        """Angle in range should be unchanged."""
        np.testing.assert_almost_equal(wrap_angle(0), 0)
        np.testing.assert_almost_equal(wrap_angle(np.pi / 2), np.pi / 2)
        np.testing.assert_almost_equal(wrap_angle(-np.pi / 2), -np.pi / 2)

    def test_wrap_angle_overflow(self):
        """Angles outside [-π, π] should be wrapped."""
        np.testing.assert_almost_equal(wrap_angle(2 * np.pi), 0)
        np.testing.assert_almost_equal(wrap_angle(3 * np.pi), -np.pi)
        np.testing.assert_almost_equal(wrap_angle(-3 * np.pi), -np.pi)

    def test_wrap_angle_positive(self):
        """wrap_angle_positive should give [0, 2π]."""
        np.testing.assert_almost_equal(wrap_angle_positive(0), 0)
        np.testing.assert_almost_equal(wrap_angle_positive(-np.pi), np.pi)
        np.testing.assert_almost_equal(wrap_angle_positive(3 * np.pi), np.pi)

    def test_angle_difference(self):
        """Shortest path angle difference."""
        np.testing.assert_almost_equal(angle_difference(0, 0), 0)
        # π and -π are equivalent, so check absolute value
        np.testing.assert_almost_equal(np.abs(angle_difference(np.pi, 0)), np.pi)
        np.testing.assert_almost_equal(np.abs(angle_difference(0, np.pi)), np.pi)

        # Should take shortest path
        np.testing.assert_almost_equal(angle_difference(0.1, 2 * np.pi - 0.1), 0.2)


# =============================================================================
# Test: Consistency Across Representations
# =============================================================================


class TestRepresentationConsistency:
    """Tests for consistency between different rotation representations."""

    def test_all_representations_same_rotation(self):
        """All representations should give same rotation."""
        # Define rotation via axis-angle
        axis = np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
        angle = np.pi / 5

        # Get all representations
        q = quat_from_axis_angle(axis, angle)
        C_from_quat = quat_to_dcm(q)
        C_from_axis = dcm_from_axis_angle(axis, angle)

        # Convert to Euler and back
        phi, theta, psi = quat_to_euler(q)
        C_from_euler = euler_to_dcm(phi, theta, psi)

        # All DCMs should match
        np.testing.assert_array_almost_equal(C_from_quat, C_from_axis)
        np.testing.assert_array_almost_equal(C_from_euler, C_from_axis)

        # Rotating a vector should give same result
        v = np.array([1, 0, 0])

        v_quat = quat_rotate(q, v)
        v_dcm = C_from_axis @ v

        np.testing.assert_array_almost_equal(v_quat, v_dcm)

    def test_composition_consistency(self):
        """Rotation composition should be consistent."""
        q1 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        q2 = quat_from_axis_angle(np.array([0, 1, 0]), np.pi / 3)

        C1 = quat_to_dcm(q1)
        C2 = quat_to_dcm(q2)

        # Compose quaternions
        q_composed = quat_multiply(q1, q2)
        C_from_quat = quat_to_dcm(q_composed)

        # Compose DCMs
        C_composed = C1 @ C2

        np.testing.assert_array_almost_equal(C_from_quat, C_composed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
