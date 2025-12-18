"""
Rotation utilities for attitude representation.

This module provides functions for working with rotation matrices (DCM)
and Euler angles, as well as conversions between different representations.

Conventions
-----------
- Euler angles are in radians
- Default Euler sequence is 'ZYX' (yaw-pitch-roll, aerospace convention)
- DCM transforms vectors from body frame to inertial frame: v_I = C @ v_B
- Right-hand rotation convention

Euler Angle Sequences
---------------------
The sequence 'ZYX' means: first rotate about Z, then Y, then X.
This is also known as yaw-pitch-roll or 3-2-1 sequence.

Common sequences:
- 'ZYX': Aerospace/航空 (yaw, pitch, roll)
- 'XYZ': Roll, pitch, yaw
- 'ZXZ': Classical Euler angles

References
----------
- Diebel (2006) - Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors
- Stevens & Lewis - Aircraft Control and Simulation
"""

from typing import Tuple

import numpy as np

from simdyn.utils.quaternion import (
    dcm_to_quat,
    quat_normalize,
    quat_to_dcm,
)

# =============================================================================
# Basic Rotation Matrices
# =============================================================================


def rotx(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix.

    Examples
    --------
    >>> R = rotx(np.pi / 2)  # 90 degrees about x
    >>> v = np.array([0, 1, 0])
    >>> R @ v  # Should give [0, 0, 1]
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(angle: float) -> np.ndarray:
    """
    Elementary rotation matrix about the z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# =============================================================================
# Skew-Symmetric Matrix
# =============================================================================


def skew(v: np.ndarray) -> np.ndarray:
    """
    Construct a skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [v]x satisfies: [v]x @ u = v x u (cross product).

    Parameters
    ----------
    v : np.ndarray, shape (3,)
        Input vector [x, y, z].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Skew-symmetric matrix.

    Examples
    --------
    >>> v = np.array([1, 2, 3])
    >>> S = skew(v)
    >>> u = np.array([4, 5, 6])
    >>> np.allclose(S @ u, np.cross(v, u))
    True

    Notes
    -----
    The matrix has the form:
        [ 0  -z   y]
        [ z   0  -x]
        [-y   x   0]
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def unskew(S: np.ndarray) -> np.ndarray:
    """
    Extract the vector from a skew-symmetric matrix.

    Inverse operation of skew().

    Parameters
    ----------
    S : np.ndarray, shape (3, 3)
        Skew-symmetric matrix.

    Returns
    -------
    np.ndarray, shape (3,)
        Vector [x, y, z].
    """
    S = np.asarray(S, dtype=np.float64)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# =============================================================================
# Euler Angles to DCM
# =============================================================================


def euler_to_dcm(phi: float, theta: float, psi: float, sequence: str = "ZYX") -> np.ndarray:
    """
    Convert Euler angles to Direction Cosine Matrix.

    Parameters
    ----------
    phi : float
        Roll angle in radians (rotation about x-axis).
    theta : float
        Pitch angle in radians (rotation about y-axis).
    psi : float
        Yaw angle in radians (rotation about z-axis).
    sequence : str, optional
        Euler angle sequence. Default 'ZYX' (yaw-pitch-roll).
        Options: 'ZYX', 'XYZ'

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix (DCM).

    Examples
    --------
    >>> # 90° yaw (about z)
    >>> C = euler_to_dcm(0, 0, np.pi/2, sequence='ZYX')

    Notes
    -----
    For sequence 'ZYX' (aerospace convention):
        - psi: yaw (rotation about z-axis)
        - theta: pitch (rotation about y-axis)
        - phi: roll (rotation about x-axis)
        - C = Rz(psi) @ Ry(theta) @ Rx(phi)

    The resulting DCM transforms vectors from body to inertial frame.
    """
    sequence = sequence.upper()

    if sequence == "ZYX":
        # Aerospace convention: yaw-pitch-roll
        # C = Rz(psi) @ Ry(theta) @ Rx(phi)
        return rotz(psi) @ roty(theta) @ rotx(phi)

    elif sequence == "XYZ":
        # Roll-pitch-yaw
        # C = Rx(phi) @ Ry(theta) @ Rz(psi)
        return rotx(phi) @ roty(theta) @ rotz(psi)

    else:
        raise NotImplementedError(f"Sequence '{sequence}' not implemented. Use 'ZYX' or 'XYZ'.")


def dcm_to_euler(C: np.ndarray, sequence: str = "ZYX") -> Tuple[float, float, float]:
    """
    Convert Direction Cosine Matrix to Euler angles.

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Rotation matrix.
    sequence : str, optional
        Euler angle sequence. Default 'ZYX'.

    Returns
    -------
    phi : float
        First rotation angle in radians.
    theta : float
        Second rotation angle in radians.
    psi : float
        Third rotation angle in radians.

    Examples
    --------
    >>> C = np.eye(3)
    >>> phi, theta, psi = dcm_to_euler(C)
    >>> print(phi, theta, psi)
    0.0 0.0 0.0

    Notes
    -----
    Gimbal lock can occur when theta = ±90° for 'ZYX' sequence.
    In this case, phi and psi are not uniquely determined.

    Warnings
    --------
    Only 'ZYX' and 'XYZ' sequences are currently implemented.
    """
    C = np.asarray(C, dtype=np.float64)
    sequence = sequence.upper()

    if sequence == "ZYX":
        # Aerospace convention (yaw-pitch-roll)
        # C = Rx(phi) @ Ry(theta) @ Rz(psi)

        # Check for gimbal lock (theta = ±90°)
        if np.abs(C[2, 0]) >= 1.0 - 1e-10:
            # Gimbal lock
            psi = 0.0  # Set yaw to zero (arbitrary)
            if C[2, 0] < 0:  # theta = +90°
                theta = np.pi / 2
                phi = np.arctan2(C[0, 1], C[0, 2])
            else:  # theta = -90°
                theta = -np.pi / 2
                phi = np.arctan2(-C[0, 1], -C[0, 2])
        else:
            theta = np.arcsin(-C[2, 0])
            phi = np.arctan2(C[2, 1], C[2, 2])
            psi = np.arctan2(C[1, 0], C[0, 0])

        return phi, theta, psi

    elif sequence == "XYZ":
        # Roll-pitch-yaw
        # C = Rz(psi) @ Ry(theta) @ Rx(phi)

        if np.abs(C[0, 2]) >= 1.0 - 1e-10:
            # Gimbal lock
            phi = 0.0
            if C[0, 2] > 0:  # theta = +90°
                theta = np.pi / 2
                psi = np.arctan2(C[1, 0], C[1, 1])
            else:  # theta = -90°
                theta = -np.pi / 2
                psi = np.arctan2(-C[1, 0], C[1, 1])
        else:
            theta = np.arcsin(C[0, 2])
            phi = np.arctan2(-C[1, 2], C[2, 2])
            psi = np.arctan2(-C[0, 1], C[0, 0])

        return phi, theta, psi

    else:
        raise NotImplementedError(f"Sequence '{sequence}' not implemented. Use 'ZYX' or 'XYZ'.")


# =============================================================================
# Euler Angles to/from Quaternion
# =============================================================================


def euler_to_quat(phi: float, theta: float, psi: float, sequence: str = "ZYX") -> np.ndarray:
    """
    Convert Euler angles to quaternion.

    Parameters
    ----------
    phi : float
        First rotation angle in radians.
    theta : float
        Second rotation angle in radians.
    psi : float
        Third rotation angle in radians.
    sequence : str, optional
        Euler angle sequence. Default 'ZYX'.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Examples
    --------
    >>> q = euler_to_quat(0, 0, np.pi/2)  # 90° yaw
    >>> print(q)
    [0.707 0.    0.    0.707]
    """
    # Convert via DCM for generality
    # Could optimize for specific sequences if needed
    C = euler_to_dcm(phi, theta, psi, sequence)
    return dcm_to_quat(C)


def quat_to_euler(q: np.ndarray, sequence: str = "ZYX") -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    sequence : str, optional
        Euler angle sequence. Default 'ZYX'.

    Returns
    -------
    phi : float
        First rotation angle in radians.
    theta : float
        Second rotation angle in radians.
    psi : float
        Third rotation angle in radians.

    Examples
    --------
    >>> q = np.array([1, 0, 0, 0])  # Identity
    >>> phi, theta, psi = quat_to_euler(q)
    >>> print(phi, theta, psi)
    0.0 0.0 0.0
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)
    C = quat_to_dcm(q)
    return dcm_to_euler(C, sequence)


# =============================================================================
# DCM Utilities
# =============================================================================


def dcm_is_valid(C: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if a matrix is a valid rotation matrix.

    A valid rotation matrix satisfies:
    - C @ C.T = I (orthogonal)
    - det(C) = +1 (proper rotation, not reflection)

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Matrix to check.
    tol : float, optional
        Tolerance for numerical checks.

    Returns
    -------
    bool
        True if C is a valid rotation matrix.
    """
    C = np.asarray(C, dtype=np.float64)

    if C.shape != (3, 3):
        return False

    # Check orthogonality
    should_be_identity = C @ C.T
    if not np.allclose(should_be_identity, np.eye(3), atol=tol):
        return False

    # Check determinant
    det = np.linalg.det(C)
    return np.isclose(det, 1.0, atol=tol)


def dcm_normalize(C: np.ndarray) -> np.ndarray:
    """
    Normalize a rotation matrix to ensure orthogonality.

    Uses SVD to find the nearest orthogonal matrix.

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Rotation matrix (possibly with numerical drift).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Orthogonalized rotation matrix.
    """
    C = np.asarray(C, dtype=np.float64)
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt


def dcm_from_two_vectors(
    v1: np.ndarray, v2: np.ndarray, v1_body: np.ndarray = None, v2_body: np.ndarray = None
) -> np.ndarray:
    """
    Construct a DCM from two vectors using TRIAD or similar method.

    Given two non-parallel reference vectors and their body-frame measurements,
    compute the rotation matrix from body to reference frame.

    Parameters
    ----------
    v1 : np.ndarray, shape (3,)
        First reference vector (e.g., sun direction in inertial frame).
    v2 : np.ndarray, shape (3,)
        Second reference vector (e.g., magnetic field in inertial frame).
    v1_body : np.ndarray, shape (3,), optional
        First vector in body frame. If None, assumes [1, 0, 0].
    v2_body : np.ndarray, shape (3,), optional
        Second vector in body frame. If None, assumes [0, 1, 0].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix from body to reference frame.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    if v1_body is None:
        v1_body = np.array([1.0, 0.0, 0.0])
    if v2_body is None:
        v2_body = np.array([0.0, 1.0, 0.0])

    v1_body = np.asarray(v1_body, dtype=np.float64)
    v2_body = np.asarray(v2_body, dtype=np.float64)

    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v1_body = v1_body / np.linalg.norm(v1_body)
    v2_body = v2_body / np.linalg.norm(v2_body)

    # Build orthonormal triads
    # Reference frame triad
    t1_ref = v1
    t2_ref = np.cross(v1, v2)
    t2_ref = t2_ref / np.linalg.norm(t2_ref)
    t3_ref = np.cross(t1_ref, t2_ref)

    # Body frame triad
    t1_body = v1_body
    t2_body = np.cross(v1_body, v2_body)
    t2_body = t2_body / np.linalg.norm(t2_body)
    t3_body = np.cross(t1_body, t2_body)

    # Rotation matrix: v_ref = C @ v_body
    M_ref = np.column_stack([t1_ref, t2_ref, t3_ref])
    M_body = np.column_stack([t1_body, t2_body, t3_body])

    C = M_ref @ M_body.T

    return C


def dcm_angle(C: np.ndarray) -> float:
    """
    Extract the rotation angle from a DCM.

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    float
        Rotation angle in radians [0, π].
    """
    C = np.asarray(C, dtype=np.float64)

    # trace(C) = 1 + 2*cos(θ)
    trace = np.trace(C)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)


def dcm_axis(C: np.ndarray) -> np.ndarray:
    """
    Extract the rotation axis from a DCM.

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit rotation axis.
    """
    C = np.asarray(C, dtype=np.float64)

    angle = dcm_angle(C)

    if angle < 1e-10:
        # No rotation, arbitrary axis
        return np.array([1.0, 0.0, 0.0])

    if np.abs(angle - np.pi) < 1e-10:
        # 180° rotation - find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(C)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        axis = np.real(eigenvectors[:, idx])
        return axis / np.linalg.norm(axis)

    # General case: axis from skew-symmetric part
    axis = np.array([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]])

    return axis / np.linalg.norm(axis)


def dcm_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix from axis-angle representation.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    axis : np.ndarray, shape (3,)
        Unit rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix.

    Examples
    --------
    >>> axis = np.array([0, 0, 1])
    >>> angle = np.pi / 2
    >>> C = dcm_from_axis_angle(axis, angle)  # 90° about z
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    K = skew(axis)
    I = np.eye(3)

    # Rodrigues' formula: C = I + sin(θ)K + (1-cos(θ))K²
    C = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return C


# =============================================================================
# Angular Velocity Utilities
# =============================================================================


def angular_velocity_to_euler_rates(omega: np.ndarray, phi: float, theta: float, sequence: str = "ZYX") -> np.ndarray:
    """
    Convert body angular velocity to Euler angle rates.

    Parameters
    ----------
    omega : np.ndarray, shape (3,)
        Angular velocity in body frame [p, q, r].
    phi : float
        Current roll angle.
    theta : float
        Current pitch angle.
    sequence : str, optional
        Euler angle sequence. Default 'ZYX'.

    Returns
    -------
    np.ndarray, shape (3,)
        Euler angle rates [phi_dot, theta_dot, psi_dot].

    Notes
    -----
    For 'ZYX' sequence:
        [phi_dot  ]   [1  sin(phi)tan(theta)  cos(phi)tan(theta)] [p]
        [theta_dot] = [0  cos(phi)            -sin(phi)         ] [q]
        [psi_dot  ]   [0  sin(phi)sec(theta)  cos(phi)sec(theta)] [r]

    Warning: singular at theta = ±90° (gimbal lock).
    """
    omega = np.asarray(omega, dtype=np.float64)
    p, q, r = omega

    if sequence.upper() != "ZYX":
        raise NotImplementedError(f"Sequence '{sequence}' not implemented.")

    sp, cp = np.sin(phi), np.cos(phi)
    tt = np.tan(theta)
    ct = np.cos(theta)

    if np.abs(ct) < 1e-10:
        raise ValueError("Gimbal lock: theta near ±90°")

    st = 1.0 / ct  # sec(theta)

    phi_dot = p + sp * tt * q + cp * tt * r
    theta_dot = cp * q - sp * r
    psi_dot = sp * st * q + cp * st * r

    return np.array([phi_dot, theta_dot, psi_dot])


def euler_rates_to_angular_velocity(
    euler_rates: np.ndarray, phi: float, theta: float, sequence: str = "ZYX"
) -> np.ndarray:
    """
    Convert Euler angle rates to body angular velocity.

    Parameters
    ----------
    euler_rates : np.ndarray, shape (3,)
        Euler angle rates [phi_dot, theta_dot, psi_dot].
    phi : float
        Current roll angle.
    theta : float
        Current pitch angle.
    sequence : str, optional
        Euler angle sequence. Default 'ZYX'.

    Returns
    -------
    np.ndarray, shape (3,)
        Angular velocity in body frame [p, q, r].
    """
    euler_rates = np.asarray(euler_rates, dtype=np.float64)
    phi_dot, theta_dot, psi_dot = euler_rates

    if sequence.upper() != "ZYX":
        raise NotImplementedError(f"Sequence '{sequence}' not implemented.")

    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    p = phi_dot - st * psi_dot
    q = cp * theta_dot + sp * ct * psi_dot
    r = -sp * theta_dot + cp * ct * psi_dot

    return np.array([p, q, r])


# =============================================================================
# Angle Wrapping Utilities
# =============================================================================


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-π, π].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in [-π, π].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_angle_positive(angle: float) -> float:
    """
    Wrap angle to [0, 2π].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in [0, 2π].
    """
    return angle % (2 * np.pi)


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the shortest signed difference between two angles.

    Parameters
    ----------
    angle1 : float
        First angle in radians.
    angle2 : float
        Second angle in radians.

    Returns
    -------
    float
        Signed difference (angle1 - angle2) in [-π, π].
    """
    return wrap_angle(angle1 - angle2)
