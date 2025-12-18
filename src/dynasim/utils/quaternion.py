"""
Quaternion utilities for attitude representation.

This module provides quaternion operations following the scalar-first convention:
    q = [q_w, q_x, q_y, q_z] = [cos(θ/2), sin(θ/2)·n]

where θ is the rotation angle and n is the unit rotation axis.

Convention Notes
----------------
- Scalar-first ordering: q = [w, x, y, z]
- Hamilton product convention
- Unit quaternions represent rotations in SO(3)
- q and -q represent the same rotation

References
----------
- Szmuk et al. (2018) - Successive Convexification for 6-DoF Powered Descent
- Diebel (2006) - Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors
"""


import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication (Hamilton product).

    Computes q1 ⊗ q2, representing the composition of rotations:
    first rotate by q2, then rotate by q1.

    Parameters
    ----------
    q1 : np.ndarray, shape (4,)
        First quaternion [w, x, y, z].
    q2 : np.ndarray, shape (4,)
        Second quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (4,)
        Product quaternion q1 ⊗ q2.

    Examples
    --------
    >>> q1 = np.array([1, 0, 0, 0])  # Identity
    >>> q2 = np.array([0.707, 0.707, 0, 0])  # 90° about x
    >>> quat_multiply(q1, q2)
    array([0.707, 0.707, 0.   , 0.   ])

    Notes
    -----
    The Hamilton product is:
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate.

    For a unit quaternion, the conjugate equals the inverse.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (4,)
        Conjugate quaternion [w, -x, -y, -z].

    Examples
    --------
    >>> q = np.array([0.707, 0.707, 0, 0])
    >>> quat_conjugate(q)
    array([ 0.707, -0.707,  0.   ,  0.   ])
    """
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    Quaternion inverse.

    For unit quaternions, this equals the conjugate.
    For non-unit quaternions: q^(-1) = q* / ||q||²

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (4,)
        Inverse quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    norm_sq = np.dot(q, q)
    return quat_conjugate(q) / norm_sq


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion.

    Examples
    --------
    >>> q = np.array([1, 1, 0, 0])
    >>> q_unit = quat_normalize(q)
    >>> np.linalg.norm(q_unit)
    1.0
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Direction Cosine Matrix (rotation matrix).

    The DCM C transforms vectors from body frame to inertial frame:
        v_I = C @ v_B

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix (DCM).

    Examples
    --------
    >>> q = np.array([1, 0, 0, 0])  # Identity
    >>> quat_to_dcm(q)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Notes
    -----
    The rotation matrix is computed as:
        C = (w² - ||v||²)I + 2vvᵀ + 2w·skew(v)
    where v = [x, y, z] is the vector part.
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)

    w, x, y, z = q

    # Precompute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    C = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )

    return C


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """
    Convert Direction Cosine Matrix to quaternion.

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    C : np.ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z] with w >= 0.

    Examples
    --------
    >>> C = np.eye(3)
    >>> dcm_to_quat(C)
    array([1., 0., 0., 0.])
    """
    C = np.asarray(C, dtype=np.float64)

    # Shepperd's method - choose largest diagonal for numerical stability
    trace = np.trace(C)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (C[2, 1] - C[1, 2]) * s
        y = (C[0, 2] - C[2, 0]) * s
        z = (C[1, 0] - C[0, 1]) * s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        w = (C[2, 1] - C[1, 2]) / s
        x = 0.25 * s
        y = (C[0, 1] + C[1, 0]) / s
        z = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        w = (C[0, 2] - C[2, 0]) / s
        x = (C[0, 1] + C[1, 0]) / s
        y = 0.25 * s
        z = (C[1, 2] + C[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        w = (C[1, 0] - C[0, 1]) / s
        x = (C[0, 2] + C[2, 0]) / s
        y = (C[1, 2] + C[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])

    # Ensure w >= 0 (canonical form)
    if q[0] < 0:
        q = -q

    return quat_normalize(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Computes v' = q ⊗ v ⊗ q*, where v is treated as a pure quaternion [0, v].
    This rotates v from body frame to inertial frame.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    v : np.ndarray, shape (3,)
        Vector to rotate.

    Returns
    -------
    np.ndarray, shape (3,)
        Rotated vector.

    Examples
    --------
    >>> q = np.array([0.707, 0.707, 0, 0])  # 90° about x
    >>> v = np.array([0, 1, 0])
    >>> quat_rotate(q, v)  # Should give approximately [0, 0, 1]
    array([0., 0., 1.])
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Method: use rotation matrix (more efficient for single vectors)
    C = quat_to_dcm(q)
    return C @ v


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by the inverse of a quaternion.

    Computes v' = q* ⊗ v ⊗ q.
    This rotates v from inertial frame to body frame.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    v : np.ndarray, shape (3,)
        Vector to rotate.

    Returns
    -------
    np.ndarray, shape (3,)
        Rotated vector.
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    C = quat_to_dcm(q)
    return C.T @ v


def quat_error(q_desired: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """
    Compute attitude error quaternion.

    Returns q_error such that q_desired = q_error ⊗ q_actual.
    Thus q_error represents the rotation from actual to desired attitude.

    Parameters
    ----------
    q_desired : np.ndarray, shape (4,)
        Desired attitude quaternion.
    q_actual : np.ndarray, shape (4,)
        Actual attitude quaternion.

    Returns
    -------
    np.ndarray, shape (4,)
        Error quaternion.

    Examples
    --------
    >>> q1 = np.array([1, 0, 0, 0])  # Identity
    >>> q2 = np.array([0.707, 0.707, 0, 0])  # 90° about x
    >>> q_err = quat_error(q1, q2)  # Error to go from q2 to q1
    """
    q_desired = np.asarray(q_desired, dtype=np.float64)
    q_actual = np.asarray(q_actual, dtype=np.float64)

    # q_error = q_desired ⊗ q_actual^(-1)
    q_error = quat_multiply(q_desired, quat_conjugate(q_actual))

    # Ensure shortest path (w >= 0)
    if q_error[0] < 0:
        q_error = -q_error

    return q_error


def quat_angle(q: np.ndarray) -> float:
    """
    Extract the rotation angle from a quaternion.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    float
        Rotation angle in radians [0, π].

    Examples
    --------
    >>> q = np.array([0.707, 0.707, 0, 0])  # 90° about x
    >>> np.degrees(quat_angle(q))
    90.0
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)

    # Ensure w is in valid range for arccos
    w = np.clip(q[0], -1.0, 1.0)

    # θ = 2 * arccos(|w|)
    return 2.0 * np.arccos(np.abs(w))


def quat_axis(q: np.ndarray) -> np.ndarray:
    """
    Extract the rotation axis from a quaternion.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray, shape (3,)
        Unit rotation axis. Returns [1, 0, 0] for identity quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)

    # Vector part
    v = q[1:4]
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-12:
        return np.array([1.0, 0.0, 0.0])  # Arbitrary axis for zero rotation

    return v / norm_v


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a quaternion from axis-angle representation.

    Parameters
    ----------
    axis : np.ndarray, shape (3,)
        Rotation axis (will be normalized).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].

    Examples
    --------
    >>> axis = np.array([1, 0, 0])
    >>> angle = np.pi / 2  # 90 degrees
    >>> quat_from_axis_angle(axis, angle)
    array([0.707, 0.707, 0.   , 0.   ])
    """
    axis = np.asarray(axis, dtype=np.float64)

    # Normalize axis
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm

    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Construct the Ω(ω) matrix for quaternion kinematics.

    The quaternion derivative is: q̇ = (1/2) Ω(ω) q

    Parameters
    ----------
    omega : np.ndarray, shape (3,)
        Angular velocity vector [ωx, ωy, ωz] in body frame.

    Returns
    -------
    np.ndarray, shape (4, 4)
        The Ω matrix.

    Notes
    -----
    The matrix has the form:
        Ω = [ 0   -ωx  -ωy  -ωz]
            [ωx    0   ωz  -ωy]
            [ωy  -ωz    0   ωx]
            [ωz   ωy  -ωx    0]
    """
    omega = np.asarray(omega, dtype=np.float64)
    wx, wy, wz = omega

    return np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute quaternion derivative given angular velocity.

    q̇ = (1/2) Ω(ω) q

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Current quaternion [w, x, y, z].
    omega : np.ndarray, shape (3,)
        Angular velocity in body frame [ωx, ωy, ωz].

    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion derivative q̇.

    Examples
    --------
    >>> q = np.array([1, 0, 0, 0])  # Identity
    >>> omega = np.array([0.1, 0, 0])  # Rotating about x
    >>> quat_derivative(q, omega)
    array([0.  , 0.05, 0.  , 0.  ])
    """
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    Omega = omega_matrix(omega)
    return 0.5 * Omega @ q


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float, normalize: bool = True) -> np.ndarray:
    """
    Integrate quaternion given constant angular velocity.

    Uses first-order integration: q_{k+1} = q_k + dt * q̇_k
    with optional normalization to maintain unit length.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Current quaternion [w, x, y, z].
    omega : np.ndarray, shape (3,)
        Angular velocity in body frame.
    dt : float
        Time step.
    normalize : bool, optional
        Whether to normalize the result. Default True.

    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion at next time step.

    Notes
    -----
    For better accuracy with large angular velocities, consider using
    the exponential map method in quat_integrate_exact.
    """
    q = np.asarray(q, dtype=np.float64)

    q_dot = quat_derivative(q, omega)
    q_next = q + dt * q_dot

    if normalize:
        q_next = quat_normalize(q_next)

    return q_next


def quat_integrate_exact(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Exact quaternion integration for constant angular velocity.

    Uses the exponential map: q_{k+1} = q_Δ ⊗ q_k
    where q_Δ = [cos(|ω|dt/2), sin(|ω|dt/2) * ω/|ω|]

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Current quaternion [w, x, y, z].
    omega : np.ndarray, shape (3,)
        Angular velocity in body frame (constant over dt).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion at next time step.
    """
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    omega_norm = np.linalg.norm(omega)

    if omega_norm < 1e-12:
        return q.copy()

    # Incremental rotation quaternion
    angle = omega_norm * dt
    axis = omega / omega_norm
    q_delta = quat_from_axis_angle(axis, angle)

    # Apply rotation
    return quat_multiply(q_delta, q)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) between two quaternions.

    Parameters
    ----------
    q1 : np.ndarray, shape (4,)
        Starting quaternion.
    q2 : np.ndarray, shape (4,)
        Ending quaternion.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    np.ndarray, shape (4,)
        Interpolated quaternion.

    Examples
    --------
    >>> q1 = np.array([1, 0, 0, 0])
    >>> q2 = np.array([0, 1, 0, 0])  # 180° about x
    >>> quat_slerp(q1, q2, 0.5)  # Should be 90° about x
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative dot, negate one quaternion (shorter path)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    # SLERP formula
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return s1 * q1 + s2 * q2


def quat_identity() -> np.ndarray:
    """
    Return the identity quaternion (no rotation).

    Returns
    -------
    np.ndarray, shape (4,)
        Identity quaternion [1, 0, 0, 0].
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_random(rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a random unit quaternion (uniformly distributed on SO(3)).

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    np.ndarray, shape (4,)
        Random unit quaternion.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use the subgroup algorithm for uniform distribution
    u1, u2, u3 = rng.random(3)

    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )

    # Reorder to scalar-first
    return np.array([q[3], q[0], q[1], q[2]])
