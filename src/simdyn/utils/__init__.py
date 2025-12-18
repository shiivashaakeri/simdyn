"""
Utility functions for simdyn.

Modules
-------
quaternion : Quaternion operations (scalar-first convention)
rotations : Rotation utilities (DCM, Euler angles, conversions)
"""

from simdyn.utils.quaternion import (
    dcm_to_quat,
    # Kinematics
    omega_matrix,
    quat_angle,
    quat_axis,
    quat_conjugate,
    quat_derivative,
    # Error and interpolation
    quat_error,
    quat_from_axis_angle,
    # Utilities
    quat_identity,
    quat_integrate,
    quat_integrate_exact,
    quat_inverse,
    # Core operations
    quat_multiply,
    quat_normalize,
    quat_random,
    # Rotation
    quat_rotate,
    quat_rotate_inverse,
    quat_slerp,
    # Conversions
    quat_to_dcm,
)
from simdyn.utils.rotations import (
    angle_difference,
    # Angular velocity
    angular_velocity_to_euler_rates,
    dcm_angle,
    dcm_axis,
    dcm_from_axis_angle,
    dcm_from_two_vectors,
    # DCM utilities
    dcm_is_valid,
    dcm_normalize,
    dcm_to_euler,
    euler_rates_to_angular_velocity,
    # Euler angle conversions
    euler_to_dcm,
    euler_to_quat,
    quat_to_euler,
    # Elementary rotations
    rotx,
    roty,
    rotz,
    # Skew matrix
    skew,
    unskew,
    # Angle utilities
    wrap_angle,
    wrap_angle_positive,
)

__all__ = [
    'angle_difference',
    # Rotations - Angular velocity
    'angular_velocity_to_euler_rates',
    'dcm_angle',
    'dcm_axis',
    'dcm_from_axis_angle',
    'dcm_from_two_vectors',
    # Rotations - DCM utilities
    'dcm_is_valid',
    'dcm_normalize',
    'dcm_to_euler',
    'dcm_to_quat',
    'euler_rates_to_angular_velocity',
    # Rotations - Euler angles
    'euler_to_dcm',
    'euler_to_quat',
    # Quaternion - Kinematics
    'omega_matrix',
    'quat_angle',
    'quat_axis',
    'quat_conjugate',
    'quat_derivative',
    # Quaternion - Error and interpolation
    'quat_error',
    'quat_from_axis_angle',
    # Quaternion - Utilities
    'quat_identity',
    'quat_integrate',
    'quat_integrate_exact',
    'quat_inverse',
    # Quaternion - Core operations
    'quat_multiply',
    'quat_normalize',
    'quat_random',
    # Quaternion - Rotation
    'quat_rotate',
    'quat_rotate_inverse',
    'quat_slerp',
    # Quaternion - Conversions
    'quat_to_dcm',
    'quat_to_euler',
    # Rotations - Elementary
    'rotx',
    'roty',
    'rotz',
    # Rotations - Skew matrix
    'skew',
    'unskew',
    # Rotations - Angle utilities
    'wrap_angle',
    'wrap_angle_positive',
]
