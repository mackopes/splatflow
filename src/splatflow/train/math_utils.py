"""Math utilities for 3D transformations."""

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """Create 3x3 rotation matrix around X-axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Create 3x3 rotation matrix around Y-axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Create 3x3 rotation matrix around Z-axis.

    Args:
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
