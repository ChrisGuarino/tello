"""
Tello Camera Intrinsic Parameters

The Tello has a fixed lens with approximately 82.6 degree horizontal FOV.
This module provides the camera intrinsic matrix for SLAM computations.
"""

import numpy as np


class TelloCamera:
    """
    Camera intrinsic parameters for the DJI Tello drone.

    The Tello camera has:
    - Resolution: 720p (1280x720) native, typically used at 640x480
    - Horizontal FOV: ~82.6 degrees
    - No accessible IMU data
    - No lens distortion data (assumed minimal)
    """

    # Default resolution (matching existing code)
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    # Tello horizontal FOV in degrees
    FOV_HORIZONTAL = 82.6

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """
        Initialize camera parameters for given resolution.

        Args:
            width: Image width in pixels (default 640)
            height: Image height in pixels (default 480)
        """
        self.width = width
        self.height = height

        # Calculate focal length from FOV
        # FOV = 2 * atan(sensor_width / (2 * focal_length))
        # focal_length = sensor_width / (2 * tan(FOV/2))
        self.fx = width / (2 * np.tan(np.radians(self.FOV_HORIZONTAL / 2)))
        self.fy = self.fx  # Assume square pixels

        # Principal point at image center
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Build intrinsic matrix K
        self.K = np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,     1.0]
        ], dtype=np.float64)

        # Distortion coefficients (assumed zero for Tello)
        # Format: [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Return the 3x3 camera intrinsic matrix K."""
        return self.K.copy()

    def get_distortion_coeffs(self) -> np.ndarray:
        """Return distortion coefficients."""
        return self.dist_coeffs.copy()

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.

        Args:
            points_3d: Nx3 array of 3D points in camera frame

        Returns:
            Nx2 array of 2D pixel coordinates
        """
        points_3d = np.atleast_2d(points_3d)

        # Homogeneous division
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]

        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.column_stack([u, v])

    def unproject(self, points_2d: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Unproject 2D pixel coordinates to 3D rays.

        Args:
            points_2d: Nx2 array of pixel coordinates
            depth: Depth value to scale the rays (default 1.0 for unit rays)

        Returns:
            Nx3 array of 3D points
        """
        points_2d = np.atleast_2d(points_2d)

        # Remove intrinsics
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy
        z = np.ones(len(points_2d))

        rays = np.column_stack([x, y, z])

        # Normalize and scale by depth
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays / norms * depth

        return rays

    def __repr__(self) -> str:
        return (
            f"TelloCamera(width={self.width}, height={self.height}, "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f})"
        )


# Convenience function for quick access
def get_tello_camera(width: int = 640, height: int = 480) -> TelloCamera:
    """Get a TelloCamera instance with specified resolution."""
    return TelloCamera(width, height)
