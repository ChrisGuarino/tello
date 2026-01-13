"""
Motion Estimation for Visual SLAM

Estimates camera motion between frames using epipolar geometry
(Essential matrix) and PnP for pose refinement with 3D points.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class MotionEstimator:
    """
    Estimates camera motion from 2D-2D and 3D-2D correspondences.

    For monocular SLAM:
    - Essential matrix: Initial pose from 2D-2D matches (up to scale)
    - PnP: Pose from 3D map points to 2D observations (absolute pose)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        ransac_threshold: float = 1.0,
        ransac_prob: float = 0.999
    ):
        """
        Initialize the motion estimator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
            ransac_threshold: RANSAC reprojection threshold in pixels
            ransac_prob: RANSAC confidence probability
        """
        self.K = camera_matrix
        self.ransac_threshold = ransac_threshold
        self.ransac_prob = ransac_prob

    def estimate_essential_matrix(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Compute Essential matrix from point correspondences.

        Args:
            pts1: Nx2 array of points in first image
            pts2: Nx2 array of corresponding points in second image

        Returns:
            Tuple of:
            - E: 3x3 Essential matrix (or None if failed)
            - inliers: Nx1 boolean mask of inlier correspondences
        """
        if len(pts1) < 5:
            return None, np.zeros(len(pts1), dtype=bool)

        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold
        )

        if E is None or mask is None:
            return None, np.zeros(len(pts1), dtype=bool)

        # Handle case where multiple Es are returned
        if E.shape[0] > 3:
            E = E[:3, :]

        inliers = mask.flatten().astype(bool)
        return E, inliers

    def decompose_essential(
        self,
        E: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover rotation and translation from Essential matrix.

        The translation is only up to scale (unit vector).

        Args:
            E: 3x3 Essential matrix
            pts1: Nx2 points in first image
            pts2: Nx2 corresponding points in second image

        Returns:
            Tuple of:
            - R: 3x3 rotation matrix
            - t: 3x1 translation vector (unit length)
            - mask: Nx1 mask of points in front of both cameras
        """
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask

    def estimate_pose_essential(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int]:
        """
        Estimate relative pose from 2D-2D correspondences.

        Combines Essential matrix estimation and decomposition.

        Args:
            pts1: Nx2 points in first (reference) frame
            pts2: Nx2 points in second (current) frame

        Returns:
            Tuple of:
            - R: 3x3 rotation matrix (or None if failed)
            - t: 3x1 translation (unit length, or None if failed)
            - inliers: Boolean mask of inliers
            - num_inliers: Number of inliers
        """
        E, inliers = self.estimate_essential_matrix(pts1, pts2)

        if E is None or np.sum(inliers) < 5:
            return None, None, inliers, 0

        # Use only inliers for pose recovery
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        R, t, _ = self.decompose_essential(E, pts1_in, pts2_in)

        return R, t, inliers, np.sum(inliers)

    def estimate_pose_pnp(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        use_extrinsic_guess: bool = False,
        rvec_init: Optional[np.ndarray] = None,
        tvec_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate pose from 3D-2D correspondences using PnP.

        This provides absolute pose once a map exists.

        Args:
            pts_3d: Nx3 array of 3D map points
            pts_2d: Nx2 array of 2D image observations
            use_extrinsic_guess: Use initial guess for refinement
            rvec_init: Initial rotation vector guess
            tvec_init: Initial translation vector guess

        Returns:
            Tuple of:
            - success: Whether PnP succeeded
            - R: 3x3 rotation matrix
            - t: 3x1 translation vector
            - inliers: Indices of inlier correspondences
        """
        if len(pts_3d) < 4:
            return False, np.eye(3), np.zeros((3, 1)), np.array([])

        pts_3d = pts_3d.astype(np.float64)
        pts_2d = pts_2d.astype(np.float64)

        if use_extrinsic_guess and rvec_init is not None and tvec_init is not None:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d,
                self.K,
                distCoeffs=None,
                rvec=rvec_init.copy(),
                tvec=tvec_init.copy(),
                useExtrinsicGuess=True,
                iterationsCount=100,
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_prob,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d,
                self.K,
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_prob,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success or inliers is None:
            return False, np.eye(3), np.zeros((3, 1)), np.array([])

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        return True, R, tvec, inliers.flatten()

    def compute_reprojection_error(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection error for 3D points.

        Args:
            pts_3d: Nx3 array of 3D points
            pts_2d: Nx2 array of observed 2D points
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            Nx1 array of reprojection errors (Euclidean distance)
        """
        # Project 3D points
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(
            pts_3d, rvec, t, self.K, distCoeffs=None
        )
        projected = projected.reshape(-1, 2)

        # Compute errors
        errors = np.linalg.norm(pts_2d - projected, axis=1)
        return errors


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from two views.

    Args:
        pts1: Nx2 points in first image
        pts2: Nx2 points in second image
        P1: 3x4 projection matrix for first camera
        P2: 3x4 projection matrix for second camera

    Returns:
        Nx3 array of triangulated 3D points
    """
    if len(pts1) == 0:
        return np.zeros((0, 3))

    pts1 = pts1.T.astype(np.float64)  # 2xN
    pts2 = pts2.T.astype(np.float64)  # 2xN

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous
    points_3d = points_4d[:3, :] / points_4d[3:4, :]

    return points_3d.T  # Nx3


def compute_parallax(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> float:
    """
    Compute median parallax angle between two views.

    Useful for determining if there's enough baseline for triangulation.

    Args:
        pts1: Nx2 points in first image
        pts2: Nx2 points in second image
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation between views
        t: 3x1 translation between views

    Returns:
        Median parallax angle in degrees
    """
    if len(pts1) == 0:
        return 0.0

    # Convert to normalized camera coordinates
    K_inv = np.linalg.inv(K)

    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    rays1 = (K_inv @ pts1_h.T).T  # Nx3
    rays2 = (K_inv @ pts2_h.T).T  # Nx3

    # Transform rays2 to first camera frame
    rays2_transformed = (R.T @ rays2.T).T

    # Normalize rays
    rays1 = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
    rays2_transformed = rays2_transformed / np.linalg.norm(
        rays2_transformed, axis=1, keepdims=True
    )

    # Compute angles
    cos_angles = np.sum(rays1 * rays2_transformed, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return np.degrees(np.median(angles))


def make_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create a 3x4 projection matrix from intrinsics and extrinsics.

    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        3x4 projection matrix P = K @ [R | t]
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt
