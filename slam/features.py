"""
Feature Detection and Tracking for Visual SLAM

Provides ORB feature detection and Lucas-Kanade optical flow tracking
optimized for real-time performance on Tello video streams.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FeatureDetector:
    """
    Detects and describes features in images using ORB.

    ORB (Oriented FAST and Rotated BRIEF) is chosen for:
    - Speed: ~10x faster than SIFT
    - Patent-free: No licensing issues
    - Rotation invariant: Handles drone rotation well
    """

    def __init__(
        self,
        detector_type: str = 'ORB',
        max_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8
    ):
        """
        Initialize the feature detector.

        Args:
            detector_type: 'ORB', 'SIFT', or 'AKAZE'
            max_features: Maximum number of features to detect
            scale_factor: Pyramid scale factor (ORB only)
            n_levels: Number of pyramid levels (ORB only)
        """
        self.detector_type = detector_type
        self.max_features = max_features

        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=scale_factor,
                nlevels=n_levels,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=20
            )
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def detect(self, frame: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Detect keypoints in the frame.

        Args:
            frame: Grayscale image (H, W)

        Returns:
            List of cv2.KeyPoint objects
        """
        keypoints = self.detector.detect(frame, None)
        return keypoints

    def detect_and_compute(
        self,
        frame: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors.

        Args:
            frame: Grayscale image (H, W)

        Returns:
            Tuple of (keypoints, descriptors)
            descriptors is Nx32 for ORB, Nx128 for SIFT
        """
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        return keypoints, descriptors

    def compute(
        self,
        frame: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Compute descriptors for given keypoints.

        Args:
            frame: Grayscale image
            keypoints: List of keypoints to describe

        Returns:
            Tuple of (filtered keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.compute(frame, keypoints)
        return keypoints, descriptors


class FeatureTracker:
    """
    Tracks features between consecutive frames using optical flow.

    Uses Lucas-Kanade sparse optical flow for fast frame-to-frame tracking,
    with descriptor matching available for keyframe-to-keyframe.
    """

    def __init__(
        self,
        window_size: int = 21,
        max_level: int = 3,
        criteria_eps: float = 0.03,
        criteria_count: int = 30
    ):
        """
        Initialize the feature tracker.

        Args:
            window_size: Size of the search window for LK flow
            max_level: Maximum pyramid level (0 = single scale)
            criteria_eps: Termination epsilon for iterative search
            criteria_count: Maximum iterations
        """
        self.window_size = window_size
        self.max_level = max_level

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(window_size, window_size),
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                criteria_count,
                criteria_eps
            )
        )

        # Matcher for descriptor-based tracking
        # Use FLANN for SIFT (float descriptors)
        # Use BFMatcher with Hamming for ORB (binary descriptors)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def track_optical_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points using Lucas-Kanade optical flow.

        Args:
            prev_frame: Previous grayscale frame
            curr_frame: Current grayscale frame
            prev_points: Nx2 array of points to track

        Returns:
            Tuple of:
            - curr_points: Nx2 array of tracked point locations
            - status: Nx1 array (1 = tracked, 0 = lost)
            - error: Nx1 array of tracking errors
        """
        if len(prev_points) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 1), dtype=np.uint8),
                np.zeros((0, 1), dtype=np.float32)
            )

        # Ensure correct shape for OpenCV
        prev_points = prev_points.reshape(-1, 1, 2).astype(np.float32)

        # Forward tracking
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_points, None, **self.lk_params
        )

        # Backward tracking for validation (forward-backward check)
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_points, None, **self.lk_params
        )

        # Calculate forward-backward error
        fb_error = np.linalg.norm(
            prev_points.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1
        )

        # Mark as lost if FB error is too large (threshold: 1 pixel)
        fb_threshold = 1.0
        fb_valid = fb_error < fb_threshold

        # Combine status checks
        status = status.flatten() & back_status.flatten() & fb_valid

        return (
            curr_points.reshape(-1, 2),
            status.reshape(-1, 1).astype(np.uint8),
            error
        )

    def match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using Lowe's ratio test.

        Args:
            desc1: First set of descriptors (Nx32 or Nx128)
            desc2: Second set of descriptors
            ratio_threshold: Ratio test threshold (lower = stricter)

        Returns:
            List of good matches (cv2.DMatch objects)
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        # KNN matching with k=2
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def get_matched_points(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched point coordinates from keypoints and matches.

        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches

        Returns:
            Tuple of (pts1, pts2) - Nx2 arrays of corresponding points
        """
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        return pts1, pts2


def keypoints_to_array(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """Convert list of cv2.KeyPoint to Nx2 numpy array."""
    if len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def array_to_keypoints(
    points: np.ndarray,
    size: float = 7.0
) -> List[cv2.KeyPoint]:
    """Convert Nx2 numpy array to list of cv2.KeyPoint."""
    return [cv2.KeyPoint(x=p[0], y=p[1], size=size) for p in points]


def filter_points_in_frame(
    points: np.ndarray,
    width: int,
    height: int,
    margin: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points that are within the frame boundaries.

    Args:
        points: Nx2 array of point coordinates
        width: Frame width
        height: Frame height
        margin: Border margin to exclude

    Returns:
        Tuple of (filtered_points, valid_mask)
    """
    if len(points) == 0:
        return points, np.array([], dtype=bool)

    valid = (
        (points[:, 0] >= margin) &
        (points[:, 0] < width - margin) &
        (points[:, 1] >= margin) &
        (points[:, 1] < height - margin)
    )

    return points[valid], valid
