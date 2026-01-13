"""
TelloSLAM: Integrated Visual SLAM System for Tello Drone

Main entry point that combines all SLAM components with
Tello drone control and video streaming.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from djitellopy import Tello

from .camera import TelloCamera
from .features import FeatureDetector, FeatureTracker, keypoints_to_array
from .motion import MotionEstimator
from .mapping import SLAMMap
from .visualization import SLAMVisualizer


class TelloSLAM:
    """
    Main SLAM interface for Tello drone.

    Combines feature detection, tracking, motion estimation,
    and mapping into a single easy-to-use class.

    Usage:
        slam = TelloSLAM()
        slam.connect()

        while True:
            if slam.update():
                x, y, z = slam.get_position()
                print(f"Position: {x:.2f}, {y:.2f}, {z:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        slam.export_map("map.ply")
        slam.shutdown()
    """

    def __init__(
        self,
        detector_type: str = 'ORB',
        max_features: int = 500,
        enable_visualization: bool = True,
        enable_3d_view: bool = True,
        min_tracked_points: int = 50,
        keyframe_interval: int = 10
    ):
        """
        Initialize TelloSLAM.

        Args:
            detector_type: Feature detector ('ORB', 'SIFT', 'AKAZE')
            max_features: Maximum features to detect per frame
            enable_visualization: Enable 2D visualization
            enable_3d_view: Enable 3D point cloud view
            min_tracked_points: Minimum points before re-detection
            keyframe_interval: Frames between keyframes
        """
        # Tello
        self.tello = Tello()
        self.frame_read = None

        # Camera
        self.camera = TelloCamera()

        # SLAM components
        self.detector = FeatureDetector(detector_type, max_features)
        self.tracker = FeatureTracker()
        self.motion_estimator = MotionEstimator(self.camera.K)
        self.slam_map = SLAMMap(self.camera.K)

        # Visualization
        self.enable_vis = enable_visualization
        self.enable_3d = enable_3d_view
        self.visualizer = None
        if enable_visualization:
            self.visualizer = SLAMVisualizer(enable_3d=enable_3d_view)

        # Parameters
        self.min_tracked_points = min_tracked_points
        self.keyframe_interval = keyframe_interval

        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points = None

        self.frame_count = 0
        self.last_keyframe_id = -1
        self.initialized = False
        self.is_connected = False

        # Statistics
        self.stats = {
            'fps': 0.0,
            'num_tracked': 0,
            'num_map_points': 0,
            'num_keyframes': 0
        }
        self.last_time = time.time()

    def connect(self, takeoff: bool = False):
        """
        Connect to Tello and start video stream.

        Args:
            takeoff: Whether to takeoff after connecting
        """
        print("Connecting to Tello...")
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")

        print("Starting video stream...")
        self.tello.streamon()
        time.sleep(1)  # Wait for stream to stabilize

        self.frame_read = self.tello.get_frame_read()
        self.is_connected = True

        # Initialize visualization
        if self.enable_vis and self.visualizer:
            if self.enable_3d:
                self.visualizer.init_3d_viewer()

        if takeoff:
            print("Taking off...")
            self.tello.takeoff()
            time.sleep(2)

        print("SLAM ready!")

    def update(self) -> bool:
        """
        Process one frame of SLAM.

        Returns:
            True if frame was processed successfully
        """
        if not self.is_connected or self.frame_read is None:
            return False

        # Get frame
        frame = self.frame_read.frame
        if frame is None or frame.size == 0:
            return False

        # Resize to standard size
        frame = cv2.resize(frame, (self.camera.width, self.camera.height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.stats['fps'] = 0.9 * self.stats['fps'] + 0.1 * (1.0 / dt)
        self.last_time = current_time

        # First frame initialization
        if not self.initialized:
            self._initialize(gray, frame)
            return True

        # Track features
        tracked_points, status, _ = self.tracker.track_optical_flow(
            self.prev_frame, gray, self.prev_points
        )

        # Count valid tracks
        valid_mask = status.flatten() == 1
        num_tracked = np.sum(valid_mask)
        self.stats['num_tracked'] = num_tracked

        # Check if we need to re-detect features
        if num_tracked < self.min_tracked_points:
            self._initialize(gray, frame)
            return True

        # Get valid correspondences
        pts1 = self.prev_points[valid_mask]
        pts2 = tracked_points[valid_mask]

        # Estimate motion
        R, t, inliers, num_inliers = self.motion_estimator.estimate_pose_essential(
            pts1, pts2
        )

        if R is not None and num_inliers >= 8:
            # Update pose
            self.slam_map.update_pose(R, t)

            # Check for keyframe
            if self._should_create_keyframe():
                self._create_keyframe(gray, frame, pts1, pts2, valid_mask)

        # Update visualization
        if self.enable_vis and self.visualizer:
            info = {
                'FPS': f"{self.stats['fps']:.1f}",
                'Tracked': num_tracked,
                'Map Points': self.slam_map.num_map_points(),
                'Keyframes': self.slam_map.num_keyframes()
            }

            self.visualizer.update_2d_view(
                frame,
                tracked_points=tracked_points,
                status=status,
                info_text=info
            )

            if self.enable_3d:
                self.visualizer.update_3d_view(
                    points=self.slam_map.get_all_map_points(),
                    colors=self.slam_map.get_map_point_colors(),
                    trajectory=self.slam_map.get_trajectory_positions(),
                    current_pose=self.slam_map.current_pose
                )

        # Update state for next frame
        self.prev_frame = gray.copy()
        self.prev_points = tracked_points[valid_mask]
        self.frame_count += 1

        self.stats['num_map_points'] = self.slam_map.num_map_points()
        self.stats['num_keyframes'] = self.slam_map.num_keyframes()

        return True

    def _initialize(self, gray: np.ndarray, color_frame: np.ndarray):
        """Initialize or re-initialize tracking."""
        keypoints, descriptors = self.detector.detect_and_compute(gray)

        if len(keypoints) == 0:
            return

        self.prev_frame = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points = keypoints_to_array(keypoints)

        if not self.initialized:
            # Create first keyframe at origin
            self.slam_map.add_keyframe(
                pose=np.eye(4),
                keypoints=keypoints,
                descriptors=descriptors,
                image=color_frame.copy()
            )
            self.last_keyframe_id = 0
            self.initialized = True
            print("SLAM initialized with first keyframe")

    def _should_create_keyframe(self) -> bool:
        """Determine if a new keyframe should be created."""
        # Simple interval-based keyframe selection
        frames_since_last = self.frame_count - self.last_keyframe_id * self.keyframe_interval
        return frames_since_last >= self.keyframe_interval

    def _create_keyframe(
        self,
        gray: np.ndarray,
        color_frame: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        valid_mask: np.ndarray
    ):
        """Create a new keyframe and triangulate points."""
        # Detect features for new keyframe
        keypoints, descriptors = self.detector.detect_and_compute(gray)

        if len(keypoints) == 0:
            return

        # Add keyframe
        kf = self.slam_map.add_keyframe(
            pose=self.slam_map.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            image=color_frame.copy()
        )

        # Triangulate points with previous keyframe
        if self.last_keyframe_id >= 0 and descriptors is not None:
            prev_kf = self.slam_map.keyframes[self.last_keyframe_id]

            if prev_kf.descriptors is not None:
                # Match descriptors
                matches = self.tracker.match_descriptors(
                    prev_kf.descriptors,
                    descriptors
                )

                if len(matches) > 10:
                    # Triangulate
                    new_points = self.slam_map.triangulate_between_keyframes(
                        prev_kf, kf, matches
                    )

                    if len(new_points) > 0:
                        print(f"Keyframe {kf.id}: +{len(new_points)} map points")

        self.last_keyframe_id = kf.id

        # Update tracking state with new features
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points = keypoints_to_array(keypoints)

    def get_pose(self) -> np.ndarray:
        """Get current 4x4 camera pose matrix."""
        return self.slam_map.current_pose.copy()

    def get_position(self) -> Tuple[float, float, float]:
        """Get current camera position (x, y, z)."""
        pos = self.slam_map.get_current_position()
        return tuple(pos)

    def get_map_points(self) -> np.ndarray:
        """Get all 3D map points as Nx3 array."""
        return self.slam_map.get_all_map_points()

    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory as Nx3 array of positions."""
        return self.slam_map.get_trajectory_positions()

    def has_pose(self) -> bool:
        """Check if SLAM has initialized and has a valid pose."""
        return self.initialized

    def export_map(self, filename: str):
        """Export map points to PLY file."""
        self.slam_map.export_ply(filename)

    def export_trajectory(self, filename: str):
        """Export trajectory to CSV file."""
        self.slam_map.export_trajectory(filename)

    def send_rc_control(
        self,
        left_right: int,
        forward_back: int,
        up_down: int,
        yaw: int
    ):
        """
        Send RC control command to Tello.

        Args:
            left_right: -100 to 100
            forward_back: -100 to 100
            up_down: -100 to 100
            yaw: -100 to 100
        """
        if self.is_connected:
            self.tello.send_rc_control(left_right, forward_back, up_down, yaw)

    def land(self):
        """Land the drone."""
        if self.is_connected:
            self.send_rc_control(0, 0, 0, 0)
            self.tello.land()

    def shutdown(self):
        """Clean shutdown of SLAM and Tello."""
        print("Shutting down...")

        if self.is_connected:
            self.tello.streamoff()
            self.tello.end()

        if self.visualizer:
            self.visualizer.close()

        self.is_connected = False
        print("Shutdown complete")

    def get_stats(self) -> dict:
        """Get current SLAM statistics."""
        return self.stats.copy()

    def get_battery(self) -> int:
        """Get Tello battery percentage."""
        if self.is_connected:
            return self.tello.get_battery()
        return 0
