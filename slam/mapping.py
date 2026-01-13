"""
Map Management for Visual SLAM

Manages keyframes, map points, and provides triangulation
for building the 3D map from visual observations.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from .motion import triangulate_points, make_projection_matrix


@dataclass
class MapPoint:
    """
    Represents a 3D point in the SLAM map.

    Attributes:
        id: Unique identifier
        position: 3D coordinates [x, y, z]
        descriptor: Feature descriptor for matching
        color: RGB color from image (optional)
        observations: List of (keyframe_id, keypoint_idx) tuples
        is_valid: Whether the point should be used
    """
    id: int
    position: np.ndarray
    descriptor: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    observations: List[Tuple[int, int]] = field(default_factory=list)
    is_valid: bool = True

    def add_observation(self, keyframe_id: int, keypoint_idx: int):
        """Add an observation from a keyframe."""
        self.observations.append((keyframe_id, keypoint_idx))

    def num_observations(self) -> int:
        """Return number of times this point has been observed."""
        return len(self.observations)


@dataclass
class KeyFrame:
    """
    Represents a camera pose with associated features.

    Attributes:
        id: Unique identifier
        pose: 4x4 transformation matrix (camera to world)
        keypoints: List of cv2.KeyPoint
        descriptors: Feature descriptors (Nx32 for ORB)
        map_point_ids: List mapping keypoint index to MapPoint id (-1 if none)
        image: Grayscale image (optional, for visualization)
    """
    id: int
    pose: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    map_point_ids: List[int] = field(default_factory=list)
    image: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.map_point_ids:
            self.map_point_ids = [-1] * len(self.keypoints)

    def get_position(self) -> np.ndarray:
        """Get camera position in world coordinates."""
        return self.pose[:3, 3].copy()

    def get_rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return self.pose[:3, :3].copy()

    def get_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """Get 3x4 projection matrix."""
        R = self.pose[:3, :3]
        t = self.pose[:3, 3:4]
        # Camera pose is world-to-camera, so invert
        R_inv = R.T
        t_inv = -R.T @ t
        return make_projection_matrix(K, R_inv, t_inv)


class SLAMMap:
    """
    Manages the SLAM map including keyframes and 3D points.

    Provides methods for adding keyframes, triangulating points,
    and querying the map.
    """

    def __init__(self, camera_matrix: np.ndarray):
        """
        Initialize the SLAM map.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
        """
        self.K = camera_matrix
        self.keyframes: Dict[int, KeyFrame] = {}
        self.map_points: Dict[int, MapPoint] = {}

        self.current_pose = np.eye(4, dtype=np.float64)
        self.next_keyframe_id = 0
        self.next_map_point_id = 0

        # Tracking state
        self.trajectory: List[np.ndarray] = [self.current_pose.copy()]

    def add_keyframe(
        self,
        pose: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        descriptors: Optional[np.ndarray],
        image: Optional[np.ndarray] = None
    ) -> KeyFrame:
        """
        Add a new keyframe to the map.

        Args:
            pose: 4x4 camera pose (camera to world)
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            image: Grayscale image (optional)

        Returns:
            The created KeyFrame
        """
        kf = KeyFrame(
            id=self.next_keyframe_id,
            pose=pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            image=image
        )
        self.keyframes[kf.id] = kf
        self.next_keyframe_id += 1
        return kf

    def add_map_point(
        self,
        position: np.ndarray,
        descriptor: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None
    ) -> MapPoint:
        """
        Add a new 3D point to the map.

        Args:
            position: 3D position [x, y, z]
            descriptor: Feature descriptor
            color: RGB color

        Returns:
            The created MapPoint
        """
        mp = MapPoint(
            id=self.next_map_point_id,
            position=position.copy(),
            descriptor=descriptor,
            color=color
        )
        self.map_points[mp.id] = mp
        self.next_map_point_id += 1
        return mp

    def triangulate_between_keyframes(
        self,
        kf1: KeyFrame,
        kf2: KeyFrame,
        matches: List[cv2.DMatch],
        min_parallax_deg: float = 1.0,
        max_reproj_error: float = 2.0
    ) -> List[MapPoint]:
        """
        Triangulate new 3D points from matched features between keyframes.

        Args:
            kf1: First keyframe
            kf2: Second keyframe
            matches: Feature matches between keyframes
            min_parallax_deg: Minimum parallax angle (degrees)
            max_reproj_error: Maximum reprojection error (pixels)

        Returns:
            List of newly created MapPoints
        """
        if len(matches) == 0:
            return []

        # Get matched points
        pts1 = np.array([kf1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([kf2.keypoints[m.trainIdx].pt for m in matches])

        # Get projection matrices
        P1 = kf1.get_projection_matrix(self.K)
        P2 = kf2.get_projection_matrix(self.K)

        # Triangulate all points
        points_3d = triangulate_points(pts1, pts2, P1, P2)

        # Filter and add valid points
        new_points = []

        for i, (match, pt3d) in enumerate(zip(matches, points_3d)):
            # Check if point is in front of both cameras
            pt_cam1 = (np.linalg.inv(kf1.pose) @ np.append(pt3d, 1))[:3]
            pt_cam2 = (np.linalg.inv(kf2.pose) @ np.append(pt3d, 1))[:3]

            if pt_cam1[2] <= 0 or pt_cam2[2] <= 0:
                continue

            # Check reprojection error
            pt3d_h = np.append(pt3d, 1)

            proj1 = P1 @ pt3d_h
            proj1 = proj1[:2] / proj1[2]
            err1 = np.linalg.norm(pts1[i] - proj1)

            proj2 = P2 @ pt3d_h
            proj2 = proj2[:2] / proj2[2]
            err2 = np.linalg.norm(pts2[i] - proj2)

            if err1 > max_reproj_error or err2 > max_reproj_error:
                continue

            # Get descriptor and color
            desc = None
            if kf1.descriptors is not None:
                desc = kf1.descriptors[match.queryIdx].copy()

            color = None
            if kf1.image is not None:
                x, y = int(pts1[i][0]), int(pts1[i][1])
                if 0 <= x < kf1.image.shape[1] and 0 <= y < kf1.image.shape[0]:
                    if len(kf1.image.shape) == 3:
                        color = kf1.image[y, x, ::-1].copy()  # BGR to RGB
                    else:
                        gray = kf1.image[y, x]
                        color = np.array([gray, gray, gray], dtype=np.uint8)

            # Create map point
            mp = self.add_map_point(pt3d, desc, color)
            mp.add_observation(kf1.id, match.queryIdx)
            mp.add_observation(kf2.id, match.trainIdx)

            # Update keyframe references
            kf1.map_point_ids[match.queryIdx] = mp.id
            kf2.map_point_ids[match.trainIdx] = mp.id

            new_points.append(mp)

        return new_points

    def update_pose(self, R: np.ndarray, t: np.ndarray):
        """
        Update current pose with relative motion.

        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        # Build relative transformation
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()

        # Accumulate: T_new = T_current @ T_relative
        self.current_pose = self.current_pose @ T_rel
        self.trajectory.append(self.current_pose.copy())

    def get_current_position(self) -> np.ndarray:
        """Get current camera position."""
        return self.current_pose[:3, 3].copy()

    def get_all_map_points(self) -> np.ndarray:
        """
        Get all valid map points as Nx3 array.

        Returns:
            Nx3 array of 3D point positions
        """
        valid_points = [
            mp.position for mp in self.map_points.values()
            if mp.is_valid
        ]
        if len(valid_points) == 0:
            return np.zeros((0, 3))
        return np.array(valid_points)

    def get_map_point_colors(self) -> np.ndarray:
        """
        Get colors for all valid map points.

        Returns:
            Nx3 array of RGB colors (0-255)
        """
        colors = []
        for mp in self.map_points.values():
            if mp.is_valid:
                if mp.color is not None:
                    colors.append(mp.color)
                else:
                    colors.append([128, 128, 128])  # Default gray
        if len(colors) == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array(colors, dtype=np.uint8)

    def get_trajectory_positions(self) -> np.ndarray:
        """
        Get all camera positions along the trajectory.

        Returns:
            Nx3 array of camera positions
        """
        return np.array([T[:3, 3] for T in self.trajectory])

    def get_local_map_points(
        self,
        center: np.ndarray,
        radius: float = 5.0
    ) -> List[MapPoint]:
        """
        Get map points within radius of a center point.

        Args:
            center: 3D center point
            radius: Search radius

        Returns:
            List of MapPoints within radius
        """
        local_points = []
        for mp in self.map_points.values():
            if mp.is_valid:
                dist = np.linalg.norm(mp.position - center)
                if dist < radius:
                    local_points.append(mp)
        return local_points

    def cull_bad_points(self, min_observations: int = 2):
        """
        Mark points with too few observations as invalid.

        Args:
            min_observations: Minimum required observations
        """
        for mp in self.map_points.values():
            if mp.num_observations() < min_observations:
                mp.is_valid = False

    def num_keyframes(self) -> int:
        """Return number of keyframes."""
        return len(self.keyframes)

    def num_map_points(self) -> int:
        """Return number of valid map points."""
        return sum(1 for mp in self.map_points.values() if mp.is_valid)

    def export_ply(self, filename: str):
        """
        Export map points to PLY file.

        Args:
            filename: Output PLY file path
        """
        points = self.get_all_map_points()
        colors = self.get_map_point_colors()

        if len(points) == 0:
            print("No points to export")
            return

        with open(filename, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write points
            for pt, col in zip(points, colors):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ")
                f.write(f"{int(col[0])} {int(col[1])} {int(col[2])}\n")

        print(f"Exported {len(points)} points to {filename}")

    def export_trajectory(self, filename: str):
        """
        Export trajectory to CSV file.

        Args:
            filename: Output CSV file path
        """
        positions = self.get_trajectory_positions()

        with open(filename, 'w') as f:
            f.write("frame,x,y,z\n")
            for i, pos in enumerate(positions):
                f.write(f"{i},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")

        print(f"Exported trajectory ({len(positions)} poses) to {filename}")
