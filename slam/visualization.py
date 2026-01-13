"""
Visualization for Visual SLAM

Provides 2D (OpenCV) and 3D (Open3D) visualization for
tracking features, point clouds, and camera trajectory.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

# Open3D is optional - import with fallback
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization disabled.")
    print("Install with: pip install open3d")


class SLAMVisualizer:
    """
    Real-time visualization for SLAM using OpenCV and Open3D.

    Features:
    - 2D view: Camera frame with tracked features
    - 3D view: Point cloud and camera trajectory (if Open3D available)
    """

    # Colors (BGR for OpenCV)
    COLOR_KEYPOINT = (0, 255, 0)      # Green
    COLOR_TRACKED = (255, 0, 0)        # Blue
    COLOR_LOST = (0, 0, 255)           # Red
    COLOR_TRAJECTORY = (255, 255, 0)   # Cyan
    COLOR_TEXT = (255, 255, 255)       # White

    def __init__(
        self,
        window_name: str = "SLAM",
        enable_3d: bool = True,
        point_size: float = 2.0
    ):
        """
        Initialize the visualizer.

        Args:
            window_name: Name for OpenCV windows
            enable_3d: Whether to enable 3D visualization
            point_size: Size of points in 3D view
        """
        self.window_name = window_name
        self.enable_3d = enable_3d and OPEN3D_AVAILABLE
        self.point_size = point_size

        # 3D visualization objects
        self.vis_3d = None
        self.pcd = None
        self.trajectory_line = None
        self.camera_frame = None
        self.is_3d_initialized = False

        # Track history for drawing trails
        self.track_history: List[List[Tuple[float, float]]] = []
        self.max_trail_length = 10

    def init_3d_viewer(self, width: int = 800, height: int = 600):
        """
        Initialize the Open3D 3D viewer.

        Args:
            width: Window width
            height: Window height
        """
        if not self.enable_3d:
            return

        self.vis_3d = o3d.visualization.Visualizer()
        self.vis_3d.create_window(
            window_name=f"{self.window_name} - 3D Map",
            width=width,
            height=height
        )

        # Add coordinate frame at origin
        self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3
        )
        self.vis_3d.add_geometry(self.camera_frame)

        # Initialize empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis_3d.add_geometry(self.pcd)

        # Initialize trajectory line set
        self.trajectory_line = o3d.geometry.LineSet()
        self.vis_3d.add_geometry(self.trajectory_line)

        # Set render options
        render_option = self.vis_3d.get_render_option()
        render_option.point_size = self.point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        # Set initial view
        ctr = self.vis_3d.get_view_control()
        ctr.set_zoom(0.5)

        self.is_3d_initialized = True

    def update_2d_view(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[cv2.KeyPoint]] = None,
        tracked_points: Optional[np.ndarray] = None,
        status: Optional[np.ndarray] = None,
        info_text: Optional[dict] = None
    ) -> np.ndarray:
        """
        Update and display the 2D camera view with features.

        Args:
            frame: BGR camera frame
            keypoints: Detected keypoints to draw
            tracked_points: Nx2 array of tracked point positions
            status: Nx1 tracking status (1=tracked, 0=lost)
            info_text: Dictionary of info to display

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        # Draw keypoints
        if keypoints is not None:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis_frame, (x, y), 3, self.COLOR_KEYPOINT, -1)

        # Draw tracked points with status
        if tracked_points is not None and len(tracked_points) > 0:
            for i, pt in enumerate(tracked_points):
                x, y = int(pt[0]), int(pt[1])

                if status is not None and i < len(status):
                    color = self.COLOR_TRACKED if status[i] else self.COLOR_LOST
                else:
                    color = self.COLOR_TRACKED

                cv2.circle(vis_frame, (x, y), 4, color, -1)

        # Draw info text
        if info_text:
            y_offset = 30
            for key, value in info_text.items():
                text = f"{key}: {value}"
                cv2.putText(
                    vis_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.COLOR_TEXT, 2
                )
                y_offset += 25

        # Display
        cv2.imshow(f"{self.window_name} - Camera", vis_frame)

        return vis_frame

    def update_3d_view(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        trajectory: Optional[np.ndarray] = None,
        current_pose: Optional[np.ndarray] = None
    ) -> bool:
        """
        Update the 3D visualization with map points and trajectory.

        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255)
            trajectory: Mx3 array of camera positions
            current_pose: 4x4 current camera pose

        Returns:
            True if visualization is still active, False if closed
        """
        if not self.enable_3d or not self.is_3d_initialized:
            return True

        # Update point cloud
        if len(points) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(points)

            if colors is not None and len(colors) == len(points):
                # Normalize colors to 0-1
                colors_normalized = colors.astype(np.float64) / 255.0
                self.pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            else:
                # Default white color
                self.pcd.colors = o3d.utility.Vector3dVector(
                    np.ones((len(points), 3)) * 0.8
                )

            self.vis_3d.update_geometry(self.pcd)

        # Update trajectory
        if trajectory is not None and len(trajectory) > 1:
            lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
            colors_line = [[0, 1, 0] for _ in lines]  # Green trajectory

            self.trajectory_line.points = o3d.utility.Vector3dVector(trajectory)
            self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
            self.trajectory_line.colors = o3d.utility.Vector3dVector(colors_line)

            self.vis_3d.update_geometry(self.trajectory_line)

        # Update camera frame position
        if current_pose is not None:
            self.camera_frame.transform(np.linalg.inv(
                self.camera_frame.get_center()
            ))
            # Reset and apply new transform
            new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            new_frame.transform(current_pose)
            self.camera_frame.vertices = new_frame.vertices
            self.camera_frame.vertex_colors = new_frame.vertex_colors
            self.vis_3d.update_geometry(self.camera_frame)

        # Poll events and render
        self.vis_3d.poll_events()
        self.vis_3d.update_renderer()

        return True

    def draw_optical_flow(
        self,
        frame: np.ndarray,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        status: np.ndarray
    ) -> np.ndarray:
        """
        Draw optical flow vectors between frames.

        Args:
            frame: BGR frame to draw on
            prev_points: Nx2 previous point positions
            curr_points: Nx2 current point positions
            status: Nx1 tracking status

        Returns:
            Frame with flow vectors drawn
        """
        vis_frame = frame.copy()
        mask = status.flatten().astype(bool)

        for prev_pt, curr_pt in zip(prev_points[mask], curr_points[mask]):
            x1, y1 = int(prev_pt[0]), int(prev_pt[1])
            x2, y2 = int(curr_pt[0]), int(curr_pt[1])

            # Draw line
            cv2.line(vis_frame, (x1, y1), (x2, y2), self.COLOR_TRAJECTORY, 1)
            # Draw current point
            cv2.circle(vis_frame, (x2, y2), 3, self.COLOR_TRACKED, -1)

        return vis_frame

    def draw_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches: int = 50
    ) -> np.ndarray:
        """
        Draw feature matches between two images.

        Args:
            img1: First image
            img2: Second image
            kp1: Keypoints in first image
            kp2: Keypoints in second image
            matches: List of matches
            max_matches: Maximum matches to draw

        Returns:
            Combined image with matches drawn
        """
        # Limit matches for clarity
        matches_to_draw = matches[:max_matches]

        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return match_img

    def close(self):
        """Close all visualization windows."""
        cv2.destroyAllWindows()

        if self.enable_3d and self.vis_3d is not None:
            self.vis_3d.destroy_window()
            self.vis_3d = None
            self.is_3d_initialized = False


def draw_trajectory_topdown(
    trajectory: np.ndarray,
    scale: float = 100.0,
    img_size: int = 500
) -> np.ndarray:
    """
    Draw a top-down (XZ plane) view of the trajectory.

    Args:
        trajectory: Nx3 array of positions
        scale: Pixels per unit distance
        img_size: Size of output image

    Returns:
        BGR image with trajectory drawn
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = img_size // 2

    if len(trajectory) < 2:
        return img

    # Scale and center trajectory
    traj_2d = trajectory[:, [0, 2]]  # X and Z
    traj_scaled = traj_2d * scale + center

    # Draw trajectory
    for i in range(len(traj_scaled) - 1):
        pt1 = tuple(traj_scaled[i].astype(int))
        pt2 = tuple(traj_scaled[i + 1].astype(int))

        # Check bounds
        if all(0 <= p < img_size for p in pt1 + pt2):
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # Draw current position
    if len(traj_scaled) > 0:
        curr = tuple(traj_scaled[-1].astype(int))
        if all(0 <= p < img_size for p in curr):
            cv2.circle(img, curr, 5, (0, 0, 255), -1)

    # Draw origin
    cv2.circle(img, (center, center), 3, (255, 255, 255), -1)
    cv2.putText(
        img, "O", (center + 5, center - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )

    return img
