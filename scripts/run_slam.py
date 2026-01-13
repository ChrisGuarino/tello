#!/usr/bin/env python3
"""
Run Tello Visual SLAM

Main script to run the SLAM system with the Tello drone.
Connects to the drone, streams video, and builds a 3D map.

Usage:
    python scripts/run_slam.py [options]

Options:
    --takeoff       Takeoff after connecting
    --no-3d         Disable 3D visualization
    --export PATH   Export map to PLY file on exit

Controls:
    q           - Quit and land
    s           - Save map snapshot
    r           - Reset SLAM
    SPACE       - Takeoff/Land toggle
"""

import sys
import os
import argparse
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slam import TelloSLAM


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tello Visual SLAM')
    parser.add_argument(
        '--takeoff',
        action='store_true',
        help='Takeoff after connecting'
    )
    parser.add_argument(
        '--no-3d',
        action='store_true',
        help='Disable 3D visualization'
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export map to PLY file on exit'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=500,
        help='Maximum features to detect (default: 500)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("       TELLO VISUAL SLAM")
    print("=" * 50)
    print()
    print("Controls:")
    print("  q       - Quit and land")
    print("  s       - Save map snapshot")
    print("  SPACE   - Takeoff/Land toggle")
    print()

    # Initialize SLAM
    slam = TelloSLAM(
        detector_type='ORB',
        max_features=args.max_features,
        enable_visualization=True,
        enable_3d_view=not args.no_3d
    )

    is_flying = False
    snapshot_count = 0

    try:
        # Connect to drone
        slam.connect(takeoff=args.takeoff)
        is_flying = args.takeoff

        print("\nSLAM running. Press 'q' to quit.\n")

        # Main loop
        while True:
            # Process SLAM frame
            slam.update()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit
                print("\nQuitting...")
                break

            elif key == ord('s'):
                # Save snapshot
                snapshot_count += 1
                filename = f"slam_snapshot_{snapshot_count}.ply"
                slam.export_map(filename)
                slam.export_trajectory(f"trajectory_{snapshot_count}.csv")
                print(f"Saved snapshot: {filename}")

            elif key == ord(' '):
                # Toggle takeoff/land
                if is_flying:
                    print("Landing...")
                    slam.land()
                    is_flying = False
                else:
                    print("Taking off...")
                    slam.tello.takeoff()
                    is_flying = True

            # Print position periodically
            if slam.frame_count % 30 == 0 and slam.has_pose():
                x, y, z = slam.get_position()
                stats = slam.get_stats()
                print(
                    f"Frame {slam.frame_count}: "
                    f"Pos=({x:.2f}, {y:.2f}, {z:.2f}) | "
                    f"FPS={stats['fps']:.1f} | "
                    f"Points={stats['num_map_points']} | "
                    f"KFs={stats['num_keyframes']}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        if is_flying:
            print("Landing drone...")
            slam.land()

        # Export map if requested
        if args.export:
            slam.export_map(args.export)
            print(f"Exported map to: {args.export}")

        slam.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
