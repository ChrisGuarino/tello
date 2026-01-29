# Tello Person Follower

Autonomous person-following system for the DJI Tello drone using YOLOv8 real-time object detection.

## Overview

The drone detects people in its camera feed using YOLOv8n and autonomously adjusts its yaw and forward/backward position to follow the target. Proportional control with exponential smoothing keeps tracking stable and responsive.

## Features

- **Real-time person detection** using YOLOv8n (COCO-pretrained)
- **Dual-axis autonomous control** — yaw rotation to center the target, forward/backward movement to maintain distance
- **Smoothed control** — exponential smoothing and dead zones to prevent jitter
- **Safety limits** — capped RC values (yaw: 20, forward/back: 12), battery monitoring, graceful shutdown
- **Multiple operating modes** — full autonomous following, detection-only streaming, and manual UDP console

## Scripts

| Script | Description |
|--------|-------------|
| `person_follow.py` | Full autonomous follower with tuned PID-like control |
| `detection_geometry.py` | Simplified detection and target tracking (no flight control) |
| `yolo_stream.py` | Live video stream with YOLO detection overlay |
| `udp_console.py` | Interactive console for manual drone control via Tello SDK |

## Requirements

- Python 3
- djitellopy
- ultralytics (YOLOv8)
- OpenCV

## Usage

```bash
# Autonomous person following
python person_follow.py

# Detection-only video stream
python yolo_stream.py

# Manual control console
python udp_console.py
```

Press `q` to land and exit during any flight mode.

## Control Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TARGET_AREA` | 0.15 | Desired target size (fraction of frame) |
| `MAX_YAW` | 20 | Maximum yaw rotation speed |
| `MAX_FB` | 12 | Maximum forward/backward speed |
| `ALPHA` | 0.2 | Exponential smoothing factor |
| `DEAD_ZONE` | 30 px | Horizontal dead zone to prevent oscillation |
