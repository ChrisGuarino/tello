1. Tello SDK + video stream (baseline layer)

Get djitellopy working reliably.

Verify basic commands: connect, takeoff, land, rc control.

Open a simple preview window showing raw frames from the video stream.

No machine learning yet — just make sure the drone → laptop → display loop is solid.

2. Detection only (no drone control yet)

Integrate a clean YOLOv8 inference loop on each incoming frame.

Draw bounding boxes, labels, and confidence scores in the preview window.

Confirm stable FPS, no freezes, and no latency spikes.

This step validates your vision pipeline.

3. Basic rule-based control loop (tracking behavior)

Use YOLO detections (box center, width, area) to compute simple control decisions:

If object is left → yaw left

If object is right → yaw right

If object is small → move forward

If object is large → move back

No reinforcement learning yet.

This layer gives you a deterministic, predictable baseline behavior.

4. RL integration (optional, more advanced)

Build a minimal Gymnasium environment that mirrors the real control problem.

Train a policy (PPO, DQN, etc.) using simulated state or processed vision features.

Add a feature flag allowing you to switch between:

--mode rule

--mode rl

Only do this once the rest of the pipeline is stable.

5. Packaging, structure, and configuration

Add a config file for:

YOLO weights

thresholds

smoothing parameters

FPS targets

Tello IP/ports

Add a small CLI launcher so you can run:

python run.py --mode detect

python run.py --mode control

python run.py --mode rl

Keep the project clean and modular as it grows.