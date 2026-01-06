from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

#Control refrence values 
# Frame geometry
FRAME_W = 640
FRAME_H = 480
FRAME_CX = FRAME_W // 2
FRAME_AREA = FRAME_W * FRAME_H

# Desired distance (≈ how big the person appears)
DESIRED_AREA = 0.15          # normalized [0–1]

# Dead zones
DEAD_ZONE_X = 30             # pixels (yaw)
AREA_DEAD_ZONE = 0.02        # normalized area band

# Control limits
MAX_YAW = 20
MAX_FB  = 12                 # slower forward/back = safer

# Control gains
K_YAW = 0.15
K_FB_RETREAT  = 40.0         # strong retreat
K_FB_APPROACH = 15.0         # gentle approach

# Smoothing
ALPHA = 0.2
smoothed_area = None

# Load a COCO-pretrained YOLOv8n model
model = YOLO("model/yolov8n.pt")

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery(), "%")

tello.streamon()

frame_read = tello.get_frame_read() #Starts a video thread stream

tello.takeoff()
time.sleep(2)

#Start of Control loop
while True:
    #Safety lopp exist if 'q' pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    #This is a defensive measure. Grabs the latest frame from the video and checks if is exits or
    # or is empty. If so it skips this loop iteration. 
    frame = frame_read.frame
    if frame is None or frame.size == 0:
        continue

    # Locks the video input size for the model. 
    frame = cv2.resize(frame, (640, 480))

    # This feeds the latest frame into the model, gets the detections and then extracts all the bounding boxes. 
    results = model(frame, verbose=False)
    r = results[0]
    boxes = r.boxes #Gets all detected bounding boxes, with coordinates, class ID, and confidence

    # Draw the bounding boxes on the frame
    annotated = r.plot()
    cv2.imshow("Tello YOLO", annotated)

    # Filter for target class
    if boxes is not None and len(boxes) > 0: #Checks for no detections
        for box in boxes: # Cycle through the detections 
            cls_id = int(box.cls[0]) #Get the detection class id value and class to an int, ex: cls_id = 0 → "person"
            conf = float(box.conf[0]) # Do the same with the confience

            # Do nothing if dection is not target. In this case "person"
            # if r.names[cls_id] != "person":
            if cls_id != 0: #Trying to use Class ID value
                tello.send_rc_control(0, 0, 0, 0)
                continue
            
            # Gets the bounding box coordinates. 
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # -------------------------------
            # TARGET GEOMETRY
            # -------------------------------
            #Get Bounding Box Center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            #Calculate Area of Bounding Box
            area = (x2 - x1) * (y2 - y1)
            '''(x1, y1) ──────────┐
               │                  │
               │     PERSON       │
               │                  │
               └──────────------(x2,y2)'''

            # Normalize area to [0–1] - Basically returns a 0-1 percentage of how large the target 
            # bounding box is in relation to the total frame area. 
            norm_area = area / FRAME_AREA

            # Exponential smoothing (reduces YOLO jitter)
            # This basically BLENDS past smoothed areas with current area. Helps to keep the area from 
            # jumping around which happens with YOLO easily. 
            if smoothed_area is None:
                smoothed_area = norm_area
            else:
                smoothed_area = ALPHA * norm_area + (1 - ALPHA) * smoothed_area

            # -------------------------------
            # YAW CONTROL (left/right)
            # -------------------------------
            yaw_error = cx - FRAME_CX # Measures how far left or right the target it from the center of the frome. 
            #P-Control - K_YAW is gain. 
            '''Big error → big turn
            Small error → gentle turn'''
            yaw_cmd = int(K_YAW * yaw_error) 

            # Dead zone to prevent twitching, if the yaw_error is 
            # close enough to the center do nothing, thats what the DEAD_ZONE_X is for. 
            if abs(yaw_error) < DEAD_ZONE_X:
                yaw_cmd = 0

            #Clamp - "Never rotate too fast" - Helps to avoid wild jumping due to YOLO gliches and bounding box jumps
            yaw_cmd = max(-MAX_YAW, min(MAX_YAW, yaw_cmd))

            # -------------------------------
            # DISTANCE CONTROL (forward/back)
            # -------------------------------
            area_error = smoothed_area - DESIRED_AREA # How close is teh drone to the target.

            if area_error > AREA_DEAD_ZONE:
                # TOO CLOSE → back away strongly
                fb_cmd = int(-K_FB_RETREAT * area_error)

            elif area_error < -AREA_DEAD_ZONE:
                # TOO FAR → approach slowly
                fb_cmd = int(-K_FB_APPROACH * area_error)

            else:
                # Within desired distance band → hover
                fb_cmd = 0

            fb_cmd = max(-MAX_FB, min(MAX_FB, fb_cmd))

            # -------------------------------
            # SEND COMMAND
            # -------------------------------
            tello.send_rc_control(0, fb_cmd, 0, yaw_cmd)

            # Debug (very useful while tuning)
            print(
                f"area raw={norm_area:.3f} "
                f"smooth={smoothed_area:.3f} "
                f"err={area_error:.3f} "
                f"fb={fb_cmd} yaw={yaw_cmd}"
            )

            break  # ← IMPORTANT: only track ONE target

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.send_rc_control(0, 0, 0, 0)   
        tello.land()
        break

        # tello.send_rc_control(0, 0, 0, 0) #Stops drone if it dont see a person

tello.send_rc_control(0, 0, 0, 0)
tello.land()

tello.streamoff()
cv2.destroyAllWindows()
tello.end()

