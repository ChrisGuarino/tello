from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

# Load a COCO-pretrained YOLOv8n model
model = YOLO("model/yolov8n.pt")

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery(), "%")

tello.streamon()

frame_read = tello.get_frame_read()  # ← move this OUTSIDE the loop

while True:
    frame = frame_read.frame
    if frame is None or frame.size == 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)
    r = results[0]
    boxes = r.boxes

    annotated = r.plot()
    cv2.imshow("Tello YOLO", annotated)

    # Filter for target class
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if r.names[cls_id] != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            target = {
                "cx": cx,
                "cy": cy,
                "area": area,
                "confidence": conf
            }

            print("TARGET:", target)

            break  # ← IMPORTANT: only track ONE target

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


tello.streamoff()
cv2.destroyAllWindows()
tello.end()

