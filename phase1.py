from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery(), "%")

tello.streamon()

while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))
    # cv2.imshow("Tello Camera", frame)
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Tello YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
tello.end()

