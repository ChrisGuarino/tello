from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
tello.streamon()

while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Tello", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
tello.land()