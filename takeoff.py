from djitellopy import Tello
import time

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery(), "%")

tello.takeoff()
time.sleep(5)   # hover for 5 seconds
tello.land()

tello.end()
