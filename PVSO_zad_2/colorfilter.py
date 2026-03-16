import cv2
import numpy as np
from ximea import xiapi


def nothing(x):
    pass


# -------------------------
# Camera setup
# -------------------------
cam = xiapi.Camera()

print("Opening camera...")
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB24")
cam.set_param("auto_wb", 1)

img = xiapi.Image()

print("Starting acquisition...")
cam.start_acquisition()


# -------------------------
# Windows
# -------------------------
cv2.namedWindow("Controls")
cv2.namedWindow("Original")
cv2.namedWindow("Mask")
cv2.namedWindow("Result")


# -------------------------
# HSV sliders
# -------------------------
cv2.createTrackbar("Lower H", "Controls", 0, 179, nothing)
cv2.createTrackbar("Lower S", "Controls", 120, 255, nothing)
cv2.createTrackbar("Lower V", "Controls", 70, 255, nothing)

cv2.createTrackbar("Upper H", "Controls", 10, 179, nothing)
cv2.createTrackbar("Upper S", "Controls", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Controls", 255, 255, nothing)


# -------------------------
# Main loop
# -------------------------
while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.resize(frame,(640,640))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read slider values
    lh = cv2.getTrackbarPos("Lower H", "Controls")
    ls = cv2.getTrackbarPos("Lower S", "Controls")
    lv = cv2.getTrackbarPos("Lower V", "Controls")

    uh = cv2.getTrackbarPos("Upper H", "Controls")
    us = cv2.getTrackbarPos("Upper S", "Controls")
    uv = cv2.getTrackbarPos("Upper V", "Controls")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Replace detected pixels with blue
    result = frame.copy()
    result[mask == 255] = [255, 0, 0]   # Blue in BGR

    # Show images
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Quit
    if cv2.waitKey(1) == ord('q'):
        break


# -------------------------
# Cleanup
# -------------------------
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()