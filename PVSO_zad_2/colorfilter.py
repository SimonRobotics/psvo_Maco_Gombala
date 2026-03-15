import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

blue = np.zeros((480, 640, 3), dtype=np.uint8)
blue[:,:,0] = 255
blue[:,:,1] = 0
blue[:,:,2] = 0 

# Create a window for sliders
cv2.namedWindow("Trackbars")

# Create RGB lower sliders
cv2.createTrackbar("Lower R", "Trackbars", 60, 255, nothing)
cv2.createTrackbar("Lower G", "Trackbars", 35, 255, nothing)
cv2.createTrackbar("Lower B", "Trackbars", 140, 255, nothing)

# Create RGB upper sliders
cv2.createTrackbar("Upper R", "Trackbars", 180, 255, nothing)
cv2.createTrackbar("Upper G", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper B", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read slider values
    lr = cv2.getTrackbarPos("Lower R", "Trackbars")
    lg = cv2.getTrackbarPos("Lower G", "Trackbars")
    lb = cv2.getTrackbarPos("Lower B", "Trackbars")

    ur = cv2.getTrackbarPos("Upper R", "Trackbars")
    ug = cv2.getTrackbarPos("Upper G", "Trackbars")
    ub = cv2.getTrackbarPos("Upper B", "Trackbars")

    lower = np.array([lb, lg, lr])
    upper = np.array([ub, ug, ur])

    # Create mask
    mask = cv2.inRange(frame, lower, upper)

    result = cv2.bitwise_and(frame, blue, mask=mask)

    result = np.where(mask[...,np.newaxis], result, frame)

    # Show frames
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()