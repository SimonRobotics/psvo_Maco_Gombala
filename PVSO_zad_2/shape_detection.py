import cv2
from ximea import xiapi
import numpy as np

MIN_CONTOUR_AREA = 1000


def load_calibration(path):
    data = np.load(path, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeff = data["dist_coeff"]
    return camera_matrix, dist_coeff


def setup_camera():
    cam = xiapi.Camera()

    print("Opening first camera...")
    cam.open_device()

    cam.set_exposure(50000)
    cam.set_param("imgdataformat", "XI_RGB24")
    cam.set_param("auto_wb", 1)

    print(f"Exposure was set to {cam.get_exposure()} us")

    img = xiapi.Image()

    print("Starting data acquisition...")
    cam.start_acquisition()

    return cam, img


def nothing(x):
    pass


def create_trackbars():
    cv2.namedWindow("Hough Controls")

    # trackbar nesmie mať float, preto dp budeme deliť 10
    cv2.createTrackbar("dp x10", "Hough Controls", 14, 50, nothing)       # 1.2
    cv2.createTrackbar("minDist", "Hough Controls", 300, 300, nothing)
    cv2.createTrackbar("param1", "Hough Controls", 100, 300, nothing)
    cv2.createTrackbar("param2", "Hough Controls", 30, 200, nothing)
    cv2.createTrackbar("minRadius", "Hough Controls", 44, 200, nothing)
    cv2.createTrackbar("maxRadius", "Hough Controls", 155, 300, nothing)

    # bonus: blur kernel
    cv2.createTrackbar("blur", "Hough Controls", 1, 31, nothing)


def get_trackbar_values():
    dp = cv2.getTrackbarPos("dp x10", "Hough Controls") / 10.0
    minDist = cv2.getTrackbarPos("minDist", "Hough Controls")
    param1 = cv2.getTrackbarPos("param1", "Hough Controls")
    param2 = cv2.getTrackbarPos("param2", "Hough Controls")
    minRadius = cv2.getTrackbarPos("minRadius", "Hough Controls")
    maxRadius = cv2.getTrackbarPos("maxRadius", "Hough Controls")
    blur_size = cv2.getTrackbarPos("blur", "Hough Controls")

    # GaussianBlur potrebuje nepárne a aspoň 1
    if blur_size < 1:
        blur_size = 1
    if blur_size % 2 == 0:
        blur_size += 1

    # ochrana proti neplatným hodnotám
    if dp < 0.1:
        dp = 0.1
    if minDist < 1:
        minDist = 1
    if param1 < 1:
        param1 = 1
    if param2 < 1:
        param2 = 1
    if maxRadius < minRadius:
        maxRadius = minRadius + 1

    return dp, minDist, param1, param2, minRadius, maxRadius, blur_size

def detect_polygons(gray, output):
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        if not cv2.isContourConvex(approx):
            continue

        # tazisko
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        vertices = len(approx)
        label = None
        color = (255, 255, 255)

        if vertices == 3:
            label = "triangle"
            color = (255, 0, 0)

        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue

            aspect_ratio = w / float(h)
            label = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
            color = (0, 165, 255)

        if label is not None:
            cv2.drawContours(output, [approx], -1, color, 2)
            cv2.putText(output, label, (cx - 40, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # vykreslenie taziska
            cv2.circle(output, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(output, f"T=({cx},{cy})", (cx + 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return edges

def main():
    calibration_path = "PVSO_zad_2/calibration_data.npz"

    camera_matrix, dist_coeff = load_calibration(calibration_path)
    cam, img = setup_camera()

    create_trackbars()

    # dp = 14
    # minDist = 300
    # param1 = 100
    # param2 = 30
    # minRadius = 44
    # maxRadius = 155
    # blur_size = 1
    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w // 4, h // 4))
        frame = cv2.undistort(frame, camera_matrix, dist_coeff)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        dp, minDist, param1, param2, minRadius, maxRadius, blur_size = get_trackbar_values()

        blur = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        output = frame.copy()

        # kruhy
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(output, "circle", (x - 20, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # dalsie tvary
        shape_edges = detect_polygons(gray, output)

        info = (
            f"dp={dp:.1f} minDist={minDist} "
            f"p1={param1} p2={param2} "
            f"rMin={minRadius} rMax={maxRadius} blur={blur_size}"
        )
        cv2.putText(output, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("frame", output)
        cv2.imshow("gray", gray)
        cv2.imshow("blur", blur)

        edges = cv2.Canny(blur, param1 // 2, param1)
        cv2.imshow("edges", edges)
        cv2.imshow("shape_edges", shape_edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    print("Stopping acquisition...")
    cam.stop_acquisition()
    cam.close_device()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()