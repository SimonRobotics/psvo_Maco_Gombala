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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dp, minDist, param1, param2, minRadius, maxRadius, blur_size = get_trackbar_values()

        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 2)

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

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

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

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    print("Stopping acquisition...")
    cam.stop_acquisition()
    cam.close_device()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()