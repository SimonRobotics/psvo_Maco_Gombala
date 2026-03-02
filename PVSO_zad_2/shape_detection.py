import cv2
import numpy as np
from ximea import xiapi

def centroid_of_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def classify_shape(contour, approx, frame_area):
    area = cv2.contourArea(contour)
    if area < 400:  # šum 
        return None, None
    if area > 0.95 * frame_area:
        return None, None

    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None, None

    circularity = 4.0 * np.pi * area / (peri * peri)

    vertices = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h) if h != 0 else 0

    # kruh: vysoká circularity + viac bodov
    if circularity > 0.80 and vertices >= 6:
        return "kruznica", {"circularity": circularity}

    if vertices == 3:
        return "trojuholnik", None

    if vertices == 4:
        if 0.92 <= aspect_ratio <= 1.08:
            return "stvorec", {"ar": aspect_ratio}
        return "obdlznik", {"ar": aspect_ratio}

    return None, None

def main():
    cam = xiapi.Camera()
    img = xiapi.Image()

    try:
        print("Opening Ximea camera...")
        cam.open_device()

        # Nastavenia 
        cam.set_exposure(50000)  # us
        cam.set_param("imgdataformat", "XI_RGB24")  # 3 kanály
        cam.set_param("auto_wb", 1)

        print(f"Exposure: {cam.get_exposure()} us")
        cam.start_acquisition()
        print("Acquisition started. ESC / q ukončí program.")

        cv2.namedWindow("shapes", cv2.WINDOW_NORMAL)
        cv2.namedWindow("edges", cv2.WINDOW_NORMAL)

        kernel_close = np.ones((3, 3), np.uint8)

        while True:
            cam.get_image(img)
            frame_rgb = img.get_image_data_numpy()          # HxWx3 (RGB)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # OpenCV-friendly

            frame_area = frame.shape[0] * frame.shape[1]

            # --- Predspracovanie ---
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

            # hrany
            edges = cv2.Canny(gray, 60, 160)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # kontúry
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue

                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                shape_name, info = classify_shape(cnt, approx, frame_area)
                if shape_name is None:
                    continue

                c = centroid_of_contour(cnt)
                if c is None:
                    continue

                # kreslenie
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                cv2.circle(frame, c, 4, (0, 0, 255), -1)

                label = shape_name
                if info and "circularity" in info:
                    label += f" (circ={info['circularity']:.2f})"
                if info and "ar" in info:
                    label += f" (ar={info['ar']:.2f})"

                cv2.putText(frame, label, (c[0] + 8, c[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("edges", edges)
            cv2.imshow("shapes", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC alebo q
                break

    finally:
        try:
            cam.stop_acquisition()
        except Exception:
            pass
        try:
            cam.close_device()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()