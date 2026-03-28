import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_dog_pyramid(img, sigmas):
    blurred = []
    for s in sigmas:
        b = cv2.GaussianBlur(img, (0, 0), s)
        blurred.append(b.astype(np.float32))

    dogs = []
    for i in range(len(blurred) - 1):
        dogs.append(blurred[i + 1] - blurred[i])

    return dogs

def find_local_extrema(dogs, threshold=5):
    keypoints = []

    for s in range(1, len(dogs) - 1):
        prev_dog = dogs[s - 1]
        curr_dog = dogs[s]
        next_dog = dogs[s + 1]

        h, w = curr_dog.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                value = curr_dog[y, x]

                if abs(value) < threshold:
                    continue

                neighborhood = []

                neighborhood.extend(prev_dog[y-1:y+2, x-1:x+2].flatten())
                neighborhood.extend(curr_dog[y-1:y+2, x-1:x+2].flatten())
                neighborhood.extend(next_dog[y-1:y+2, x-1:x+2].flatten())

                neighborhood = np.array(neighborhood)

                center_index = 13
                neighbors = np.delete(neighborhood, center_index)

                if value > neighbors.max() or value < neighbors.min():
                    keypoints.append((x, y, s, abs(value)))

    return keypoints

def select_strongest_points(keypoints, max_points=200):
    keypoints = sorted(keypoints, key=lambda k: k[3], reverse=True)
    return keypoints[:max_points]

def draw_keypoints(img, keypoints, color=(0, 0, 255)):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y, s, response in keypoints:
        cv2.circle(out, (x, y), 6, color, -1)  # -1 = vyplnený kruh

    return out

def detect_sift(img):
    sift = cv2.SIFT_create(nfeatures=200)
    kps = sift.detect(img, None)
    return kps

def draw_sift_keypoints(img, keypoints):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)

    return out

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(base_dir, "images", "toto.jpg")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Chyba: obrazok sa nepodarilo nacitat.")
        return

    sigmas = [1.0, 1.6, 2.2, 2.8, 3.4]
    dogs = build_dog_pyramid(img, sigmas)
    
    my_kps = find_local_extrema(dogs, threshold=5)
    my_kps = select_strongest_points(my_kps, max_points=200)

    my_img = draw_keypoints(img, my_kps)

    sift_kps = detect_sift(img)
    # sift_img = cv2.drawKeypoints(img, sift_kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_img = draw_sift_keypoints(img, sift_kps)

    plt.figure(figsize=(18, 6))

    # 1. povodny obrazok
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Povodny obrazok")
    plt.axis("off")

    # 2. tvoja detekcia
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Moja detekcia: {len(my_kps)} bodov")
    plt.axis("off")

    # 3. SIFT
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    plt.title(f"OpenCV SIFT: {len(sift_kps)} bodov")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()