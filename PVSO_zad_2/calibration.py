import cv2
import numpy as np
import glob


imgs = glob.glob('./*png')

CHECKERBOARD = (7,5)

objpoints = []

imgpoints = [] 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

for file in imgs:
    img = cv2.imread(file)

    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    retval, corners = cv2.findChessboardCorners(img, [7,5], None)

    if retval:
        
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(img_grey, corners, (11,11),(-1,-1), criteria)

        stefan = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, retval)

        imgpoints.append(corners2)
    # cv2.imshow('img',stefan)
    # cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_grey.shape[::-1], None, None)
 
print("\n===== VÝSLEDKY KALIBRÁCIE =====\n")

print(f"RMS chyba (reprojection error): {ret:.6f} PX\n")

print("Camera matrix (intrinsic parameters):")
print(f"fx = {mtx[0,0]:.4f}")
print(f"fy = {mtx[1,1]:.4f}")
print(f"cx = {mtx[0,2]:.4f}")
print(f"cy = {mtx[1,2]:.4f}\n")

print("Distortion coefficients:")
for i, d in enumerate(dist[0]):
    print(f"k{i+1} = {d:.6f}")

print("\nPočet snímok použitých na kalibráciu:", len(rvecs))

np.savez("calibration_data.npz",
         camera_matrix=mtx,
         dist_coeff=dist,
         rvecs=rvecs,
         tvecs=tvecs)

print("Kalibrácia uložená do calibration_data.npz")


# img = cv2.imread("./img4.png")
# h, w = img.shape[:2]

# new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# und = cv2.undistort(img, mtx, dist, None, new_mtx)

# cv2.imshow("orig", img)
# cv2.imshow("undistorted", und)
# cv2.waitKey(0)
# cv2.destroyAllWindows()