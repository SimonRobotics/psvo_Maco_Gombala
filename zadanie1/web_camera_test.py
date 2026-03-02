import cv2
import numpy as np

cam = cv2.VideoCapture(0)

num_of_img = 4

size_of_img = 240

images = np.zeros((480, 480, 3), dtype=np.uint8)

for x in range(0,num_of_img):
    ref, img = cam.read()

    i = (x%2)
    j = (x//2)

    i = i*size_of_img
    j = j*size_of_img

    i_end = i + size_of_img
    j_end = j + size_of_img

    images[i:i_end, j:j_end,:] = cv2.resize(img,(size_of_img,size_of_img))

kernel = np.ones((3, 3), np.int32)
kernel = kernel *-1
kernel[1,1] = 8

images[0:size_of_img,0:size_of_img,:] = cv2.filter2D(src=images[0:size_of_img,0:size_of_img,:], ddepth=-1, kernel=kernel)

for k in range(240,size_of_img+240):
    for l in range(0, size_of_img):

        images[l,(k*-1)+239,:] = images[k,l,:]

images[240:size_of_img+240,0:size_of_img,0] = 0 #B
images[240:size_of_img+240,0:size_of_img,1] = 0 #G
#images[240:size_of_img+240,0:size_of_img,2] = 255 #R

print("Tvar obrazu (height, width, channels):", images.shape)
print("Dátový typ pixelov:", images.dtype)
print("Celkový počet pixelov:", images.size)

cv2.imshow("test", images)
cv2.waitKey(0)
cv2.destroyAllWindows()

cam.release()


