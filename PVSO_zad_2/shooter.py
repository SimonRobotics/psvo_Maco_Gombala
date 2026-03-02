import cv2
import numpy as np
from ximea import xiapi

cam = xiapi.Camera()

print('Opening first camera...')
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB24")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

i = 0
while True: 

    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(257,308))


    cv2.imshow("test", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite("img"+str(i)+".png", image)
        i+=1
