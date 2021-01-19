import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('../data/Motorcycle-perfect/im0.png',0)
imgR = cv.imread('../data/Motorcycle-perfect/im1.png',0)
plt.imshow(imgL)
plt.show()
stereo = cv.StereoBM_create(numDisparities=256, blockSize=15)
print(len(imgL.shape))
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()