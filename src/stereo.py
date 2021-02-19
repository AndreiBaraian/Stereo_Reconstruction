import numpy as np
import cv2
from matplotlib import pyplot as plt


print('loading images...')

imgL = cv2.imread('../data/Motorcycle-imperfect/im0.png',0)
imgR = cv2.imread('../data/Motorcycle-imperfect/im1.png',0)

stereo = cv2.StereoBM_create(numDisparities=288, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

disparity_arr = np.array(disparity)
print(np.max(disparity_arr))

print('Done')
