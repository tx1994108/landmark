import numpy as np
import cv2
import os
#img=cv2.imread('C:\\Users\\tx\\Desktop\\数据集\\train\\Aaron_Eckhart_0001.jpg')
img=cv2.imread('C:\\Users\\tx\\Desktop\\sjj\\train\\Aaron_Eckhart_0001.jpg')
cv2.rectangle(img,(80,90),(160,170),(0,255,0),2)
cv2.circle(img, (106, 107), 10, (0, 0, 255), 1)
cv2.circle(img, (146, 112), 10, (0, 0, 255), 1)
cv2.circle(img, (125, 142), 10, (0, 0, 255), 1)
cv2.circle(img, (105, 157), 10, (0, 0, 255), 1)
cv2.circle(img, (139, 161), 10, (0, 0, 255), 1)
cv2.circle(img, (84, 92), 10, (0, 255, 255), 1)
cv2.circle(img, (161, 169), 10, (0, 255, 255), 1)
cv2.imshow('image',img)
cv2.waitKey(0)
# 123
