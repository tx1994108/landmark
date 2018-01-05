#!/usr/bin/env python2.7
# coding: utf-8

import os
from os.path import join, exists
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt



def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)   # 实现变量类型转换：

    for i, img in enumerate(imgs):
       # print(i,img)
        m = img.mean()  # 表示对整个二维数组的平均，即全部加起来除以个数
        print('tx_test1_m : ',m)
        s = img.std()  # 求标准差
        print('tx_test1_s : ', s)
        imgs[i] = (img - m) / s
       # print(imgs[i])

    return imgs

class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]                 #(0,2)   (1,2)
        self.h = bbox[3] - bbox[2]                 #(0,3)   (1,3)

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y]) # numpy.asarray(a,dtype=None,order=None) 将输入数据（列表的列表，元组的元组，元组的列表等）转换为矩阵形式

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        #print(leftDelta)
        rightDelta = self.w * rightR
        #print(leftDelta)
        topDelta = self.h * topR
        #print(topDelta)
        bottomDelta = self.h * bottomR
        #print(bottomDelta)
        left = self.left + leftDelta
        #print(left)
        right = self.left + rightDelta
        #print(right)
        top = self.top + topDelta
        #print(top)
        bottom = self.top + bottomDelta
        #print(bottom )

        return BBox([left, right, top, bottom])


#--------------------------------------------------------------------------
with open('C:\\Users\\tx\\Desktop\\sjj\\train\\11.txt', 'r') as fd:
    lines = fd.readlines()
    #print (lines)
result = []
for line in lines:
        line = line.strip()    # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
       # print(line)
        components = line.split(' ')  # Python split()通过指定分隔符对字符串进行切片
          # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
       # print (bbox[0],bbox[1],bbox[2],bbox[3])
        bbox = [int(_) for _ in bbox]  #  把bbox里面的全都整形型
        #print (bbox[0],bbox[1],bbox[2],bbox[3])
#         # landmark
#         if not with_landmark:
#             result.append((img_path, BBox(bbox)))
#             continue
        landmark = np.zeros((5, 2))
        #print (landmark)
        for index in range(0, 5):  # 0,1,2,3,4
            rv = (float(components[5+2*index]), float(components[5+2*index+1])) # 每隔2个跳一个。因为x后面是呀y，跳过了才是下一个x
            landmark[index] = rv
      #  print(landmark)
        for index, one in enumerate(landmark):
           # print (one[0],' ',bbox[0],' ',bbox[1],' ',bbox[0],' ',one[1],' ',bbox[2],' ',bbox[3],' ',bbox[2],' ')
            rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))
            landmark[index] = rv
            # print (landmark[index])
result.append(("C:\\Users\\tx\\Desktop\\sjj\\train\\Aaron_Eckhart_0001.jpg",BBox(bbox),landmark))
print (len(result))

print (result)
error = np.zeros((len(result), 5))
#print(error)


for i in range(len(result)):
    imgPath, bbox, landmarkGt = result[i]
   # print (result)
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    # print(result[i])
    cv2.line(img, (bbox.left, bbox.top), (bbox.right, bbox.top), (255, 0, 0), 3)
    cv2.line(img, (bbox.left, bbox.top), (bbox.left, bbox.bottom),(255, 0, 0), 3)
    cv2.line(img, (bbox.right, bbox.top),(bbox.right,bbox.bottom),(255, 0, 0),3)
    cv2.line(img, (bbox.left, bbox.bottom),(bbox.right, bbox.bottom),(255,0, 0),3)
    cv2.imshow('image1', img)



f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)

cv2.line(img, (bbox.left, bbox.top), (bbox.right, bbox.top), (255, 0, 0), 3)
cv2.line(img, (bbox.left, bbox.top), (bbox.left, bbox.bottom),(255, 0, 0), 3)
cv2.line(img, (bbox.right, bbox.top),(bbox.right,bbox.bottom),(255, 0, 0),3)
cv2.line(img, (bbox.left, bbox.bottom),(bbox.right, bbox.bottom),(255,0, 0),3)
#cv2.imshow('image2', img)

f_face = img[int(f_bbox.top):int(f_bbox.bottom+1),int(f_bbox.left):int(f_bbox.right+1)]
cv2.imshow('image3', f_face)
face_flipped_by_x = cv2.flip(f_face, 1)

# landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])              # 输出图形 旋转变换后的。。。
# print('landmark:',landmark,     '\n'  ,         'landmark_',landmark_)

#print(f_bbox.top,' ',f_bbox.bottom+1,' ',f_bbox.left,' ',f_bbox.right+1 )
#print(f_face.shape)
cv2.imshow('image4', face_flipped_by_x)
cv2.waitKey(0)
f_face = cv2.resize(f_face, (39, 39))         #正脸！！！！！
# print(f_face)
# cv2.imshow('image4', f_face)
# cv2.waitKey(0)

en_face = f_face[:31, :]
# cv2.imshow('image3', en_face)          #   眼睛鼻子
# en_face = cv2.resize(en_face, (31, 39))
# cv2.imshow('image4', en_face)
#cv2.waitKey(0)
nm_face = f_face[8:, :]         # 鼻子嘴巴
#print(nm_face.shape)
# cv2.imshow('image6', nm_face)
#cv2.waitKey(0)
f_face = f_face.reshape((1, 1, 39, 39))
# print(f_face.shape)


f_face = processImage(f_face)
# print(f_face)                     #正脸做过处理
# print(f_face.shape)
# a=f_face[0][0]
# print(a)
# cv2.imshow('image5', a)
# cv2.waitKey(0)

# en_face = cv2.resize(en_face, (31, 39)).reshape((1, 1, 31, 39))    #眼睛鼻子做过处理
# en_face = processImage(en_face)
# b=en_face[0][0]
# print(b)
# print(b.shape)
# cv2.imshow('image5', b)
# cv2.waitKey(0)

nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 1, 31, 39))
nm_face = processImage(nm_face)
c=nm_face[0][0]
# cv2.imshow('image5', c)
# cv2.waitKey(0)

