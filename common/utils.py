# coding: utf-8

import os
from os.path import join, exists
import time
import cv2
import numpy as np
from .cnns import getCNNs


def logger(msg):
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def getDataFromTxt(txt, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    dirname = os.path.dirname(txt)  # 获取当前运行脚本的绝对路径
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()    # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
        components = line.split(' ')  # Python split()通过指定分隔符对字符串进行切片
        img_path = os.path.join(dirname, components[0].replace('\\', '/')) # file path   是在拼接路径 os.path.join(“home”, "me", "mywork") “home/me/mywork"
        # .replace  用 / 代替 \\

        # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
        bbox = [int(_) for _ in bbox]  #  把bbox里面的全都整形型
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):  # 0,1,2,3,4
            rv = (float(components[5+2*index]), float(components[5+2*index+1])) # 每隔2个跳一个。因为x后面是呀y，跳过了才是下一个x
            landmark[index] = rv
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))  #各个坐标点横纵坐标占框的比例
            landmark[index] = rv
        result.append((img_path, BBox(bbox), landmark))
    return result

def getPatch(img, bbox, point, padding):
    """
        Get a patch iamge around the given point in bbox with padding
        point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox


def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)   # 实现变量类型转换：
    for i, img in enumerate(imgs):
        m = img.mean()  # 表示对整个二维数组的平均，即全部加起来除以个数
        s = img.std()  # 求标准差
        imgs[i] = (img - m) / s
    return imgs

def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass

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
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])
