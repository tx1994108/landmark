import re
import sys
import os
import cv2

str=[]

fa = open("C:\\Users\\tx\\Desktop\\sjj\\train\\B.txt", 'r')
for line in fa.readlines():
    str2 = line.split(' ')[-15]
    str.append(str2)

for i in range(0,len(str)):
    a=str[i]
    b='C:\\Users\\tx\\Desktop\\sjj\\train\\'+a
    img = cv2.imread(b)
    c='C:\\Users\\tx\\Desktop\\sjj\\train\\11\\'+a
    print(c)
    cv2.imwrite(c, img)


