# -*- coding: UTF-8 -*-
import re
import sys
import os

xList = []
fa = open("C:\\Users\\tx\\Desktop\\sjj\\train\\trainImageList.txt", 'r')

for line in fa.readlines():

#test
    str2 = line.split(' ')[-10]
    a = float(str2)
    str1 = line.split(' ')[-8]
    b = float(str1)
    str3 = line.split(' ')[-6]
    c=float(str3)
    d=abs(a-c)
    e=abs(b-c)
    f= abs(d-e)
    if  f<5:
     xList.append(line)
fb = open("C:\\Users\\tx\\Desktop\\sjj\\train\\B.txt", 'w')
for i in xList:
    fb.write(i)

fa.close()
fb.close()
