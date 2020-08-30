import cv2
import numpy as np
k=1
b=2
x1=0
y1=k*x1+b
x2=500
y2=k*x2+b
pt1=(x1,y1)
pt2=(x2,y2)
img=np.zeros((1000,1000,3),dtype=np.uint8)
cv2.line(img,pt1,pt2,(0,0,255),thickness=2)
cv2.imshow('test',img)
cv2.waitKey(0)
