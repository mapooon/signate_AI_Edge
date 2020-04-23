import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
import json
from numpy.random import *
from glob import glob
import cv2

image_path="../../data/seg_test_images/test_062.jpg"

bgr = cv2.imread(image_path)
org=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)#cv2.COLOR_BGR2YUV)#
#lab_planes = cv2.split(lab)

yuv=cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
#yuv_planes=cv2.split(yuv)

clahe = cv2.createCLAHE(clipLimit=4,tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
yuv[:,:,0] = clahe.apply(yuv[:,:,0])

#lab = cv2.merge(lab_planes)
lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)#cv2.COLOR_YUV2BGR)#
lab_rgb = cv2.cvtColor(lab_bgr, cv2.COLOR_BGR2RGB)

#yuv = cv2.merge(yuv_planes)
yuv_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)#
yuv_rgb = cv2.cvtColor(yuv_bgr, cv2.COLOR_BGR2RGB)



plt.subplot(2,2,1)
plt.imshow(lab_rgb)
plt.title("lab")

plt.subplot(2,2,2)
plt.imshow(yuv_rgb)
plt.title("yuv")

plt.subplot(2,2,3)
plt.imshow(org)
plt.title("org")

plt.show()