from skimage import io, transform
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
from tqdm import tqdm
import cv2

label_dir="../../data/seg_train_annotations/"
image_dir="../../data/seg_train_images/"

json_list=sorted(glob(os.path.join(label_dir,"*.json")))
image_list=sorted(glob(os.path.join(image_dir,"*.jpg")))
#print(json_list)
day_count=0
morning_count=0
sum_array=[0]*len(json_list)
label_array=[0]*len(json_list)
color_array=[""]*len(json_list)
for idx in tqdm(range(len(json_list))):
    with open(json_list[idx]) as f:
        a=json.load(f)
        bgr = cv2.imread(image_list[idx])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        sum_array[idx]=lab[:,:,0].sum()
        label_array[idx]=int(a["attributes"]["timeofday"]!="night")
        color_array[idx]="r" if a["attributes"]["timeofday"]!="night" else "b"

plt.scatter(label_array,sum_array,label=label_array)        
plt.show()