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

label_dir="../../data/seg_train_annotations/"

json_list=sorted(glob(os.path.join(label_dir,"*.json")))
#print(json_list)
day_count=0
morning_count=0
for json_name in json_list:
    with open(json_name) as f:
        a=json.load(f)
        if a["attributes"]["timeofday"]=="day":# or a["attributes"]["timeofday"]=="morning":
            day_count+=1
        elif a["attributes"]["timeofday"]=="morning":
            morning_count+=1
night_count=len(json_list)-day_count-morning_count
print(morning_count,day_count,night_count)