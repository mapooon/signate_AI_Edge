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

label_dir="../../data/seg_train_annotations/train_1088.json"

json_list=sorted(glob(os.path.join(label_dir,"*.json")))
#print(json_list)
day_count=0
morning_count=0

with open(label_dir) as f:
    a=json.load(f)
    print(a)