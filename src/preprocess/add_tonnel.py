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
import pandas as pd
from tqdm import tqdm

label_dir="../../data/seg_train_annotations/"
tonnel_info_dir="../../data/tunnel.txt"
tonnel_info=pd.read_csv(tonnel_info_dir, sep=' ')
#print(tonnel_info)
json_list=sorted(glob(os.path.join(label_dir,"*.json")))
#print(len(json_list),json_list[len(json_list)-1])
#print(json_list)
day_count=0
morning_count=0
for idx,json_name in enumerate(tqdm(json_list)):
    with open(json_name) as f:
        a=json.load(f)
        if idx in tonnel_info.ix[:,0]:
            a["attributes"]['in_tunnel']=True
        else:
            a["attributes"]['in_tunnel']=False
        with open(json_name,'w') as fw:
            json.dump(a,fw)
night_count=len(json_list)-day_count-morning_count
print(morning_count,day_count,night_count)