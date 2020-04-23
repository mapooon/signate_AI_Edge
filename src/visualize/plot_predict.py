from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
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
import torch.nn.functional as F
import time
from collections import OrderedDict
from sklearn.model_selection import KFold
from tqdm import tqdm
#sn.set()

pred_path='../models/SE-ResNext101-DeeplabV3+/predict/ensemble'
image_path='../../data/seg_test_images'

pred_list=sorted(glob(os.path.join(pred_path,'*.png')))
image_list=sorted(glob(os.path.join(image_path,'*.jpg')))
for i in tqdm(range(len(pred_list))):
    pred=io.imread(pred_list[i])
    image=io.imread(image_list[i])

    #plt.subplot(2,1,1)


    #plt.subplot(2,1,2)
    plt.imshow(image,alpha=0.9)
    plt.imshow(pred,alpha=0.5)
    plt.axis('off')
    plt.title(pred_list[i].split('/')[-1])
    plt.savefig('overlaid/'+pred_list[i].split('/')[-1])
    plt.figure()
    #plt.show()