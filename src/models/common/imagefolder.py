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
import my_transform
from sklearn.model_selection import KFold

class EdgeDataset(Dataset):
    def __init__(self,input_dir,seed=None,label_dir=None,input_transform=None,label_transform=None,mode="train",per_train=0.9,randcrop=None,n_class=4,randhflip=None):
        """
        root_dir:
        """
        self.input_dir=input_dir
        self.label_dir=label_dir
        self.input_transform=input_transform
        self.label_transform=label_transform
        self.mode=mode
        if randcrop:
            self.randcrop=my_transform.RandomCrop(randcrop,up_cut=0,down_cut=0,right_cut=0,left_cut=0)
        else:
            self.randcrop=None
        if randhflip:
            self.randhflip=my_transform.RandomHorizontalFlip(p=randhflip)
        else:
            self.randhflip=None
        all_image_list=sorted(glob(os.path.join(input_dir,"*.jpg")))
        print(len(all_image_list))
        kf = KFold(n_splits=5)
        if mode=="train":
            #image_list=all_image_list[:int(per_train*len(all_image_list))]
            image_list=[all_image_list[i] for i in list(kf.split(all_image_list))[seed-1][0]]
        elif mode=="validation":
            image_list=[all_image_list[i] for i in list(kf.split(all_image_list))[seed-1][1]]
        elif mode=="predict":
            image_list=all_image_list

        print("number of sample: "+str(len(image_list)))

        if self.mode!="predict":
            all_label_list=sorted(glob(os.path.join(label_dir,"*.png")))
        if self.mode=="train":
            label_list=[all_label_list[i] for i in list(kf.split(all_label_list))[seed-1][0]]
        elif self.mode=="validation":
            label_list=[all_label_list[i] for i in list(kf.split(all_label_list))[seed-1][1]]
        elif self.mode=="predict":
            label_list=None#all_label_list

        self.images={"image":image_list,"label":label_list}

        self.category_list = OrderedDict([('car',[0,0,255]),
                 ('signal',[255,255,0]),
                 ('pedestrian',[255,0,0]),
                 ('lane',[69,47,142]),
                 ('bus',[193, 214, 0]),
                 ('truck',[180,0,129]),
                 ('motorbike',[65,166,1]),
                 ('bicycle',[208,149,1]),
                 ('svehicle',[255,121,166]),
                 ('signs',[255,134,0]),
                 ('sky',[0,152,225]),
                 ('building',[0,203,151]),
                 ('natural',[85,255,50]),
                 ('wall',[92,136,125]),
                 ('ground',[136,45,66]),
                 ('sidewalk',[0,255,255]),
                 ('roadshoulder',[215,0,255]),
                 ('obstacle',[180,131,135]),
                 ('others',[81,99,0]),
                 ('own',[86,62,67])])
        # if use_fullclass:
        #     self.n_class=len(self.category_list)
        # else:    
        self.n_class=n_class
        self.class_dict={tuple(self.category_list[key]):idx for idx,key in enumerate(self.category_list)}
        self.class_array=np.array([np.nan]*(256**3)).reshape((256,256,256)).astype("uint8")
        for idx,key in enumerate(self.category_list):
            x1,x2,x3=tuple(self.category_list[key])
            self.class_array[x1,x2,x3]=idx 
        #print(self.class_array)
    
    def get_class_map(self,image):
        height=image.shape[0]
        width=image.shape[1]
        class_map=np.zeros((height,width,self.n_class))
        for y in range(height):
            for x in range(width):
                class_map[y,x,self.class_dict[tuple(image[y,x,:])]]=1
        return class_map

    def get_class_map2(self,image):
        height=image.shape[0]
        width=image.shape[1]
        class_map=np.zeros((height,width))
        for y in range(height):
            for x in range(width):
                class_map[y,x]=self.class_dict[tuple(image[y,x,:])]
        return class_map

    def get_class_map3(self,label):
        """
        label:(height,width,3)
        """
        height=label.shape[0]
        width=label.shape[1]
        class_map=np.zeros((height,width,self.n_class+1))
        for idx,cla in enumerate(self.category_list):
            if idx==self.n_class:
                break
            x1,x2,x3=tuple(self.category_list[cla])
            #true_idx=np.ones((height,width)).astype('uint8')
            #for i in range(3):
                #true_idx*=label[:,:,i]==self.category_list[cla][i]
            x1_idx=label[:,:,0]==x1
            x2_idx=label[:,:,1]==x2
            x3_idx=label[:,:,2]==x3
            true_idx=x1_idx*x2_idx*x3_idx
            class_map[true_idx,idx]=1
        #other classのラベル付け
        class_map[class_map.sum(2)==0,self.n_class]=1
            #print(label[x,y])
            #x1,x2,x3=tuple(label[y,x])
            #if (x1,x2,x3)==(0,0,255):
            #   print("car")
            #print(self.class_array[x1,x2,x3])
            #class_map[y,x,self.class_array[x1,x2,x3]]=1
        return class_map

    def __len__(self):
        return len(self.images["image"])

    def __getitem__(self,idx):

        if self.mode!='predict':
            return self.getitem_train_val(idx)
        else:
            return self.getitem_predict(idx)
        #print(str(idx)+"...",flush=True,end="")
        # img_name=self.images["image"][idx]
        # label_name=self.images["label"][idx]
        # #print("load image...",end="")
        # image=io.imread(img_name)#.astype("float32")#.transpose(2,0,1)
        # if self.input_transform:
        #     image=self.input_transform(image)
        # image=image/255
        # label=(io.imread(label_name)).astype("uint8")#.transpose(2,0,1)
        # if self.randcrop:
        #     image,label=self.randcrop(image,label)
        
        # image=image.transpose(2,0,1).astype("float32")
        # label=self.get_class_map3(label).astype("float32").transpose(2,0,1)
        
        
        # return (image,label)

    def getitem_train_val(self,idx):
        img_name=self.images["image"][idx]
        label_name=self.images["label"][idx]
        #print("load image...",end="")
        image=io.imread(img_name)#.astype("float32")#.transpose(2,0,1)
        if self.input_transform:
            image=self.input_transform(image)
        
        image=image/255
        label=(io.imread(label_name)).astype("uint8")#.transpose(2,0,1)
        if self.label_transform:
            label=self.label_transform(label)
        if self.randcrop:
            image,label=self.randcrop(image,label)
        if self.randhflip:
            image,label=self.randhflip(image,label)
        
        image=image.transpose(2,0,1).astype("float32")
        label=self.get_class_map3(label).astype("float32").transpose(2,0,1)
        
        
        return image,label

    def getitem_predict(self,idx):
        img_name=self.images["image"][idx]
        #print("load image...",end="")
        image=io.imread(img_name)#.astype("float32")#.transpose(2,0,1)
        if self.input_transform:
            image=self.input_transform(image)
        image=image/255
        
        image=image.transpose(2,0,1).astype("float32")
        
        return image,np.array([idx])
    
    def get_tod_info(self,idx):
        json_name=self.images["label"][idx].replace('.png','.json')
        with open(json_name) as f:
            a=json.load(f)
            #print(a["attributes"]["timeofday"]=='night',a["attributes"]["in_tonnel"])
            return np.array([a["attributes"]["timeofday"]=='night',a["attributes"]["in_tunnel"]]).astype('float64')

    def get_label_ratio(self):
        ratio_array=np.ones(self.n_class+1)
        for idx in range(self.__len__()):
            label_name=self.images["label"][idx]
            label=(io.imread(label_name)).astype("uint8")
            label=self.get_class_map3(label).astype("float32").transpose(2,0,1)
            ratio_array+=label.sum((1,2))
        return ratio_array
    # def __getitem__(self,idx):
    #     #print(str(idx)+"...",flush=True,end="")
    #     sample_idx=int(idx/16)
    #     pos=idx-sample_idx*16
    #     x_pos=pos%4
    #     y_pos=int(pos/4)
    #     img_name=self.images["image"][sample_idx]
    #     label_name=self.images["label"][sample_idx]
    #     #print("load image...",end="")
    #     image=(io.imread(img_name)/255).astype("float32")#.transpose(2,0,1)
    #     label=(io.imread(label_name)).astype("uint8")#.transpose(2,0,1)
    #     height=int(image.shape[0]/4)
    #     width=int(image.shape[1]/4)
    #     #image=Image.fromarray(image)
    #     #print("complete")
    #     start_time=time.time()
        
    #     act_time=time.time()-start_time
    #     #print("gen label time:"+str(act_time),flush=True)
    #     image=image[height*y_pos:height*(y_pos+1),width*x_pos:width*(x_pos+1),:].transpose(2,0,1)#.float()#.cuda()
    #     #image=image.cuda()
    #     label=label[height*y_pos:height*(y_pos+1),width*x_pos:width*(x_pos+1),:]#.transpose(2,0,1)#.float()#.cuda()
    #     label=self.get_class_map3(label).astype("float32").transpose(2,0,1)
    #     #label=label.cuda()

    #     # if self.input_transform is not None:
    #     #     image=self.input_transform(image)
    #     # if self.label_transform is not None:
    #     #     label=self.label_transform(label)
    #     #print("generate label...",end="")
    #     #print("complete")
    #     #print(image.shape,label.shape)
    #     #return torch.Tensor(image),torch.Tensor(label)
    #     #print("complete",flush=True)
    #     return (image,label)