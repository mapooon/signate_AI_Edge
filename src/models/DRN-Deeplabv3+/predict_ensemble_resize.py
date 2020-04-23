import torch
from PIL import Image
from torchvision import models,datasets,transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import os
import time
import copy
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
from visdom import Visdom
import random
import sys
sys.path.append(os.path.join(os.getcwd(),"../common"))
import imagefolder
from iou import iou
import my_transform
sys.path.append(os.path.join(os.getcwd(),"modeling"))
from deeplab import DeepLab
import argparse
from glob import glob
#from aspp import ASPP

# seed=1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def get_weight_name(path):
    #print(path)
    weight_list = glob(os.path.join(path,'*.pth'))
    print(len(weight_list))
    best_iou=0
    for weight in weight_list:
        iou_score = float(weight.split('/')[-1].split('_')[-1][:-4])
        if iou_score>best_iou:
            best_iou=iou_score
            res=weight
    return res
#@profile
def predict(model,dataloader):
    batch_size=4
    since = time.time()
    with torch.no_grad():
        for step in tqdm(range(dataset_size//batch_size+int(dataset_size%batch_size>0))):
            for phase in ['drn','seresnext']:
                for data in dataloader[phase]:#tqdm(dataloaders[phase]):
                    inputs, data_idx = data
                    if use_gpu:
                        inputs=torch.Tensor(inputs).to(device)#.unsqueeze_(1)
                        
                    else:
                        inputs=Variable(inputs)#.unsqueeze_(1)
                        
                    
                    batch_size,n_input_channel,img_height,img_width=tuple(inputs.shape)
                    #outputs=torch.zeors()
                    for mc,model in enumerate(models[phase]):
                        if phase=='drn':
                            if mc==0:
                                outputs=F.softmax(F.interpolate(model(inputs),size=(1216,1936), mode='bilinear', align_corners=True))
                            else:
                                outputs+=F.softmax(F.interpolate(model(inputs),size=(1216,1936), mode='bilinear', align_corners=True))
                            outputs+=F.softmax(F.interpolate(model(inputs.flip(3)).flip(3),size=(1216,1936), mode='bilinear', align_corners=True))
                        else:
                            outputs+=F.softmax(model(inputs))
                            outputs+=F.softmax(model(inputs.flip(3)).flip(3))
                        
                    break
            for batch,jdx in enumerate(data_idx):
                output_img=np.zeros((3,img_height,img_width))
                label_img=np.zeros((3,img_height,img_width))
                output_argmax=outputs[batch].argmax(0)#(height,width)
                for idx,cla in enumerate(image_datasets[phase].category_list):
                    if idx==4:
                        break
                    #for y in range(img_height):
                        #for x in range(img_width):
                    #print(cla,labels[0,idx].cpu().data.numpy().sum(),np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
                    output_img+=((output_argmax==idx).float().cpu().data.numpy().reshape((1,img_height,img_width))*np.array(image_datasets[phase].category_list[cla]).reshape((3,1,1)))
                output_img=output_img.astype('uint8').transpose((1,2,0))
                img_name=os.path.basename(image_datasets[phase].images['image'][jdx]).replace('.jpg','.png')
                full_img_path=os.path.join(save_path,img_name)
                io.imsave(full_img_path,output_img)
                

            # deep copy the model
            # 精度が改善したらモデルを保存する
            print()

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return 


if __name__=="__main__":
    #parser=argparse.ArgumentParser()
    #parser.add_argument('-s',dest='seed',type=int)
    #parser.add_argument('-w',dest='weight_name',type=str)
    #parsed=parser.parse_args()
    #ip=parsed.seed
    #weight_name=parsed.weight_name
    device = torch.device('cuda')
    #ip=int(os.uname()[1][2:])
    current=os.getcwd()
    #weight_path=os.path.join(os.path.join(current,"result/ip_"+str(ip)),weight_name)
    save_path=os.path.join(current,"predict/ensemble")
    os.makedirs(save_path, exist_ok=True)
    #data_dir="../../../data/double_input/"
    input_dir="../../../data/seg_test_images/"
    batch_size=4
    #batch_size=16
    #num_sample=2242
    data_transforms = {'drn':transforms.Compose([
            my_transform.Clahe(clipLimit=4,tileGridSize=(8,8)),
            my_transform.Resize(size=(512,1024)),
            ]),
            'seresnext':transforms.Compose([
            my_transform.Clahe(clipLimit=4,tileGridSize=(8,8)),
            ])}
    image_datasets={'drn':imagefolder.EdgeDataset(input_dir=input_dir,
                                            #label_dir=label_dir,
                                            input_transform=data_transforms['drn'],
                                            #label_transform=data_transforms[x],
                                            mode='predict',
                                            #randcrop=(1024,1024),
                                            ),
                    'seresnext':imagefolder.EdgeDataset(input_dir=input_dir,
                                            #label_dir=label_dir,
                                            input_transform=data_transforms['seresnext'],
                                            #label_transform=data_transforms[x],
                                            mode='predict',
                                            #randcrop=(1024,1024),
                                            )
    }
                    
    dataset_size = len(image_datasets['drn'])
    print(dataset_size)
    dataloaders = {'drn':torch.utils.data.DataLoader(image_datasets['drn'],
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2),
                    'seresnext':torch.utils.data.DataLoader(image_datasets['seresnext'],
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)  
    }  

    use_gpu = torch.cuda.is_available()
    ips=['ip_2','ip_3','ip_4','ip_5','ip_7']
    models={'drn':[],'seresnext':[]}
    for i in range(5):
        weight_path=os.path.join(current,'result/resize/'+ips[i]+'/'+str(i+1))
        weight_name=get_weight_name(weight_path)
        model=DeepLab(num_classes=5,backbone='drn54').eval()
        #model.load_state_dict(torch.load(os.path.join(current,"pretrained_models/model_13_2_2_2_epoch_580.pth")))
        #model.aspp.conv_1x1_4 = nn.Conv2d(256, 5, kernel_size=1)
        """
        for idx,p in enumerate(model.parameters()):
            if idx!=0:
                p.requires_grad = False
        """
        if use_gpu:
            #torch.distributed.init_process_group(backend="nccl")
            model = nn.DataParallel(model).to(device)
            #model = model.cuda()
            #print(model.module)
        
        model.load_state_dict(torch.load(weight_name))
        models['drn'].append(model)
    for i in range(5):
        weight_path=os.path.join(current,'../SE-ResNext101-DeepLabV3+/result/clahe/'+ips[i]+'/'+str(i+1))
        weight_name=get_weight_name(weight_path)
        model=DeepLab(num_classes=5,backbone='seresnext50').eval()
        #model.load_state_dict(torch.load(os.path.join(current,"pretrained_models/model_13_2_2_2_epoch_580.pth")))
        #model.aspp.conv_1x1_4 = nn.Conv2d(256, 5, kernel_size=1)
        """
        for idx,p in enumerate(model.parameters()):
            if idx!=0:
                p.requires_grad = False
        """
        if use_gpu:
            #torch.distributed.init_process_group(backend="nccl")
            model = nn.DataParallel(model).to(device)
            #model = model.cuda()
            #print(model.module)
        
        model.load_state_dict(torch.load(weight_name))
        models['seresnext'].append(model)
    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.001)#, momentum=0.9)

    #model=train_model(model,optimizer,dataloaders=dataloaders,num_epochs=1)
    #for p in model.parameters():
    #    p.requires_grad = True

    predict(models,dataloader=dataloaders)