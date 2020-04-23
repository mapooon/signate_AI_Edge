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
import pydensecrf.densecrf as dcrf
#from aspp import ASPP

# seed=1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 5
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))

def get_weight_name(path):
    #print(path)
    weight_list = glob(os.path.join(path,'*.pth'))
    #print(len(weight_list))
    best_iou=0
    for weight in weight_list:
        iou_score = float(weight.split('/')[-1].split('_')[-1][:-4])
        if iou_score>best_iou:
            best_iou=iou_score
            res=weight
    return res
#@profile
def predict(model,dataloader):
    since = time.time()
    with torch.no_grad():

        for step,data in enumerate(tqdm(dataloader)):#tqdm(dataloaders[phase]):
            inputs, data_idx = data
            if use_gpu:
                inputs=torch.Tensor(inputs).to(device)#.unsqueeze_(1)
                
            else:
                inputs=Variable(inputs)#.unsqueeze_(1)
                
            
            batch_size,n_input_channel,img_height,img_width=tuple(inputs.shape)
            #outputs=torch.zeors()
            for mc,model in enumerate(models):
                with torch.no_grad():
                    if mc==0:
                        outputs=model(inputs).cpu().data.numpy()
                        sig=outputs.argmax(1)==1
                        ped=outputs.argmax(1)==2
                    else:
                        outputs+=model(inputs).cpu().data.numpy()
                        sig+=outputs.argmax(1)==1
                        ped+=outputs.argmax(1)==2
                    outputs+=model(inputs.flip(3)).flip(3).cpu().data.numpy()
                    sig+=outputs.argmax(1)==1
                    ped+=outputs.argmax(1)==2
            #outputs=F.softmax(outputs).cpu().data.numpy()
            outputs[:,1]=sig>0
            outputs[:,2]=ped>0
            for batch,jdx in enumerate(data_idx):
                output_img=np.zeros((3,img_height,img_width))
                label_img=np.zeros((3,img_height,img_width))
                output_argmax=outputs[batch].argmax(0)#(height,width)
                for idx,cla in enumerate(image_dataset.category_list):
                    if idx==4:
                        break
                    #for y in range(img_height):
                        #for x in range(img_width):
                    #print(cla,labels[0,idx].cpu().data.numpy().sum(),np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
                    output_img+=((output_argmax==idx).reshape((1,img_height,img_width))*np.array(image_dataset.category_list[cla]).reshape((3,1,1)))
                output_img=output_img.astype('uint8').transpose((1,2,0))
                img_name=os.path.basename(image_dataset.images['image'][jdx]).replace('.jpg','.png')
                full_img_path=os.path.join(save_path,img_name)
                io.imsave(full_img_path,output_img)
                

            # deep copy the model
            # 精度が改善したらモデルを保存する
            print()

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return 


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-c',dest='use_crf',action='store_true')
    #parser.add_argument('-s',dest='seed',type=int)
    #parser.add_argument('-w',dest='weight_name',type=str)
    parsed=parser.parse_args()
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
    data_transform = transforms.Compose([
            my_transform.Clahe(clipLimit=4,tileGridSize=(8,8))
            #transforms.Resize((32,32),interpolation=Image.BILINEAR),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    #data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in["train","validation"]}

    image_dataset=imagefolder.EdgeDataset(input_dir=input_dir,
                                            #label_dir=label_dir,
                                            input_transform=data_transform,
                                            #label_transform=data_transforms[x],
                                            mode='predict',
                                            #randcrop=(1024,1024),
                                            )
                    
    dataset_size = len(image_dataset)
    print(dataset_size)

    #num_samples={x:dataset_sizes[x]//20+batch_size-(dataset_sizes[x]//20)%batch_size for x in ['train', 'validation']}
    #sample_weight={x:[1]*dataset_sizes[x] for x in ['train', 'validation']}
    #data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in ["train","validation"]}

    dataloader = torch.utils.data.DataLoader(image_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)

    use_gpu = torch.cuda.is_available()
    ips=['ip_2','ip_3','ip_4','ip_5','ip_7']
    models=[]
    for i in range(5):
        weight_path=os.path.join(current,'result/clahe/'+ips[i]+'/'+str(i+1))
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
        models.append(model)
    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.001)#, momentum=0.9)

    #model=train_model(model,optimizer,dataloaders=dataloaders,num_epochs=1)
    #for p in model.parameters():
    #    p.requires_grad = True

    predict(models,dataloader=dataloader)