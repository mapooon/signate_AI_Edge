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
import matplotlib.pyplot as plt
from tqdm import tqdm
from visdom import Visdom
import random
import sys
sys.path.append(os.path.join(os.getcwd(),"../common"))
import imagefolder
from iou import iou
import my_transform
from loss import SoftDiceLoss,CustomizedSoftDiceLoss,HardDiceLoss
import lovasz_losses as L
sys.path.append(os.path.join(os.getcwd(),"modeling"))
from model import UNetWithResnet50Encoder
import shutil
import argparse
import gc
#from aspp import ASPP



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        #inputs = inputs.contiguous()
        #targets = targets.contiguous()
        return self.nll_loss(F.log_softmax(inputs), targets)

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()

#@profile
def train_model(model,optimizer,dataloaders, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_iou = 0.0
    #class_weights=1/torch.Tensor(image_datasets['train'].get_label_ratio())
    cel = CrossEntropyLoss2d()#weight=(class_weights*((1024**2)/class_weights.sum())).cuda())#my_model.Weighted_BCELoss(pos_weight=[0.0062,1])
    # dice= SoftDiceLoss()
    # customdice=CustomizedSoftDiceLoss()
    # focal=FocalLoss(gamma=2)
    # tod_loss=nn.BCEWithLogitsLoss()
    # tunnel_loss=nn.BCEWithLogitsLoss()
    # hdice=HardDiceLoss()
    # Optimizerの第1引数には更新対象のfc層のパラメータのみ指定
    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.00001, last_epoch=-1)

    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    loss_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
    iou_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', "validation"]:
            if phase == 'train':
                scheduler.step()
                model.train(True)   # training mode
            else:
                model.train(False)  # evaluate mode

            running_loss = 0.0
            running_iou = 0#np.zeros(4)
            n_time=time.time()
            for step,data in enumerate(tqdm(dataloaders[phase])):#tqdm(dataloaders[phase]):
                inputs, labels = data
                if parsed.last_epoch>epoch:
                    break
                if use_gpu:
                    inputs=torch.Tensor(inputs).to(device)#.unsqueeze_(1)
                    labels = torch.Tensor(labels).to(device)
                    #tod_labels=torch.Tensor(tod_labels.float()).to(device)
                else:
                    inputs=torch.Tensor(inputs)#.unsqueeze_(1)
                    labels = torch.Tensor(labels)#.float()

                batch_size,n_input_channel,img_height,img_width=tuple(inputs.shape)
                optimizer.zero_grad()
                if phase=='train':
                    outputs=model(inputs)
                else:
                    with torch.no_grad():
                        outputs=model(inputs)
                # label_weight_sum=labels.sum(dim=(0,2,3))
                # label_weight_sum[label_weight_sum==0]=1
                # class_weights=1/label_weight_sum
                # if ip==3:#128.6
                #     loss = cel(outputs, labels.argmax(1))+customdice(outputs, labels)#AZ.log()#+0.1*tod_loss(tod_outputs[:,0],tod_labels[:,0])+0.1*tunnel_loss(tod_outputs[:,1],tod_labels[:,1])#cel(outputs, labels.argmax(1))+dice(outputs,labels).log()#weight=class_weights)(outputs, labels.argmax(1))
                # elif ip==5:#128.8
                #     loss = cel(outputs, labels.argmax(1))+L.lovasz_softmax(F.softmax(outputs), labels.argmax(1), per_image=True)
                # elif ip==2:#128.11
                #     loss=cel(outputs, labels.argmax(1))+dice(outputs, labels, per_image=True)#hdice(F.softmax(outputs),labels)
                # elif ip==4:#128.7
                #     loss=cel(outputs, labels.argmax(1))+dice(outputs, labels)
                # elif ip==7:#128.10
                #loss=0.25*cel(outputs, labels.argmax(1))
                if epoch<100:
                    loss=cel(outputs, labels.argmax(1))#+0.75*L.lovasz_softmax(F.softmax(outputs), labels.argmax(1), per_image=True)
                elif epoch<200:
                    loss=0.25*cel(outputs, labels.argmax(1))+0.75*L.lovasz_softmax(F.softmax(outputs), labels.argmax(1), per_image=True)
                else:
                    loss=L.lovasz_softmax(F.softmax(outputs), labels.argmax(1), per_image=True)
                       
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()#* batch_size
                running_iou += iou(outputs,labels).item()#,average=False)#.item()#*batch_size
                torch.cuda.empty_cache()
            if parsed.last_epoch>epoch:
                break
            # サンプル数で割って平均を求める
            epoch_loss = running_loss / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]
            epoch_iou = running_iou / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]

            print('{} Loss: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_iou))
            #print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            loss_dict[phase][epoch]=epoch_loss
            iou_dict[phase][epoch]=epoch_iou
            #visdom
            if phase=="validation":
                output_img=np.zeros((3,img_height,img_width))
                label_img=np.zeros((3,img_height,img_width))
                output_argmax=outputs[0].argmax(0)#(height,width)
                for idx,cla in enumerate(image_datasets["train"].category_list):
                    if idx==4:
                        break
                    #for y in range(img_height):
                        #for x in range(img_width):
                    #print(cla,labels[0,idx].cpu().data.numpy().sum(),np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
                    output_img+=((output_argmax==idx).float().cpu().data.numpy().reshape((1,img_height,img_width))*np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
                    # if cla=="car":
                    #     print(label_img)
                    #     print(labels[0,idx].sum().cpu().data.numpy())
                    #     print(image_datasets["train"].category_list[cla])
                    label_img+=(labels[0,idx].cpu().data.numpy().reshape((1,img_height,img_width))*np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))

                    #if idx==3:
                        #break
                #print(output_img.shape)
                win_output=viz.image(output_img/255,win="output",opts=dict(title='output'))
                win_label=viz.image(label_img/255,win="label",opts=dict(title='label'))
                win_input=viz.image(inputs[0].cpu().data.numpy(),win="input",opts=dict(title='input'))
                del output_img,label_img
                x_array=np.arange(epoch+1)
                if epoch>0:
                    x_array=np.arange(epoch+1)
                    viz.line(X=x_array,Y=loss_dict["train"][:epoch+1],update="replace",win="loss",name="train",opts=dict(title='loss'))
                    viz.line(X=x_array,Y=iou_dict["train"][:epoch+1],update="replace",win="iou",name="train",opts=dict(title='iou'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,0],update="replace",win="car",name="train",opts=dict(title='car'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,1],update="replace",win="signal",name="train",opts=dict(title='sig'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,2],update="replace",win="pedestrian",name="train",opts=dict(title='ped'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,3],update="replace",win="lane",name="train",opts=dict(title='lan'))
                    viz.line(X=x_array,Y=loss_dict["validation"][:epoch+1],update="replace",win="loss",name="validation",opts=dict(title='loss'))
                    viz.line(X=x_array,Y=iou_dict["validation"][:epoch+1],update="replace",win="iou",name="validation",opts=dict(title='iou'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,0],update="replace",win="car",name="validation",opts=dict(title='car'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,1],update="replace",win="signal",name="validation",opts=dict(title='sig'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,2],update="replace",win="pedestrian",name="validation",opts=dict(title='ped'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,3],update="replace",win="lane",name="validation",opts=dict(title='lan'))

                else:
                    win_loss=viz.line(X=x_array,Y=loss_dict["train"][:epoch+1],win="loss",name="train",opts=dict(title='loss'))
                    win_iou=viz.line(X=x_array,Y=iou_dict["train"][:epoch+1],win="iou",name="train",opts=dict(title='iou'))
                    # win_car=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,0],win="car",name="train",opts=dict(title='car'))
                    # win_sig=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,1],win="signal",name="train",opts=dict(title='sig'))
                    # win_ped=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,2],win="pedestrian",name="train",opts=dict(title='ped'))
                    # win_lan=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1,3],win="lane",name="train",opts=dict(title='lan'))
                    viz.line(X=x_array,Y=loss_dict["validation"][:epoch+1],win="loss",name="validation",opts=dict(title='loss'))
                    viz.line(X=x_array,Y=iou_dict["validation"][:epoch+1],win="iou",name="validation",opts=dict(title='iou'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,0],win="car",name="validation",opts=dict(title='car'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,1],win="signal",name="validation",opts=dict(title='sig'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,2],win="pedestrian",name="validation",opts=dict(title='ped'))
                    # viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1,3],win="lane",name="validation",opts=dict(title='lan'))
                del loss,outputs,win_output,win_label,win_input,output_argmax
                gc.collect()

            # deep copy the model
            # 精度が改善したらモデルを保存する
            if phase == "validation" and epoch_iou > 0.65:
                #print("save weights...",end="")
                #best_iou = epoch_iou#.mean()
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_path,"{}_{:.4f}_{:.4f}.pth".format(epoch,epoch_loss,epoch_iou)))
                #print("complete")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val iou: {:.4f}'.format(best_iou))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return #model


if __name__=="__main__":
    ip=int(os.uname()[1][2:])
    parser=argparse.ArgumentParser()
    parser.add_argument('-s',dest='seed',default=None)
    parser.add_argument('-w',dest='weight_name',default=None)
    parser.add_argument('-c',dest='use_clahe',action='store_true')
    parser.add_argument('-r',dest='use_resize',action='store_true')
    parser.add_argument('-l',dest='last_epoch',type=int,default=0)
    parsed=parser.parse_args()

    seeds={2:1,3:2,4:3,5:4,7:5}
    if parsed.seed:
        seed=parsed.seed
    else:
        seed=seeds[ip]
    print("seed: "+str(seed))
    #seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    current=os.getcwd()
    #shutil.rmtree(os.path.join(current,"result/ip_"+str(seed)))
    if parsed.use_resize:
        save_path=os.path.join(os.path.join(current,"result/resize/ip_"+str(ip)),str(seed))
    else:
        save_path=os.path.join(os.path.join(current,"result/normal/ip_"+str(ip)),str(seed))
    os.makedirs(save_path, exist_ok=True)
    viz = Visdom(port=8097,server="http://localhost")
    #data_dir="../../../data/double_input/"
    input_dir="../../../data/seg_train_images/"
    label_dir="../../../data/seg_train_annotations/"
    per_train=0.9
    batch_size=4
    #batch_size=16
    #num_sample=2242
    if parsed.use_resize:
        data_transforms = {
            'train': transforms.Compose([
                my_transform.Clahe(clipLimit=4,tileGridSize=(8,8)),
                my_transform.Resize(size=(512,1024)),
                #transforms.Resize((32,32),interpolation=Image.BILINEAR),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor()
                #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "validation": transforms.Compose([
                my_transform.Clahe(clipLimit=4,tileGridSize=(8,8)),
                my_transform.Resize(size=(512,1024)),
                #transforms.Resize((32, 32),interpolation=Image.BILINEAR),
                #transforms.CenterCrop(224),
                #transforms.ToTensor()
                #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                my_transform.Clahe(clipLimit=4,tileGridSize=(8,8))
                #my_transform.Resize(size=(512,1024)),
                #transforms.Resize((32,32),interpolation=Image.BILINEAR),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor()
                #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "validation": transforms.Compose([
                my_transform.Clahe(clipLimit=4,tileGridSize=(8,8))
                #my_transform.Resize(size=(512,1024)),
                #transforms.Resize((32, 32),interpolation=Image.BILINEAR),
                #transforms.CenterCrop(224),
                #transforms.ToTensor()
                #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
            ]),
        }
    target_transforms={
        'train': transforms.Compose([
            my_transform.Resize(size=(512,1024)),
        ]),
        "validation": transforms.Compose([
            my_transform.Resize(size=(512,1024)),
        ]),
    }

    #data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in["train","validation"]}

    image_datasets={x:imagefolder.EdgeDataset(input_dir=input_dir,
                                            label_dir=label_dir,
                                            input_transform=data_transforms[x],
                                            label_transform=target_transforms[x] if parsed.use_resize else None,
                                            mode=x,
                                            randcrop=(1024,1024) if x=='train' and not parsed.use_resize else None,
                                            n_class=4,
                                            seed=seed,
                                            randhflip=0.5 if x=='train' else None,
                                            per_train=per_train)
                    for x in ["train","validation"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    print(dataset_sizes)

    num_samples={'train':dataset_sizes['train']//2+batch_size-(dataset_sizes['train']//2)%batch_size,
                    'validation':dataset_sizes['validation']//2+batch_size-(dataset_sizes['validation']//2)%batch_size}
    # num_samples={'train':dataset_sizes['train'],
    #                 'validation':dataset_sizes['validation']}

    sample_weight={x:[1]*dataset_sizes[x] for x in ['train', 'validation']}
    data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in ["train","validation"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=batch_size,
                                              #shuffle=x=="train",
                                              sampler=data_sampler[x],
                                              worker_init_fn=np.random.seed(seed),
                                              num_workers=2) for x in ['train', 'validation']}

    use_gpu = torch.cuda.is_available()


    model=UNetWithResnet50Encoder()
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
    if parsed.weight_name:
        model.load_state_dict(torch.load(os.path.join(save_path,parsed.weight_name)))
    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.001)#, momentum=0.9)

    #model=train_model(model,optimizer,dataloaders=dataloaders,num_epochs=1)
    #for p in model.parameters():
    #    p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model,optimizer,dataloaders=dataloaders,num_epochs=300)
