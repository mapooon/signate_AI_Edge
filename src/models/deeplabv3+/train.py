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
sys.path.append(os.path.join(os.getcwd(),"modeling"))
from deeplab import DeepLab
#from aspp import ASPP

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

#@profile
def train_model(model,optimizer,dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    #criterion = nn.CrossEntropyLoss()#weight=class_weights)#my_model.Weighted_BCELoss(pos_weight=[0.0062,1])

    # Optimizerの第1引数には更新対象のfc層のパラメータのみ指定
    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.01, momentum=0.9)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    loss_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
    iou_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', "validation"]:
            if phase == 'train':
                #scheduler.step()
                model.train(True)   # training mode
            else:
                model.train(False)  # evaluate mode

            running_loss = 0.0
            running_iou = 0
            n_time=time.time()
            for data in tqdm(dataloaders[phase]):#tqdm(dataloaders[phase]):
                #print("load image...",end="", flush=True)
                act_time4=time.time()-n_time
                #print("load time:"+str(act_time4),flush=True)
                inputs, labels = data
                #print(inputs.shape,flush=True)
                #print(inputs)
                #print("complete", flush=True)
                #print(torch.max(inputs))
                #print(np.sum(labels.data.numpy()),len(labels.data.numpy()))
                #print("cast...",end="",flush=True)
                cast_start=time.time()
                if use_gpu:
                    inputs=torch.Tensor(inputs).to(device)#.unsqueeze_(1)
                    n_time=time.time()
                    act_time=n_time-cast_start
                    #print("input time:"+str(act_time),flush=True)
                    n_time=time.time()
                    labels = torch.Tensor(labels).to(device)
                    act_time2=time.time()-n_time
                    #print("label time:"+str(act_time2),flush=True)
                else:
                    inputs=Variable(inputs)#.unsqueeze_(1)
                    labels = Variable(labels)#.float()
                
                batch_size,n_input_channel,img_height,img_width=tuple(inputs.shape)
                n_time=time.time()
                #print("complete",flush=True)
                #print("zero_grad()...",end="",flush=True)
                optimizer.zero_grad()
                #print("complete",flush=True)
                #print("get output...",end="",flush=True)
                # forward
                outputs=model(inputs)
                #print("complete",flush=True)
                #print("calc class weight...",end="",flush=True)
                label_weight_sum=labels.sum(dim=(0,2,3))
                label_weight_sum[label_weight_sum==0]=1
                class_weights=1/label_weight_sum
                #print("complete",flush=True)
                #print(outputs.shape,labels.shape,class_weights.shape)
                #print("calc loss...",end="", flush=True)
                loss = CrossEntropyLoss2d(class_weights)(outputs, labels.argmax(1))#weight=class_weights)(outputs, labels.argmax(1))
                #print((outputs.argmax(1)==4).sum())
                #print(class_weights.cpu().data.numpy())
                #loss = criterion(outputs, labels.argmax(1))
                #print("complete", flush=True)

                if phase == 'train':
                    #print("backward and optimize...",end="", flush=True)
                    loss.backward()
                    optimizer.step()
                    #print("complete")
                #print("plus loss...",end="", flush=True)
                # statistics
                #batch_size=len(outputs)
                running_loss += loss.data* batch_size
                #print("complete", flush=True)
                #print("calc iou...",end="", flush=True)
                running_iou += iou(outputs,labels).cpu().data.numpy()*batch_size
                #print("complete", flush=True)
                #print(loss.data,outputs.cpu().data.numpy().reshape((4,))-labels.cpu().data.numpy().reshape((4,)))
                act_time3=time.time()-n_time
                #print("process time:"+str(act_time3),flush=True)
                n_time=time.time()
                
            # サンプル数で割って平均を求める
            epoch_loss = running_loss / dataset_sizes[phase]#dataset_sizes[phase]
            epoch_iou = running_iou / dataset_sizes[phase]#dataset_sizes[phase]
            
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

                if epoch>0:
                    viz.line(X=np.arange(epoch+1),Y=loss_dict["train"][:epoch+1],update="replace",win="loss",name="train")
                    viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1],update="replace",win="iou",name="train")
                    viz.line(X=np.arange(epoch+1),Y=loss_dict["validation"][:epoch+1],update="replace",win="loss",name="validation")
                    viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1],update="replace",win="iou",name="validation")

                else:
                    win_loss=viz.line(X=np.arange(epoch+1),Y=loss_dict["train"][:epoch+1],win="loss",name="train")
                    win_iou=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1],win="iou",name="train")
                    viz.line(X=np.arange(epoch+1),Y=loss_dict["validation"][:epoch+1],win="loss",name="validation")
                    viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1],win="iou",name="validation")



            # deep copy the model
            # 精度が改善したらモデルを保存する
            if phase == "validation" and epoch_iou > best_iou:
                #print("save weights...",end="")
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path+"weights.pth")
                #print("complete")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__=="__main__":
    device = torch.device('cuda')
    ip=int(os.uname()[1][2:])
    current=os.getcwd()
    save_path=os.path.join(current,"result/ip_"+str(ip))
    os.makedirs(save_path, exist_ok=True)
    viz = Visdom(port=8097,server="http://localhost")
    #data_dir="../../../data/double_input/"
    input_dir="../../../data/seg_train_images/"
    label_dir="../../../data/seg_train_annotations/"
    per_train=0.9
    #batch_size=16
    #num_sample=2242
    data_transforms = {
        'train': transforms.Compose([
            my_transform.Clahe(clipLimit=4,tileGridSize=(8,8))
            #transforms.Resize((32,32),interpolation=Image.BILINEAR),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "validation": transforms.Compose([
            my_transform.Clahe(clipLimit=4,tileGridSize=(8,8))
            #transforms.Resize((32, 32),interpolation=Image.BILINEAR),
            #transforms.CenterCrop(224),
            #transforms.ToTensor()
            #transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ]),
    }
    #data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in["train","validation"]}

    image_datasets={x:imagefolder.EdgeDataset(input_dir=input_dir,
                                            label_dir=label_dir,
                                            input_transform=data_transforms[x],
                                            #label_transform=data_transforms[x],
                                            mode=x,
                                            per_train=per_train)
                    for x in ["train","validation"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    print(dataset_sizes)

    num_samples={x:dataset_sizes[x]//20 for x in ['train', 'validation']}
    sample_weight={x:[1]*dataset_sizes[x] for x in ['train', 'validation']}
    data_sampler={x:torch.utils.data.sampler.WeightedRandomSampler(sample_weight[x], num_samples=num_samples[x], replacement=True) for x in ["train","validation"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=2,
                                              #shuffle=x=="train",
                                              sampler=data_sampler[x],
                                              num_workers=2) for x in ['train', 'validation']}

    use_gpu = torch.cuda.is_available()


    model=DeepLab()
    #model.load_state_dict(torch.load(os.path.join(current,"pretrained_models/model_13_2_2_2_epoch_580.pth")))
    #model.aspp.conv_1x1_4 = nn.Conv2d(256, 20, kernel_size=1)
    """
    for idx,p in enumerate(model.parameters()):
        if idx!=0:
            p.requires_grad = False
    """
    if use_gpu:
        #torch.distributed.init_process_group(backend="nccl")
        model = nn.DataParallel(model).to(device)
        #model = model.cuda()
        print(model.module)

    #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.001)#, momentum=0.9)

    #model=train_model(model,optimizer,dataloaders=dataloaders,num_epochs=1)
    #for p in model.parameters():
    #    p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model=train_model(model,optimizer,dataloaders=dataloaders,num_epochs=500)