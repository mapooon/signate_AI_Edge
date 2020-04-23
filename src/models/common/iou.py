import torch
import numpy as np
def iou(y_pred,y_true,average=True):#(bs,n_classes,height,width)
    if not average:
        return each_iou(y_pred,y_true)
    #print("n_class",flush=True)
    bs,_,height,width=tuple(y_true.shape)#the number of channels
    max_map=y_pred.argmax(1)
    score=0
    #pred=torch.zeros((bs,n_classes,height,width)).to("cuda:0")
    count_class=y_true.sum((2,3))>0#(bs,n_classes)
    for batch in range(bs):
        class_list=set(y_true[batch].argmax(0).cpu().data.numpy().astype("uint8").flatten())
        class_list=class_list.intersection(set([0,1,2,3]))
        #class_list.remove(4)
        class_score=0
        for cla in class_list:
            pred = (max_map[batch]==cla).float()
            cap = (pred * y_true[batch,cla]).float().sum()
            cup = ((pred + y_true[batch,cla]) >= 1).float().sum()
            #cup[cup==0]=1
            class_score += cap/cup
        class_score/=len(class_list)
        score += class_score
    score /= bs
   
    return score#output

def each_iou(y_pred,y_true):
    bs,_,height,width=tuple(y_true.shape)#the number of channels
    max_map=y_pred.argmax(1)
    score=np.zeros(4)
    #pred=torch.zeros((bs,n_classes,height,width)).to("cuda:0")
    count_class=y_true.sum((2,3))>0#(bs,n_classes)
    class_count=np.zeros(4)
    for batch in range(bs):
        class_list=set(y_true[batch].argmax(0).cpu().data.numpy().astype("uint8").flatten())
        class_list=class_list.intersection(set([0,1,2,3]))
        #class_list.remove(4)
        #class_score=0
        for cla in class_list:
            class_count[cla]+=1
            pred = (max_map[batch]==cla).float()
            cap = (pred * y_true[batch,cla]).float().sum()
            cup = ((pred + y_true[batch,cla]) >= 1).float().sum()
            #cup[cup==0]=1
            class_score = cap/cup
            score[cla]+=class_score.item()
        
    score /= bs
    
    return score

def calc_iou(y_pred,y_true):
    """
    y_pred,y_true:(batch_size,n_class,height,width)
    """
    bs=y_true.shape[0]
    #print("3.1",flush=True)
    cap=y_pred * y_true
    #print("3.2",flush=True)
    cap=cap.sum()
    #print("3.3",flush=True)
    #print(cap)
    cup=(y_pred + y_true)>=1
    #print("3.4",flush=True)
    cup=cup.sum()
    #print("3.5",flush=True)
    #print(cup)
    score=cap/cup
    #print(y_pred,y_true)
    
    return score/bs