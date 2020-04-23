import torch
import torch.nn as nn
import torch.nn.functional as F

def criterion(logits, labels, is_weight=False):
    #l = BCELoss2d()(logits, labels)
    #l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)
    """
    a   = F.avg_pool2d(labels,kernel_size=3,padding=1,stride=1)
    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    weights  = Variable(torch.ones(a.size())).cuda()

    if is_weight:
        w0 = weights.sum()
        weights = weights + ind*2
        w1 = weights.sum()
        weights = weights/w1*w0
    """
    l =  SoftDiceLoss()(logits,labels)+Weighted_MapLoss().cuda()(logits, labels)

    return l

class CustumizedCELoss2d(nn.Module):
    def __init__(self):
        super(CustumizedCELoss2d).__init__()


class HardDiceLoss(nn.Module):
    def __init__(self,per_image=True):
        super(HardDiceLoss,self).__init__()
        self.per_image=per_image


    def _hard_sigmoid(self,x):
        x[x>1]=1
        return x

    def _per_image_forward(self,outputs,labels):
        class_list=set(labels.cpu().data.numpy().astype("uint8").flatten())
        intersection=outputs*labels
        union=self._hard_sigmoid(outputs+labels)
        loss=0
        for c in class_list:
            loss += intersection[c].sum()/union[c].sum()
        return loss/len(class_list)

    def forward(self,outputs,labels):
        bs=outputs.size(0)
        if self.per_image:
            loss=0
            for i in range(bs):
                loss+=self._per_image_forward(outputs[i],labels[i])
            return 1-loss/bs
        else:
            return 1-self._forward(outputs,labels)

class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        #self.bce_loss = StableBCELoss()
    def forward(self, logits, labels):
        logits_flat = logits.contiguous().view (-1)
        labels_flat = labels.contiguous().view(-1)
        return self.bce_loss(logits_flat, labels_flat)

class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score

class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def _per_image_forward(self,probs,labels):
        class_list=set(labels.cpu().data.numpy().astype("uint8").flatten())
        intersection=probs*labels
        union=probs+labels
        loss=0
        for c in class_list:
            loss += 1-2*intersection[c].sum()/union[c].sum()
        return loss/len(class_list)

    def forward(self, logits, labels,per_image=False):
        probs = F.softmax(logits)
        num = labels.size(0)
        if per_image:
            loss=0
            for i in range(num):
                loss+=self._per_image_forward(probs[i],labels[i])
            return loss/num

        n_classes=labels.size(1)
        m1  = probs.contiguous().view (num,n_classes,-1)
        m2  = labels.contiguous().view(num,n_classes,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(2)+1) / (m1.sum(2) + m2.sum(2)+1)
        score = 1- score
        score=score.sum()/(num*n_classes)
        return score

class CustomizedSoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(CustomizedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.softmax(logits)
        num = labels.size(0)
        n_classes=labels.size(1)
        m1  = probs.contiguous().view(num,n_classes,-1)
        m2  = labels.contiguous().view(num,n_classes,-1)
        intersection = (m1 * m2)
        score = (intersection.sum(2)+1) / (F.tanh(m1 + m2).sum(2)+1)
        score = 1- score
        score=score.sum()/(num*n_classes)
        return score

class SoftDiceLoss2d(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss2d, self).__init__()
        self.diceloss=SoftDiceLoss()

    def forward(self, logits, labels):
        bs,n_classes,height,width=tuple(logits.shape)
        max_map=logits.argmax(1)
        for batch in range(bs):
            score=0
            class_list=set(labels[batch].argmax(0).cpu().data.numpy().astype("uint8").flatten())
            #class_list=class_list.intersection(set([0,1,2,3]))
            for cla in class_list:
                score+=self.diceloss(logits[:,cla],labels[:,cla])
            score/=(cla*bs)
        return score



class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()


# ダイス係数を計算する関数
    def dice_coef(self, y_pred, y_true):
        #print(y_pred.shape,y_true.shape)
        num=y_true.size(0)
        y_true = y_true.contiguous().view(num,-1)
        y_pred = y_pred.contiguous().view(num,-1)
        intersection = y_true * y_pred
        score = (2.0 * intersection.sum(1) + 1) / (y_true.sum(1) + y_pred.sum(1) + 1)
        return score.sum()/num

    # ロス関数
    def forward(self, y_pred, y_true):
        return 1.0 - self.dice_coef(y_true, y_pred)

class TverskyLoss(nn.Module):
    def __init__(self,alpha=None,beta=None):
        super().__init__()
        if alpha:
            self.ALPHA = alpha # 0～1.0の値、Precision重視ならALPHAを大きくする
        else:
            self.ALPHA = 1
        if beta:
            self.BETA=beta # 0～1.0の値、Recall重視ならALPHAを小さくする
        else:
            self.BETA = 1


    def tversky_index(self, y_pred, y_true):
        num=y_true.size(0)
        y_pred=y_pred.sigmoid()
        y_true = y_true.contiguous().view(num,-1)
        y_pred = y_pred.contiguous().view(num,-1)
        intersection = y_true * y_pred
        false_positive = (1.0 - y_true) * y_pred
        false_negative = y_true * (1.0 - y_pred)
        score = (intersection.sum(1)+1) / (intersection.sum(1) + self.ALPHA*false_positive.sum(1) + self.BETA*false_negative.sum(1)+1)
        return score.sum()/num
    def forward(self, y_pred, y_true):
        return 1.0 - self.tversky_index(y_pred,y_true)


class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False)

    def forward(self, input, target):
        loss = self.nll(input, target)

        one_hot = make_one_hot(target.unsqueeze(dim=1), input.size()[1])
        inv_probs = 1 - input.exp()
        focal_weights = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights
