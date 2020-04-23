import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ConvBNActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True),BatchNorm=None):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.bn = BatchNorm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1,
                 BatchNorm=None, activation=nn.ReLU(inplace=True)):
        super(UpBlock, self).__init__()

        
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1)
        
        #self.bn=BatchNorm(out_channels)
        self.act=activation
        self.convblock=ConvBNActivation(in_channels,out_channels,BatchNorm=BatchNorm)

    def forward(self, x):
        x = self.up(x)
        #x = self.bn(x)
        #x = self.act(x)
        x = self.convblock(x)
    
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1,
                 BatchNorm=None, activation=nn.ReLU(inplace=True),n_iter=2):
        super(DeconvBlock, self).__init__()
        #deconv_path = []
        #deconv_path.append(UpBlock(in_channels,out_channels,BatchNorm=BatchNorm))
        # for i in range(1,n_iter):
        #     deconv_path.append(UpBlock(out_channels,out_channels,BatchNorm=BatchNorm))
        self.deconvblock1=UpBlock(in_channels,out_channels,BatchNorm=BatchNorm)
        self.deconvblock2=UpBlock(out_channels,out_channels,BatchNorm=BatchNorm)

    def forward(self,x):
        x=self.deconvblock1(x)
        x=self.deconvblock2(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet50' or backbone == 'seresnet101' or backbone == 'seresnet50':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'seresnext101':
            low_level_inplanes = 256
        elif backbone == 'seresnext50':
            low_level_inplanes = 256
        elif backbone == 'drn38':
            low_level_inplanes = 64
        elif backbone == 'drn54':
            low_level_inplanes = 256

        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU(inplace=True)
        if backbone=='drn54':
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(inplace=True),
                                        #nn.Dropout(0.2),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        #SEModule(256,16),
                                        nn.ReLU(inplace=True),
                                        #SEModule(256,16),
                                        #nn.Dropout(0.1),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        else:
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        #SEModule(256,16),
                                        nn.ReLU(inplace=True),
                                        #SEModule(256,16),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)