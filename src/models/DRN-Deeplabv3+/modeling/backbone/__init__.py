from modeling.backbone import resnet, xception, drn, mobilenet,resnext,seresnext

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet50' or backbone == 'resnet100':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'seresnet50':
        return resnet.SEResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn38':
        return drn.drn_d_38(BatchNorm)
    elif backbone == 'drn54':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'drn105':
        return drn.drn_d_105(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'seresnext101':
        return seresnext.se_resnext101_32x4d()
    elif backbone == 'seresnext50':
        return seresnext.se_resnext50_32x4d(BatchNorm=BatchNorm)
    else:
        raise NotImplementedError
