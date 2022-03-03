import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import timm

from efficientnet_pytorch import EfficientNet



def resnet152(classes):
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.fc = nn.Linear(in_features=2048, out_features=classes, bias=True)
    
    nn.init.xavier_uniform_(resnet152.conv1.weight)
    nn.init.xavier_uniform_(resnet152.fc.weight)
    
    stdv = 1.0/np.sqrt(classes)
    resnet152.fc.bias.data.uniform_(-stdv, stdv)
    
    return resnet152


def efficientnet_b3(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes=classes)
    return effnet

def efficientnet_b4(classes):
    # effnet = timm.create_model('tf_efficientnet_b4', pretrained=True)
    #    for n, p in effnet.named_parameters():
    #     if '_fc' not in n:
    #         p.requires_grad = False
    # effnet = torch.nn.parallel.DistributedDataParallel(effnet)
    effnet = EfficientNet.from_pretrained('efficientnet-b4', num_classes=classes)
    
    return effnet

def efficientnet_b7(classes):
    effnet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=classes)
    return effnet

def get_model(model_name:str, classes:int):
    """[summary]
    Args:
        model_name (str): [name for a pre-trained model]
        classes (int): [output channels of called pre-trained model]
    Returns:
        [nn.Module]: [description]
    """
    model_dict = {
        'resnet152':resnet152,
        'efficientnet_b3':efficientnet_b3,
        'efficientnet_b4':efficientnet_b4,
        'efficientnet_b7':efficientnet_b7,
    }

    return model_dict[model_name](classes)