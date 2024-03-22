import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import time
from torch.autograd import Variable
import random
import os
from PIL import Image
#import matplotlib.pyplot as plt
import torch.nn.functional
import timm
from used_attacks import MIFGSM,PGD,TIFGSM,HSFGSM,HSPGD,AAIPGD,NIFGSM,DIFGSM,SINIFGSM,Po_TI_Trip_FGSM
from scipy import stats as st
import argparse
from torch.autograd import Variable
import logging


class EnsembleModel(torch.nn.Module):
    def __init__(self, model_names, pretrained=True, device='cuda:0'):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList()

        for model_name in model_names:
            model = timm.create_model(model_name, pretrained=pretrained).to(device)
            self.models.append(model)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)

        with torch.no_grad():
            norms = [torch.norm(output, p=2, dim=1, keepdim=True) for output in outputs]

        normalized_outputs = [output / norm for output, norm in zip(outputs, norms)]

        with torch.no_grad():
            weights = torch.mean(torch.cat(norms, dim=1), dim=1)[:, None]

        averaged_output = torch.mean(torch.stack(normalized_outputs), dim=0)

        return averaged_output * weights


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def pretrained_model(model_name):

    if model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg19bn":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "incres_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model("ens_adv_inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "adv_inc_v3":
        model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
    elif model_name == "Ensemble":
        
        model = EnsembleModel(["resnet50.tv_in1k", "inception_v3", "inception_v4"], pretrained = True, device = device)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model

#model = pretrained_model("ResNet50")
#model = pretrained_model("ResNet152")
#model = pretrained_model("vgg19bn")
#model = pretrained_model("DenseNet121")
model = pretrained_model("incres_v2")
print('incres_v2')
model = pretrained_model("ens_adv_inception_resnet_v2")
model = pretrained_model("inception_v3")
model = pretrained_model("inception_v4")
model = pretrained_model("vit")
model = pretrained_model("Ensemble")