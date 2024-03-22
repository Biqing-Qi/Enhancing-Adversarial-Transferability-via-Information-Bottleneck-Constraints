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
import torch.nn.functional
import timm
from used_attacks import IBTA_TIFGSM, IBTA_MIFGSM, IBTA_DIFGSM, IBTA_Logits_TI, IBTA_RAP_Logits_TI, IBTA_S2I_MIFGSM
from scipy import stats as st
import argparse
from torch.autograd import Variable
import logging
import torch.distributions as dist
from torch.distributions import Normal
import math
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--method_num', default=0, help='index of attack method')
parser.add_argument('--exp', type=int, default=0, help='experiment number')
parser.add_argument('--batch_size', type=int, default=48, help='batch_size')
parser.add_argument('--steps', type=int, default=20, help='number of steps')
parser.add_argument('--step_size', type=float, default=1.25, help='number of steps')
parser.add_argument('--targeted', type=bool, default=False, help='IF Targeted' )
parser.add_argument('--lmd', type=float, default=0.1, help='Lambda' )
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma' )
parser.add_argument('--sigma', type=float, default=0.1, help='Sigma' )
args = parser.parse_args()
if args.targeted:
    print("targeted:",args.targeted)


def kl_divergence_to_standard_normal(logits):
    probs = F.softmax(logits, dim=-1)
    
    q = torch.distributions.Normal(0, 1).log_prob(torch.randn_like(probs))
   
    kl_div = torch.sum(probs * (probs.log() - q))/probs.shape[0]
    return kl_div

def IB_loss(logits, labels):
    cri = nn.CrossEntropyLoss()
    loss = cri(logits, labels) - 0.3*kl_divergence_to_standard_normal(logits)
    return loss


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

#Src Models
if args.exp == 1:
    modellist = ['ResNet50', 'ResNet152', 'vgg19bn', 'DenseNet121']
elif args.exp == 2:
    modellist = ["inception_v3","Ensemble"]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])
train_data=ImageFolder('imagenet_val',transform=transform)


targets = [31,56,241,335,458,532,712,766,887,975]

def norm(t):
        t_c = t.clone()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        t_c[:, 0, :, :] = (t_c[:, 0, :, :] - mean[0])/std[0]
        t_c[:, 1, :, :] = (t_c[:, 1, :, :] - mean[1])/std[1]
        t_c[:, 2, :, :] = (t_c[:, 2, :, :] - mean[2])/std[2]

        return t_c

def advtest(model,target_models,attack,targeted=True,target=None):

    attack.target = target

    total=0

    train_data_excepttarget = ImageFolder('imagenet_val',transform=transform)
    source_samples = []
    for img_name, label in train_data_excepttarget.samples:
        if label in targets:
            source_samples.append((img_name, label))
    train_data_excepttarget.samples = source_samples
    if args.targeted:
        source_samples = []
        j=0
        for img_name, label in train_data_excepttarget.samples:
            if label!=target:
                source_samples.append((img_name, label))
                j+=1
    
        train_data_excepttarget.samples = source_samples
        train_data_excepttarget_set=DataLoader(dataset=train_data_excepttarget,batch_size=args.batch_size,num_workers=10)
    
    else:
        train_data_excepttarget_set=DataLoader(dataset=train_data_excepttarget,batch_size=args.batch_size,num_workers=10)
    successed = [0]*len(target_models)
    
    for i,(inputs,labels) in enumerate(train_data_excepttarget_set):
        
        inputs,labels=Variable(inputs.to(device),requires_grad=True),labels.to(device)
        total+=inputs.shape[0]
        advinputs=attack(inputs,labels)
        with torch.no_grad():
            for j, modelname1 in enumerate(target_models):
                
                model1=torch.load(modelname1).to('cuda:0')
                model1.eval()
                for param in model1.parameters():
                    param.requires_grad=False
                output1=model1(norm(advinputs))
                pred1=torch.argsort(output1,dim=-1,descending=True)[:,0]
                if targeted:
                    successed[j]+=torch.eq(pred1,target).sum().item()
                else:
                    successed[j]+=torch.ne(pred1,labels).sum().item()
                del model1
                torch.cuda.empty_cache()
                
               
        del inputs,labels,advinputs
        torch.cuda.empty_cache()
    
    del model

    torch.cuda.empty_cache()
    return [x / total for x in successed]
methodlist = [IBTA_TIFGSM, IBTA_MIFGSM, IBTA_DIFGSM, IBTA_Logits_TI, IBTA_RAP_Logits_TI, IBTA_S2I_MIFGSM]
methodnames = ['IBTA_TIFGSM', 'IBTA_MIFGSM', 'IBTA_DIFGSM', 'IBTA_Logits_TI', 'IBTA_RAP_Logits_TI', 'IBTA_S2I_MIFGSM']

if args.exp == 1:
    tar_modellist = ["ResNet50","ResNet152","vgg19bn",'DenseNet121']
elif args.exp == 2:
    tar_modellist = ["inception_v3", "Ensemble", "inception_v4","incres_v2","adv_inc_v3", "ens_adv_inception_resnet_v2"]

for modelname in modellist:
    model = torch.load(modelname).to('cuda:0')
    model.eval()
    method=methodlist[int(args.method_num)]
    methodname=methodnames[int(args.method_num)]

    if methodname == 'IBTA_RAP_Logits_TI':
        attack=method(model,steps=400,eps=16/255,alpha=args.step_size/255,targeted=args.targeted,lmd=args.lmd,gamma=args.gamma,sigma=args.sigma)
    elif methodname == 'IBTA_Logits_TI':
        attack=method(model,steps=300,eps=16/255,alpha=args.step_size/255,targeted=args.targeted,lmd=args.lmd,gamma=args.gamma,sigma=args.sigma)
    else :
        attack=method(model,steps=args.steps,eps=16/255,alpha=args.step_size/255,targeted=args.targeted,lmd=args.lmd,gamma=args.gamma,sigma=args.sigma)
    
    for param in model.parameters():
       param.requires_grad=False
    tar_modellist = [modelname]
    if args.targeted:
        for i in range(10):
            if i==0:
                transfer_rates = advtest(model,tar_modellist,attack,True,targets[i])
            else:
                transfer_rates = [x + y for x, y in zip(transfer_rates, advtest(model,tar_modellist,attack,True,targets[i]))]
    else:
        transfer_rates = advtest(model,tar_modellist,attack,False,None)
    
    if args.targeted:
        for i,model_name1 in enumerate(tar_modellist): 
            print("transfer rate of the %s on the %s attack for %s: %.2f %% "%(modelname,methodnames[int(args.method_num)],model_name1,100*transfer_rates[i]/10))
    else:
        for i,model_name1 in enumerate(tar_modellist): 
            print("transfer rate of the %s on the %s attack for %s: %.2f %% "%(modelname,methodnames[int(args.method_num)],model_name1,100*transfer_rates[i]))
