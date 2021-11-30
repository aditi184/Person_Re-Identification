import os
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# weights initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class FC_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, droprate=0.5, num_bottleneck=256, return_features=False):
        super(FC_Classifier, self).__init__()
        self.return_features = return_features
        add_block = []
        # if linear:
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        # else:
        #    num_bottleneck = input_dim
        # if bnorm:
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        # if relu:
        #    add_block += [nn.LeakyReLU(0.1)]
        # if droprate > 0:
        add_block+= [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier+= [nn.Linear(num_bottleneck, num_classes)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_features:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

class LATransformer(nn.Module):
    def __init__(self, ViT, lmbd, num_classes=751, test=False):
        super(LATransformer, self).__init__()
        self.test = test
        self.class_num = num_classes # output number of classes
        
        # ViT model
        self.model = ViT
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token # 1, 1, 768
        self.pos_embed = self.model.pos_embed # 1, 197, 768

        # these are ViT model internal hyper-parameters (FIXED) 
        # self.num_blocks = 12 # number of sequential blocks in ViT
        
        # there are 196 patches in each image; thus, we split them into 14 x 14 grid
        self.num_rows = 14 
        self.num_cols = 14

        # Locally aware network
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_rows,768))
        self.lmbd = lmbd

        if not self.test:
            # ensemble of classifiers
            # for i in range(self.num_rows):
            #     name = 'classifier'+str(i)
            #     setattr(self, name, FC_Classifier(input_dim=768, num_classes=self.class_num, droprate=0.5, num_bottleneck=256, return_features=False))
            name = 'classifier'+str(0)
            setattr(self, name, FC_Classifier(input_dim=768, num_classes=self.class_num, droprate=0.5, num_bottleneck=256, return_features=False))

    def forward(self, x):
        # x shape = 32, 3, 224, 224
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x) # 32, 196, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 32, 1, 768
        x = torch.cat((cls_token, x), dim=1) # 32, 197, 768
        trnsfrmr_inp = self.model.pos_drop(x + self.pos_embed) # dropout with p = 0; idk!
        
        # Feed forward the x = (patch_embeddings+position_embeddings) through transformer blocks
        # for i in range(self.num_blocks):
        #    x = self.model.blocks[i](x)
        x = self.model.blocks(trnsfrmr_inp)
        x_trnsfrmr_encdd = self.model.norm(x) # layer normalization; shape = 32, 197, 768
        
        # extract the cls token
        cls_token_out = x_trnsfrmr_encdd[:, 0].unsqueeze(1)
        
        # Average pool
        Q = x_trnsfrmr_encdd[:, 1:]
        L = self.avgpool(Q) # 32, 14, 768
                
        if self.test:
            return L # moving this down the global-cls addition drops the testing score
        
        # Add global cls token to each local token 
        for i in range(self.num_rows):
            out = torch.mul(L[:, i, :], self.lmbd)
            L[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)
            
        # L, _ = torch.max(L, dim=1)
        L = torch.mean(L, dim=1)
        
        # Locally aware network
        part = {}
        predict = {}
        # for i in range(self.num_rows):
        #     part[i] = L[:,i,:] # 32, 768
        #     name = 'classifier'+str(i)
        #     c = getattr(self, name)
        #     predict[i] = c(part[i]) # 32, 751
        name = 'classifier'+str(0)
        c = getattr(self, name)
        predict[0] = c(L) # 32, 751

        return predict

def freeze_all_blocks(model):
    # frozen_blocks = 12
    assert len(model.model.blocks) == 12
    for block in model.model.blocks: # [:frozen_blocks]
        for param in block.parameters():
            param.requires_grad=False

def unfreeze_block(model, block_num = 1):
    # unfreeze transformer blocks from last
    for block in model.model.blocks[11-block_num :]:
        for param in block.parameters():
            param.requires_grad=True
    return model

def save_network(network, model_dir, name):
    save_path = os.path.join(model_dir, name + ".pth")
    torch.save(network.cpu().state_dict(), save_path)
    
    if torch.cuda.is_available():
        network.cuda()

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes=62, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss