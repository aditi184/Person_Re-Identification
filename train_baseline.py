import os
import time
import random
import argparse
import timm
import numpy as np
from PIL import Image
# from tqdm.notebook import tqdm
from tqdm import tqdm
from collections import OrderedDict

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
from model import save_network

def parse_args():
    parser = argparse.ArgumentParser(description='Train Person ReID Model')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--train_data_dir', type=str, default="/home/shubham/CVP/data/train/")
    parser.add_argument('--model_name', type=str, default="la-tf_baseline")
    parser.add_argument('--model_dir', type=str, default="/home/shubham/CVP/model/")
    parser.add_argument('--num_epochs', type=int, default=15)
    args = parser.parse_args()
    return args

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
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
            for i in range(self.num_rows):
                name = 'classifier'+str(i)
                setattr(self, name, FC_Classifier(input_dim=768, num_classes=self.class_num, droprate=0.5, num_bottleneck=256, return_features=False))

    def forward(self, x):
        # x shape = 32, 3, 224, 224
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x) # 32, 196, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 32, 1, 768
        x = torch.cat((cls_token, x), dim=1) # 32, 197, 768
        trnsfrmr_inp = self.model.pos_drop(x + self.pos_embed) # dropout with p = 0; idk!
        
        x = self.model.blocks(trnsfrmr_inp)
        x_trnsfrmr_encdd = self.model.norm(x) # layer normalization; shape = 32, 197, 768
        
        # extract the cls token
        cls_token_out = x_trnsfrmr_encdd[:, 0].unsqueeze(1)
        
        # Average pool
        Q = x_trnsfrmr_encdd[:, 1:]
        L = self.avgpool(Q) # 32, 14, 768
        
        if self.test:
            return L
        
        # Add global cls token to each local token 
        for i in range(self.num_rows):
            out = torch.mul(L[:, i, :], self.lmbd)
            L[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)
        
        # Locally aware network
        part = {}
        predict = {}
        for i in range(self.num_rows):
            part[i] = L[:,i,:] # 32, 768
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i]) # 32, 751
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


def train_one_epoch(epoch, model, loader, optimizer, loss_fn):
    model.train()
    epoch_accuracy, epoch_loss = 0, 0
    total_samples, correct_predictions = 0, 0
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)

        # predictions
        optimizer.zero_grad()
        output = model(data)
        score = 0.0
        sm = nn.Softmax(dim=1)
        for k, v in output.items():
            score += sm(output[k])
        _, preds = torch.max(score.data, 1)
        
        # backpropagation through ensemble
        loss = 0.0
        for k,v in output.items():
            loss += loss_fn(output[k], target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += (loss.item()/data.shape[0])
        correct_predictions += (preds.eq(target.data).sum().item())
        total_samples += data.size(0)
        epoch_accuracy = correct_predictions/total_samples
        # print(f"Epoch : {epoch}; loss : {epoch_loss:.4f}; acc: {epoch_accuracy:.4f}", end="\r")

    # print("total_samples", total_samples, "correct", correct_predictions)
    epoch_loss /= len(loader)
    return OrderedDict([('train_loss', epoch_loss), ("train_accuracy", epoch_accuracy)])
    
args = parse_args()
fix_seed(args.seed)
train_data_dir, model_name, model_dir, num_epochs = args.train_data_dir, args.model_name, args.model_dir, args.num_epochs

### hyper parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
lr = 3e-4
gamma = 0.7
unfreeze_after = 2 # unfreeze transformer blocks after 2 epochs
lr_decay = .8
lmbd = 8

### Load Data
transform_train_list = [
    transforms.Resize((224,224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
data_transforms = {
'train': transforms.Compose( transform_train_list ),
'val': transforms.Compose(transform_val_list),
}

train_dir = train_data_dir
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
train_loader = DataLoader(dataset = image_datasets['train'], batch_size=batch_size, shuffle=True )
class_names = image_datasets['train'].classes # '001','003', etc
print("number of classes in train data", len(class_names))

### Model
# Load pre-trained ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)

# Create LA Transformer
model = LATransformer(ViT=vit_base, lmbd=lmbd, num_classes=62).to(device) # len(class_names)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

freeze_all_blocks(model)
unfreeze_block_id = 0

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(),weight_decay=5e-4, lr=lr)

print("training...")
for epoch in range(num_epochs):

    if epoch % unfreeze_after == 0: # and epoch != 0:
        unfreeze_block_id += 1
        model = unfreeze_block(model, unfreeze_block_id)
        optimizer.param_groups[0]['lr'] *= lr_decay 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Unfrozen Blocks: {unfreeze_block_id}, Current lr: {optimizer.param_groups[0]['lr']}, Trainable Params: {trainable_params}")

    train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
    # val_metrics = eval_one_epoch(epoch, model, valid_loader, criterion)
    ta = train_metrics['train_accuracy']
    tl = train_metrics['train_loss']
    
    print(f"Epoch : {epoch}; trainacc : {ta:.4f}")

save_network(model, model_dir, model_name) 
print(model_name +" saved at " + model_dir)