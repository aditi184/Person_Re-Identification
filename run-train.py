import os
import time
import argparse
import random
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
from utils import *
from model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train Person ReID Model')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--train_data_dir', type=str, default="/home/shubham/CVP/data/train/")
    parser.add_argument('--model_name', type=str, default="la-tranformer")
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
        # loss = 0.0
        # for k,v in output.items():
        #     loss += loss_fn(output[k], target)
        loss = 0.0
        for loss_function in loss_fn:
            for k,v in output.items():
                loss += loss_function(output[k], target)
        
        loss.backward()
        optimizer.step()
        
        # print(preds, target.data)
        # acc = (preds == target.data).float().mean()
        # print(acc)
        
        # print(acc)
        # epoch_loss += loss/len(loader)
        # epoch_accuracy += acc / len(loader)
        # if acc:
        #     print(acc, epreds, target.data)
        
        epoch_loss += (loss.item()/data.shape[0])
        correct_predictions += (preds.eq(target.data).sum().item())
        total_samples += data.size(0)
        epoch_accuracy = correct_predictions/total_samples
        # print(f"Epoch : {epoch}; loss : {epoch_loss:.4f}; acc: {epoch_accuracy:.4f}", end="\r")

    # print("total_samples", total_samples, "correct", correct_predictions)
    epoch_loss /= len(loader)
    return OrderedDict([('train_loss', epoch_loss), ("train_accuracy", epoch_accuracy)])

# def eval_one_epoch(epoch, model, loader, loss_fn):
#     model.eval()
#     epoch_accuracy, epoch_loss = 0, 0
#     total_samples, correct_predictions = 0, 0
#     with torch.no_grad():
#         for data, target in tqdm(loader):
#             data, target = data.to(device), target.to(device)

#             # predictions
#             output = model(data)
#             score = 0.0
#             sm = nn.Softmax(dim=1)
#             for k, v in output.items():
#                 score += sm(output[k])
#             _, preds = torch.max(score.data, 1)

#             # backpropagation through ensemble
#             loss = 0.0
#             for k,v in output.items():
#                 loss += loss_fn(output[k], target)

#             epoch_loss += (loss.item()/data.shape[0])
#             correct_predictions += (preds.eq(target.data).sum().item())
#             total_samples += data.size(0)
#             epoch_accuracy = correct_predictions/total_samples
#             # print(f"Epoch : {epoch}; loss : {epoch_loss:.4f}; acc: {epoch_accuracy:.4f}", end="\r")

#     # print("total_samples", total_samples, "correct", correct_predictions)
#     epoch_loss /= len(loader)
#     return OrderedDict([('val_loss', epoch_loss), ("val_accuracy", epoch_accuracy)])

args = parse_args()
fix_seed(args.seed)
train_data_dir, model_name, model_dir, num_epochs = args.train_data_dir, args.model_name, args.model_dir, args.num_epochs

### hyper parameters
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# device = "cpu" 
device = "cuda" if torch.cuda.is_available() else "cpu"
# num_epochs = 15
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

# data_dir = "/home/shubham/CVP/TrainData_split/"
# data_dir = "/home/shubham/CVP/data/"
# image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
#                                           data_transforms['train'])
# image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
#                                           data_transforms['val'])

train_dir = train_data_dir
# val_dir = "/home/shubham/CVP/data/val/all_imgs/"

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
# image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])

train_loader = DataLoader(dataset = image_datasets['train'], batch_size=batch_size, shuffle=True )
# valid_loader = DataLoader(dataset = image_datasets['val'], batch_size=batch_size, shuffle=True)
class_names = image_datasets['train'].classes # '001','003', etc
print("number of classes in train data", len(class_names))
# print(len(image_datasets['val'].classes)) # '001','003', etc

### Model
# Load pre-trained ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)
# vit_base.eval()

# Create LA Transformer
model = LATransformer(ViT=vit_base, lmbd=lmbd, num_classes=62).to(device) # len(class_names)
# model.eval()

# model_name = "la-tranformer"
# model_dir = "/home/shubham/CVP/model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

freeze_all_blocks(model)
unfreeze_block_id = 0

# loss function
# criterion = nn.CrossEntropyLoss()
# criterion = TripletLoss()
# criterion = CrossEntropyLabelSmooth()
criterion = [CrossEntropyLabelSmooth(), TripletLoss()]

# optimizer
optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=lr)

# # scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

print("training...")
# num_eps = 10
# pbar = tqdm(np.arange(num_eps).tolist())
for epoch in range(num_epochs):
    # if epoch == num_epochs//2:
    #    criterion = TripletLoss()

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
    # va = val_metrics['val_accuracy']
    # vl = val_metrics['val_loss']
    # pbar.set_description(f"Train Acc : {ta}, Train Loss : {tl}, Val Acc : {va}, Val Loss : {vl}")
    
    print(f"Epoch : {epoch}; trainacc : {ta:.4f}")
    # print(f"Epoch : {epoch}; trainacc : {ta:.4f}; valacc: {va:.4f}", end="\r")

    # deep copy the model
    # last_model_wts = model.state_dict()
    # if eval_metrics['val_accuracy'] > best_acc:
    #     best_acc = eval_metrics['val_accuracy']
    #     save_network(model, epoch,name)
    #     print("SAVED!")

save_network(model, model_dir, model_name) 
print(model_name +" saved at " + model_dir)