import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

import os 
from os import path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from tqdm import tqdm
from PIL import Image

# read tiff
import zipfile
from tifffile import imread
from torchvision.transforms import ToTensor
import random
import csv

import matplotlib.pyplot as plt

from Dataset import train_preprocess, test_preprocess, Keyhole, Keyhole_Test
from utils import initiate_model, cosine_scheduler, train, validation, save_model, save_loss_record

train_data_path = "/home/ec2-user/absorption/training_dataset" # flag
test_data_path = "/home/ec2-user/absorption/testing_dataset"   # flag

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 2 if cuda else 0

batch_size = 32
split_num="1" # choose one from 1, 2, 3, 4, 5 as i had 5 different splits

### Load data
train_dataset = Keyhole(train_data_path, transform= train_preprocess, train=True, split=split_num)
val_dataset = Keyhole(train_data_path, transform= test_preprocess, train=False, split=split_num)
test_dataset = Keyhole_Test(test_data_path, transform= test_preprocess)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

### Load model
pretrain_flag = False   ### flaf
torch.cuda.empty_cache()
model = initiate_model('convnext', pretrained=pretrain_flag)
model.cuda()

### Set up training
epochs=300
criterion = torch.nn.SmoothL1Loss()
if type(model) == torchvision.models.resnet.ResNet:
    lr = 4e-3
    lr_min = 1e-6
if type(model) == torchvision.models.convnext.ConvNeXt:
    lr = 1e-4
    lr_min = 1e-7

num_training_steps_per_epoch = len(train_dataset) // batch_size
lr_schedule_values = cosine_scheduler(lr, lr_min, 300, num_training_steps_per_epoch, warmup_epochs=20)

optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)

scheduler = None

epochs = 300

scaler = torch.cuda.amp.GradScaler()

### Train
train_loss_record = []
val_loss_record = []
for epoch in range(epochs):
  train_loss = train(model, 
                     device, 
                     train_loader, 
                     optimizer, 
                     criterion, 
                     scaler, 
                     num_training_steps_per_epoch,
                     scheduler=None,
                     start_steps=epoch*num_training_steps_per_epoch, 
                     lr_schedule_values=lr_schedule_values)
  print("Epoch {}/{}: Train Loss {:.04f}, lr {:.08f}".format(epoch + 1, epochs, train_loss, optimizer.param_groups[0]['lr']))
  val_loss = validation(model, device, val_loader, optimizer, criterion)
  print("val loss: {:.4f}".format(val_loss))
  train_loss_record.append(train_loss)
  val_loss_record.append(val_loss)
  if epoch == 299:
    ####### Change the name!!! #######
    if pretrain_flag:
        name = "Convnext_pretrain_split" # "Resnet_pretrain_split"
    else:
        name = "Convnext_nopretrain_split"
    save_model(model, epoch, name+split_num, optimizer, scheduler, batch_size)
    save_loss_record(train_loss_record, val_loss_record, name+split_num+"_loss_log.csv")


save_model(model, epoch, "Resnet_nopretrain_split1", optimizer, scheduler, batch_size)
save_loss_record(train_loss_record, val_loss_record, "Resnet_nopretrain_split1_loss_log.csv")

### Test
checkpoint = torch.load("ResNet50_nopretrain_epoch_300")
model.load_state_dict(checkpoint['model_state_dict'])