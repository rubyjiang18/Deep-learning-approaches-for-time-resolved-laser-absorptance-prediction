import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import sys
sys.path.append("utils/")

# Create the argument parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('model_name', type=str, default='unet', help='Model name')
parser.add_argument('--pretrain', type=bool, default=False, help='Pretrained on ImageNet')
parser.add_argument('--split_num', type=str, default='1', help='Split number')
# Parse the arguments
args = parser.parse_args()
print(args)

from unet_model.unet import UNet
from keyholeDataset import Keyhole
from loss import DiceBCEWithActivationLoss 
from augmentation import get_training_augmentation, preprocess
from utils import plot_2_sidebyside, plot_3_sidebyside, save_model, save_loss_record
from iou import iou_numpy
from train import train
from validation import validation
import segmentation_models_pytorch as smp

# Set the device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0
print("Cuda = " + str(cuda)+" with num_workers = "+str(num_workers))

# Set the dataset
data_path = '/content/Keyhole/keyhole_segmentation_data'
batch_size = 2
split_num = args.split_num if args.split_num in ["1", "2", "3", "4", "5"] else "1"
csv_split_name = f"/image_and_split_{split_num}.csv"

train_dataset = Keyhole(data_path, 
                        transform=get_training_augmentation(),
                        preprocess=None,
                        mode="train", 
                        csv_name=csv_split_name)
val_dataset = Keyhole(data_path, 
                      transform=None, 
                      preprocess=None, 
                      mode="val", 
                      csv_name=csv_split_name)
test_dataset = Keyhole(data_path, 
                       transform=None, 
                       preprocess=None, 
                       mode="test", 
                       csv_name=csv_split_name)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Set the model
model_name = args.model_name if args.model_name in ["vanilla_unet", "unet", "unetplusplus", "deeplabv3"] else "unet"
pretrain = args.pretrain if args.pretrain in [True, False] else False
if model_name == "vanilla_unet":
    model = UNet(n_channels=3, n_classes=1, bilinear=1)
elif model_name == "unet":
    model = smp.Unet(encoder_name="resnet50", 
                     encoder_weights="imagenet" if pretrain else None, 
                     classes=1, 
                     activation=None)
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])
elif model_name == "unetplusplus":
    model = smp.UnetPlusPlus(encoder_name="resnet50", 
                             encoder_weights="imagenet" if pretrain else None, 
                             classes=1, 
                             activation=None)
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])
elif model_name == "deeplabv3":
    model = smp.DeepLabV3(encoder_name="resnet50", 
                          encoder_weights="imagenet" if pretrain else None, 
                          classes=1, 
                          activation=None)
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])
else:
    print("Model not found")

full_model_name = f"{model_name}_pretrain_{pretrain}_split_{split_num}"
model.cuda()

# if you want to load the trained mode, uncomment the following lines
# path = "/content/drive/MyDrive/DL_segmentation_models/UnetRes50_Split4_epoch_107"
# checkpoint = torch.load(path)
# model.load_state_dict(checkpoint['model_state_dict'])
# for key, value in checkpoint.items():
#     print(key)

# train the model
optimizer =  optim.RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.99) # 0.99
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
criterion = DiceBCEWithActivationLoss()

epochs = 300
amp = True
train_loss_record= []
val_loss_record= []
lr_record = []
# record the number of times lr changes
prev_lr = 100; # 100 to simulate int.max_value
lr_count = -1

for epoch in range(0, epochs):
  # lr - early stop
  curr_lr = optimizer.param_groups[0]['lr']
  lr_record.append(curr_lr)
  print('New peoch lr: ', curr_lr)
  if curr_lr < prev_lr:
    prev_lr = curr_lr
    lr_count += 1
  if (lr_count == 3):
    print("Early Stop")
    save_model(model, epoch, model_name, optimizer, scheduler, grad_scaler, batch_size)
    save_loss_record(train_loss_record, val_loss_record, lr_record, model_name+".csv")
    break
  # train
  train_loss = train(model, device, train_loader, optimizer, criterion, scheduler, grad_scaler, epoch, epochs, amp=True)
  train_loss_record.append(train_loss)
  # validation
  val_loss = validation(model, device, val_loader, optimizer, criterion, scheduler, epoch, epochs, amp=True)
  val_loss_record.append(val_loss)

save_model(model, epoch, full_model_name, optimizer, scheduler, grad_scaler, batch_size)
save_loss_record(train_loss_record, val_loss_record, lr_record, full_model_name+".csv")


# test
val_loss = validation(model, device, val_loader, optimizer, criterion, scheduler, 0, epochs, amp=True)
train_loss = validation(model, device, train_loader, optimizer, criterion, scheduler, 0, epochs, amp=True)
test_loss = validation(model, device, test_loader, optimizer, criterion, scheduler, 0, epochs, amp=True)
print("Train loss: ", train_loss)
print("Val loss: ", val_loss)
print("Test loss: ", test_loss)