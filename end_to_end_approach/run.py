import sys
import argparse

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from Dataset import train_preprocess, test_preprocess, Keyhole, Keyhole_Test
from utils import initiate_model, cosine_scheduler, train, validation, save_model, save_loss_record

# Create the argument parser
parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('model_name', type=str, default='convnext', help='Model name')
parser.add_argument('--pretrain', type=bool, default=False, help='Pretrained on ImageNet')
parser.add_argument('--split_num', type=str, default='1', help='Split number')
# Parse the arguments
args = parser.parse_args()
print(args)

### Set up device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 2 if cuda else 0

### Load data
# choose one from "1", "2", "3", "4", "5" as 5 different splits are used in the paper
split_num= args.split_num if args.split_num in ["1", "2", "3", "4", "5"] else "1"
batch_size = 32

train_data_path = "/home/ec2-user/absorption/training_dataset"
test_data_path = "/home/ec2-user/absorption/testing_dataset"

train_dataset = Keyhole(train_data_path, transform= train_preprocess, train=True, split=split_num)
val_dataset = Keyhole(train_data_path, transform= test_preprocess, train=False, split=split_num)
test_dataset = Keyhole_Test(test_data_path, transform= test_preprocess)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

### Load model
model_name = args.model_name if args.model_name in ["resnet", "convnext"] else "convnext"
pretrain_flag = args.pretrain if args.pretrain in [True, False] else False
torch.cuda.empty_cache()
model = initiate_model(model_name, pretrained=pretrain_flag)
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
  # save model at the end of last epoch
  if epoch == 299:
    if pretrain_flag:
        name = model_name + "_pretrain_split" + split_num
    else:
        name = model_name + "_nopretrain_split" + split_num
    save_model(model, epoch, name, optimizer, scheduler, batch_size)
    save_loss_record(train_loss_record, val_loss_record, name+"_loss_log.csv")

# if you need to load the trained model, use the following code
# checkpoint = torch.load("ResNet50_nopretrain_epoch_300")
# model.load_state_dict(checkpoint['model_state_dict'])