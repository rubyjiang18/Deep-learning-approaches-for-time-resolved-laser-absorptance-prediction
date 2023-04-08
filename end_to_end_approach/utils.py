import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import convnext_tiny
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def initiate_model(model_name: str, pretrained: bool):
    """
    initiate model
    model_name: 'resnet', 'convnext'
    """
    if model_name == 'resnet':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, 1)
    elif model_name == 'convnext':
        model = convnext_tiny(pretrained=pretrained)
        model.classifier[2] = nn.Linear(768, 1)
    else:
        raise NotImplementedError
    return model

'''
Copied from ConvNeXt github
'''
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train(model, device, train_loader, optimizer, criterion, scaler, num_training_steps_per_epoch, scheduler=None, start_steps=None, lr_schedule_values=None):
    """
    train one epoch
    """

    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):

        optimizer.zero_grad()


        if i >= num_training_steps_per_epoch:
            continue
        it = start_steps + i  # global training iteration
        if lr_schedule_values is not None:
            for j, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]


        x = x.float().to(device) 
        y = y.float().to(device)
        assert(len(x) == len(y))

        with torch.cuda.amp.autocast():   

            outputs = model(x) 
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, y)

            # print("outputs", outputs)
            # print("target", y)
            # print("loss", loss)
            # print("---")

        total_loss += float(loss)
         
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()

        if scheduler:
            scheduler.step() 
        #
        del x
        del y
        torch.cuda.empty_cache()

    train_loss = float(total_loss / len(train_loader))

    return train_loss

def validation(model, device, val_loader, optimizer, criterion):
    """
    validate one epoch
    """
    model.eval()
    total_loss = 0

    for i, (x, y) in enumerate(val_loader):

        optimizer.zero_grad()
        x = x.float().to(device) 
        y = y.float().to(device)
        assert(len(x) == len(y))

        with torch.cuda.amp.autocast():   

            outputs = model(x) 
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, y)
            # print("outputs", outputs)
            # print("target", y)
            # print("loss", loss)
            # print("---")

        total_loss += float(loss)

        del x
        del y
        torch.cuda.empty_cache()

    val_loss = float(total_loss / len(val_loader))
  
    return val_loss

def test(model, batch_size, test_loader, criterion, optimizer, scaler):
    '''
    batch size = 1
    '''
    true_y = []
    pred_y = []
    
    model.eval()
    total_loss = 0

    for i, (x, y) in enumerate(test_loader):

        optimizer.zero_grad()
        x = x.float().to(device) 
        y = y.float().to(device)
        assert(len(x) == len(y))

        with torch.cuda.amp.autocast():   

            outputs = model(x) 
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, y)

            true_y.append(y[0].cpu().detach().numpy())
            pred_y.append(outputs[0].cpu().detach().numpy())


        total_loss += float(loss)

        del x
        del y
        torch.cuda.empty_cache()

    test_loss = float(total_loss / len(test_loader))

    return test_loss, true_y, pred_y

def save_model(model, epoch, model_name, optimizer, scheduler, batch_size):
    torch.save({  
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        #'scheduler_state_dict' : scheduler.state_dict(),
                        'batch_size': batch_size,
                        'lr': optimizer.param_groups[0]['lr'],
            }, model_name + '_epoch_' + str(epoch+1))
    

def save_loss_record(train_loss_record, val_loss_record, csv_file_name):
    df = pd.DataFrame(columns=['train_loss', 'val_loss'])
    df['train_loss'] = train_loss_record
    df['val_loss'] = val_loss_record
    df.to_csv(csv_file_name, index=False)

def check_match(data_loader):
    """
    check if image and label match
    """
    
    for i, data in enumerate(data_loader):
        x , y = data

        x = x.float().to(device)  # [b_s, 1, 300, 300]
        print('x shape', x.size())
        y = y.float().to(device)
        print('y shape', y.size())
        print('y', y)

        for img in x:
            print('absorptivity', )
            img = img.squeeze(0)
            img = img.squeeze(0)
            img = img.cpu()
            plt.imshow(img[0])
            plt.show()