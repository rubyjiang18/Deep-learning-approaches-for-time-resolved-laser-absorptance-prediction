import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
from skimage.color import gray2rgb


def plot_2_sidebyside(img, mask):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    plt.gray()
    ax1.imshow(img)
    ax2.imshow(mask)

    ax1.set_title("img")
    ax2.set_title("mask")

    plt.show()

def plot_3_sidebyside(img, img2, img3):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)  # left side
    ax2 = fig.add_subplot(132)  # right side
    ax3 = fig.add_subplot(133)
    plt.gray()
    ax1.imshow(img)
    ax2.imshow(img2)
    ax3.imshow(img3)

    ax1.set_title("img")
    ax2.set_title("mask")
    ax3.set_title("pred mask")
    plt.show()



'''
Save model to path, default my google drive
set path = None to save to colab
'''
def save_model(model, epoch, model_name, optimizer, scheduler, grad_scaler, batch_size,
              path = "/content/drive/MyDrive/DL_segmentation_models/"):
    torch.save({  
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'grad_scaler': grad_scaler.state_dict(),
                        'batch_size': batch_size,
                        'lr': optimizer.param_groups[0]['lr'],
            }, 
            path + model_name + '_epoch_' + str(epoch))


      
# def save_loss_record(train_loss_record, val_loss_record, csv_file_name):
#     df = pd.DataFrame(columns=['train_loss', 'val_loss'])
#     df['train_loss'] = train_loss_record
#     df['val_loss'] = val_loss_record
#     df.to_csv(csv_file_name, index=False)

def save_loss_record(train_loss_record, val_loss_record, lr_record, csv_file_name, 
                path = "/content/drive/MyDrive/DL_segmentation_models/"):
    df = pd.DataFrame(columns=['train_loss', 'val_loss', 'lr'])
    df['train_loss'] = train_loss_record
    df['val_loss'] = val_loss_record
    df['lr'] = lr_record[:-1]
    df.to_csv("/content/drive/MyDrive/DL_segmentation_models/" + csv_file_name, index=False)


# For evaluation
def visualize_prediction_accuracy(pred_mask, true_mask):
    '''
    both pred and true mask are grayscale, only one channel
    '''
    true_mask = gray2rgb(true_mask) #(576, 576, 3)
    out = np.zeros(true_mask.shape, dtype='uint8') #(576, 576, 3)
    t = np.all(true_mask == 1, axis=-1) #(576, 576)
    p = pred_mask # (576, 576)

    channels = [t & p, t & ~p, ~t & p] # true positive (white), false negative (pink), false positive (blue)
    colors = [[255, 255, 255], [255, 0, 255], [0, 255, 255]]

    for n in range(3):
        ch = channels[n]
        color = colors[n]

        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                if ch[i, j] == 1:
                    out[i,j,:] = color
    
    return out