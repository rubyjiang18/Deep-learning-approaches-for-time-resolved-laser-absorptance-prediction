import os
import torch
import cv2 as cv
import csv
import numpy as np
from tifffile import imread
from torch.utils.data import Dataset
from torchvision import transforms

# This one is for model trainig, you have both images and masks
class Keyhole(Dataset):
    @staticmethod
    def pad_to_576(image, mode):
        assert mode in {'image', 'mask'}
        # pad val
        pad_val = 0 # if mask, pad_val = 0 black
        if mode == "image":
          pad_val = np.mean(image, axis=(0, 1))
        # pad dimension
        dimension = 576 #572 # goal (572, 572)
        height = image.shape[0]
        width = image.shape[1]
        d_height = dimension - height
        d_width = dimension - width
        
        top, bottom, left, right = 0, 0, 0, 0
        if d_height > 0:
            top = d_height // 2
            bottom = d_height - top
        if d_width > 0:
            left = d_width // 2
            right = d_width - left

        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, value = pad_val)
        return image
    @staticmethod
    def make_even(full_list):
        if (len(full_list)%2 != 0):
            full_list.append(full_list[-1])
        return full_list

    def __init__(self, data_path, transform=None, preprocess=None, mode="train", csv_name = "/image_and_split.csv"):
        '''
        transform -> data augmentation
        preprocess -> normalize 3 channel same as imagenet
        csv_name -> the csv file that contains the data split for train, val, and test
        '''

        assert mode in {"train", "val", "test"}

        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.preprocess = preprocess

        # X represents images, Y represents masks
        self.X_dir = os.path.join(self.data_path, "images")
        self.Y_dir = os.path.join(self.data_path, "masks")
        self.X_files = sorted(os.listdir(self.X_dir))
        self.Y_files = sorted(os.listdir(self.Y_dir))
        # full dataset_X and dataset_Y
        fullset_X = []
        fullset_Y = []
        for idx, name in enumerate(self.X_files):
            if 'tif' not in name:
                continue
            # print("image name ",name)
            img_path = os.path.join(self.X_dir, name)
            # Use you favourite library to load the image
            image = imread(img_path) 
            # pad image to 572*572
            fullset_X.append(self.pad_to_576(image, "image"))
        for idx, name in enumerate(self.Y_files):
            # print("mask name ",name)
            if 'tif' not in name:
                continue
            mask_path = os.path.join(self.Y_dir, name)
            mask = imread(mask_path)
            mask = cv.normalize(mask, None, 0, 1, cv.NORM_MINMAX) # normalize to [0,1]
            fullset_Y.append(self.pad_to_576(mask, "mask"))

        # train_val_test index
        train_idx = []
        val_idx = []
        test_idx = []
        csv_path = self.data_path + csv_name # "/image_and_split.csv"

        with open(csv_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, None)  # skip the headers
            for i, row in enumerate(spamreader):
                #index
                flag = int(row[1])
                if flag == 1:
                    train_idx.append(i)
                elif flag == 0:
                    val_idx.append(i)
                else:
                    test_idx.append(i)

        if mode == "train":
            self.X = [fullset_X[i] for i in train_idx]
            self.Y = [fullset_Y[i] for i in train_idx]
            self.X = self.make_even(self.X)
            self.Y = self.make_even(self.Y)
        if mode == "val":
            self.X = [fullset_X[i] for i in val_idx]
            self.Y = [fullset_Y[i] for i in val_idx]
            self.X = self.make_even(self.X)
            self.Y = self.make_even(self.Y)
        if mode == "test":
            self.X = [fullset_X[i] for i in test_idx]
            self.Y = [fullset_Y[i] for i in test_idx]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]
        
        #apply augmentation
        if self.transform:
            sample = self.transform(image=x, mask=y)
            x, y = sample['image'], sample['mask']

        #to_tensor
        x = torch.tensor(x,dtype=torch.float64).unsqueeze_(0)
        x = x.repeat(3, 1, 1) #torch.Size([3, 572, 572])
        #print("x", x.shape)
        y = torch.tensor(y,dtype=torch.float64).unsqueeze_(0)
        y = y.repeat(1, 1, 1)

        if self.preprocess:
          x =  self.preprocess(x)
        
        return { 'image': x, 'mask': y}



# This one is for inference, you only have images, no masks
# I use this for data to predict absorptivity
class KeyholeNoMask(Dataset):
    @staticmethod
    def pad_to_576(image):
        # pad val
        pad_val = np.mean(image, axis=(0, 1))
        # pad dimension
        dimension = 576 #572 # goal (572, 572)
        height = image.shape[0]
        width = image.shape[1]
        d_height = dimension - height
        d_width = dimension - width
        
        top, bottom, left, right = 0, 0, 0, 0
        if d_height > 0:
            top = d_height // 2
            bottom = d_height - top
        if d_width > 0:
            left = d_width // 2
            right = d_width - left

        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, value = pad_val)
        return image

    def __init__(self, data_path, transform=None, preprocess=None):
        '''
        transform -> data augmentation
        preprocess -> normalize 3 channel same as imagenet
        '''

        self.data_path = data_path
        self.transform = transform
        self.preprocess = preprocess

        # X represents images
        self.X_dir = os.path.join(self.data_path, "images")
        self.X_files = sorted(os.listdir(self.X_dir))
        # full dataset_X and dataset_Y
        fullset_X = []
        for idx, name in enumerate(self.X_files):
            if 'tif' not in name:
                continue
            # print("image name ",name)
            img_path = os.path.join(self.X_dir, name)
            # Use you favourite library to load the image
            image = imread(img_path) 
            # pad image to 572*572
            fullset_X.append(self.pad_to_576(image))
        self.X = fullset_X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        x = self.X[idx]
        
        #apply augmentation
        if self.transform:
            x = self.transdorm(x)

        #to_tensor
        x = torch.tensor(x,dtype=torch.float64).unsqueeze_(0)
        x = x.repeat(3, 1, 1) #torch.Size([3, 572, 572])

        if self.preprocess:
          x =  self.preprocess(x)
        
        return x