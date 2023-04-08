import os
import torch
from tifffile import imread
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor


train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(7),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# test_process for both val and test
test_preprocess =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Keyhole(Dataset):
  def __init__(self, data_path, transform=train_preprocess, train=True, split="1"):

    self.X_dir = data_path + "/images/"
    self.X_files = sorted(os.listdir(self.X_dir))
    print(self.X_files)
    fullset_X = []
    for idx, name in enumerate(self.X_files):
        if 'tif' not in name:
            continue
        img_name = self.X_dir + str(name)
        image = imread(img_name)
        fullset_X.append(image)

    # full dataset_Y
    fullset_Y = []
    train_idx = []
    val_idx = []
    # train and val split
    csv_path = data_path + "/labels_and_split_" + split + ".csv" 
  
    with open(csv_path, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      next(spamreader, None)  # skip the headers
      for i, row in enumerate(spamreader):
        # [LaserType	RelativeAbsorption	TrainValSplit]
        fullset_Y.append(float(row[1])) 
        # train val index
        flag = int(row[2])
        if flag == 1:
            train_idx.append(i)
        else:
            val_idx.append(i)
    
    print("len fullset_X",len(fullset_X))
    print("len train_idx",len(train_idx))
    print("len val_idx",len(val_idx))

    if train:
      self.X = [fullset_X[i] for i in train_idx]
      self.Y = [fullset_Y[i] for i in train_idx]
    else:
      self.X = [fullset_X[i] for i in val_idx]
      self.Y = [fullset_Y[i] for i in val_idx]

    self.transform = transform

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):

    x = torch.tensor(self.X[idx],dtype=torch.float64).unsqueeze_(0)
    x = x.repeat(3, 1, 1)
    
    y = torch.tensor(self.Y[idx])
    return self.transform(x), y


class Keyhole_Test(Dataset):
  """
  Difference for Keytest dataset is that it doesn't have train/val split
  """
  def __init__(self, data_path, partition='test', transform=test_preprocess):

    self.X_dir = data_path + "/images/"
    self.X_files = sorted(os.listdir(self.X_dir))
    print(self.X_files)
    # full dataset_X
    fullset_X = []
    for idx, name in enumerate(self.X_files):
        if 'tif' not in name:
            continue
        #print(name)
        img_name = self.X_dir + str(name)
        # Use you favourite library to load the image
        image = imread(img_name)
        #image.unsqueeze_(0)
        fullset_X.append(image)

    # full dataset_Y
    fullset_Y = []
    csv_path = data_path + "/labels.csv"
  
    with open(csv_path, newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      next(spamreader, None)  # skip the headers
      for row in spamreader:
        fullset_Y.append(float(row[-1])) # relative_absorption
    
    # X
    print("len fullset_X",len(fullset_X))
    
    self.X = fullset_X
    self.Y = fullset_Y
    self.transform = transform
    assert(len(self.X) == len(self.Y)), "X and Y length doesn't match"

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    x = torch.tensor(self.X[idx],dtype=torch.float64).unsqueeze_(0)
    x = x.repeat(3, 1, 1)
    
    y = torch.tensor(self.Y[idx])
    return self.transform(x), y
