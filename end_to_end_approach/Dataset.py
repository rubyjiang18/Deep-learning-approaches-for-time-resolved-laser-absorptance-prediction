import os
import torch
from tifffile import imread
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import pandas as pd


train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.RandomRotation(7), # extra
    transforms.RandomHorizontalFlip(), # extra
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# test_process for both val and test
test_preprocess =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
Keyhole class for training and validation
"""
class Keyhole(Dataset):
  def __init__(self, data_path, transform=train_preprocess, train=True, split="1"):

    self.X_dir = data_path + "/images/"
    self.X_files = sorted(os.listdir(self.X_dir))
    #print(self.X_files)
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
    
    # X
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

    x = torch.tensor(self.X[idx],dtype=torch.float64).unsqueeze_(0) # , dtype=torch.float64
    x = x.repeat(3, 1, 1)
    
    y = torch.tensor(self.Y[idx])
    return self.transform(x), y

"""
Keyhole class for testing
"""
class Keyhole_Test(Dataset):
  def __init__(self, data_path, partition='test', transform=test_preprocess):

    self.X_dir = data_path + "/images/"
    self.X_files = sorted(os.listdir(self.X_dir))
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
    # transform not done yet
    # return self.transforms(self.X[idx]), torch.tensor(self.Y[idx])
    
    x = torch.tensor(self.X[idx],dtype=torch.float64).unsqueeze_(0) # , dtype=torch.float64
    x = x.repeat(3, 1, 1)
    
    y = torch.tensor(self.Y[idx])
    return self.transform(x), y
    #return x, y
    #return self.X[idx], self.Y[idx]

"""
Keyhole class for fine-tuning on the absorptance dataset with a powder layer
Get train, val, and test all from the same dataset
"""
class Keyhole_FewShot(Dataset):
  def __init__(self, data_path, partition='train', transform=None, ratio=0):
    """
    ratio means ratio for train set
    ratio 0 -> 0% train, val + test
    ratio 5 -> 5% train, 95% val + test
    ratio 10 -> 10% train, 90% val + test
    ratio 15 -> 15% train, 85% val + test
    ratio 20 -> 20% train, 80% val + test
    ratio 25 -> 25% train, 75% val + test
    ratio 30 -> 30% train, 70% val + test
    ratio 40 -> 40% train, 60% val + test

    partion in train, val, test
    """

    self.X_dir = data_path + "/images/"
    self.X_files = sorted(os.listdir(self.X_dir))
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
    train_idx = []
    val_idx = []
    test_idx = []
    # train and val split
    csv_path = data_path + "/labels_and_split.csv"
    df = pd.read_csv(csv_path)
    absorptance = df['RelativeAbsorption'].values
    fullset_Y = absorptance
    split = df['Index_' + str(ratio)+'train'].values
    for i in range(len(split)):
        #Rule: Train=0, val=1, test=2
        if split[i] == 0:
            train_idx.append(i)
        elif split[i] == 1:
            val_idx.append(i)
        else:
            test_idx.append(i)
    
    # X
    print("len fullset_X",len(fullset_X))
    print("len train_idx",len(train_idx))
    print("len val_idx",len(val_idx))
    print("len test_idx",len(test_idx))

    # assign X and Y
    fullset_X = fullset_X
    X_train = [fullset_X[i] for i in train_idx]
    Y_train = [fullset_Y[i] for i in train_idx]
    X_val = [fullset_X[i] for i in val_idx]
    Y_val = [fullset_Y[i] for i in val_idx]
    X_test = [fullset_X[i] for i in test_idx]
    Y_test = [fullset_Y[i] for i in test_idx]

    if partition == 'train':
      self.X = X_train
      self.Y = Y_train
    elif partition == 'val':
      self.X = X_val
      self.Y = Y_val
    elif partition == 'test':
      self.X = X_test
      self.Y = Y_test
    else:
        raise ValueError("Partition must be train, val, or test")

    self.transform = transform
    self.partition = partition

  def __len__(self):
     return len(self.X)
  
  def __getitem__(self, idx):
    x = torch.tensor(self.X[idx],dtype=torch.float64).unsqueeze_(0)
    x = x.repeat(3, 1, 1)
    
    y = torch.tensor(self.Y[idx])
    return self.transform(x), y
    
if __name__ == '__main__':
   batch_size = 16
   num_workers = 2
   base = "/Users/rubyjiang/Desktop/keyhole-absorptivity-prediction/absorption_data_all/powder_dataset/"
   train_dataset = Keyhole_FewShot(base, partition='train', ratio=0)
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)