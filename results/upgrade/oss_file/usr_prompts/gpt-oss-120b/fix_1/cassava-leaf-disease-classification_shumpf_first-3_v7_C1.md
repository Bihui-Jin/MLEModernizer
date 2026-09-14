# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from torchvision.models import resnext50_32x4d as resnext
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
print('gpu,',torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## === cell 1
class LeafNet(nn.Module): 
    def __init__(self, hidden_size=1280,num_cls=5):
        super(LeafNet, self).__init__()
        self.model_base = resnext().double()
        self.linear1 = nn.Linear(1000,hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size,num_cls)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss() 
        self.bn0 = nn.BatchNorm2d(1000)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        
    def forward(self,inputs):
        '''
        inputs: B x [img_size x img_size x num_channel]
        labels: B x num_cls
        '''
        img,labels = inputs
        x = self.model_base(img)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        ls = self.loss(x,labels)
        return ls
    
    def inference(self,inputs):
        '''
        inputs: B x [img_size x img_size x num_channel]
        labels: B x num_cls
        '''
        img = inputs
        x = self.model_base(img)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        x = torch.argmax(x,dim=1)
        return x

## === cell 2
'''
def one_hot(lst,num_cls):
    tables = np.zeros((len(lst),num_cls))
    for i,line in enumerate(lst):
        tables[i,line] = 1
    return tables
'''
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import albumentations as A
class Leafdataset(Dataset):
    def __init__(self,path,mode_train=False,num_cls=5):
        self.mode_train = mode_train
        '''
        if self.mode_train:
            image_base = os.path.join(path,'train_images/')
            csv_path = os.path.join(path,'train.csv')
        else:
            image_base = os.path.join(path,'test_images/')
            csv_path = os.path.join(path,'test.csv')
        '''
        
        image_base = os.path.join(path,'train_images/')
        csv_path = os.path.join(path,'train.csv')
        info = pd.read_csv(csv_path)
        
        labels = []
        for i in range(len(info['label'])):
            labels.append(info['label'][i])
        
        img_names = info['image_id']
        
        num_total = len(labels)
        imgs = list()
        
        for img_name in img_names:
            img_name = os.path.join(image_base,img_name)
            imgs.append(img_name)
            
        if self.mode_train:
            self.imgs = imgs#[:int(0.98*len(imgs))]
            self.labels = labels#[:int(0.*len(labels))]
        else:
            self.imgs = imgs[int(0.8*len(imgs)):]
            self.labels = labels[int(0.8*len(labels)):]
            
        self.preprocess = A.Compose([
                                              CenterCrop(256,256, p=1.),
                                              Resize(256,256),
                                              Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                                              ToTensorV2(p=1.0),
                                              ])
        self.augmentation = A.Compose([
            RandomResizedCrop(256, 256),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ])
        print(len(self.labels),len(self.imgs))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img_name = self.imgs[idx]
        img = Image.open(img_name)
        img = np.array(img)
        if self.mode_train:
            img = self.augmentation(image=img)['image'].float()
        else:
            img = self.preprocess(image=img)['image'].float()
        label = torch.tensor(self.labels[idx])
        return img,label

class Leafdataset_val(Dataset):
    def __init__(self,path,num_cls=5):
        import glob
        image_base = os.path.join(path,'test_images/')
        self.imgs = glob.glob(image_base+'*.jpg')
        
        
        self.preprocess = A.Compose([
                                      CenterCrop(256,256, p=1.),
                                      Resize(256,256),
                                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                                      ToTensorV2(p=1.0),
                                      ])
        print(len(self.imgs))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_name = self.imgs[idx]
        img = Image.open(img_name)
        img = np.array(img)
        
        img = self.preprocess(image=img)['image'].float()
        return img_name.split('/')[-1],img

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_10/186555024.py in <cell line: 0>()
      6     return tables
      7 '''
----> 8 from albumentations import (
      9     HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
     10     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

ImportError: cannot import name 'IAAPerspective' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 3

def validate_test(model,dataset_test):
    predictions = []
    ids = []
    
    for i,(img_name,img) in enumerate(dataset_test):
        img = img.unsqueeze(0).to(device)
        pred = model.inference(img).cpu().detach().item()
        predictions.append(pred)
        ids.append(img_name)
    sub = pd.DataFrame({'image_id': ids, 'label': predictions})
    sub.to_csv('./submission.csv', index = False)
        
def validate(model,dataset_test):
    model.eval()
    
    num_corr = 0
    num_total = len(dataset_test)
    for i,(img,label) in enumerate(dataset_test):
        img = img.unsqueeze(0).to(device)
        pred = model.inference(img).cpu().detach().item()
        
        label = label.cpu().detach().item()
        
        if pred==label:
            num_corr += 1
    
    print("accuracy is", num_corr*1.0/num_total, num_corr,'/',num_total)

## === cell 4
def inference(model, test_loader, device):
    model.to(device)
    
    predictions = []
    ids = []
    for i,(img_name,img) in enumerate(test_loader):
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
                
            pred = model.inference(img).cpu().detach().item()
            predictions.append(pred)
            ids.append(img_name)
 
    sub = pd.DataFrame({'image_id': ids, 'label': predictions})
    sub.to_csv('./submission.csv', index = False)
    sub.head()

## === cell 5
seed=42
torch.manual_seed(seed)

lr = 1e-4
batch_size = 50
num_epochs = 10
path = "../input/cassava-leaf-disease-classification/"
dataset_train = Leafdataset(path, mode_train=True)
dataset_test = Leafdataset(path, mode_train=False)
dataset_val = Leafdataset_val(path)
train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/686376253.py in <cell line: 0>()
      6 num_epochs = 10
      7 path = "../input/cassava-leaf-disease-classification/"
----> 8 dataset_train = Leafdataset(path, mode_train=True)
      9 dataset_test = Leafdataset(path, mode_train=False)
     10 dataset_val = Leafdataset_val(path)

NameError: name 'Leafdataset' is not defined

## === cell 6
model = LeafNet().float()
model.load_state_dict(torch.load('../input/first-2/ckpt_best.pt'), strict=True)
inference(model, dataset_val, device)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/3598660900.py in <cell line: 0>()
      2 model = LeafNet().float()
      3 #model = enet_v2(enet_type[i], out_dim=5)
----> 4 model.load_state_dict(torch.load('../input/first-2/ckpt_best.pt'), strict=True)
      5 #state_dict = torch.load('../input/first-2/ckpt_best.pt')
      6 #states = state_dict

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/first-2/ckpt_best.pt'
