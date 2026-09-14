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
seaborn==0.12.2
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

0.8292535509217286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
!pip install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar

## === cell 3
!ls -lrt ../input/ramki-cassava-weights/weight.pt

## === cell 4
!ls ../input/cassava-leaf-disease-classification

## === cell 6
Training = False

## === cell 7
import numpy as np 
import pandas as pd 
import os
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import models
import time
from tqdm import tqdm
import random
import sys
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate 
import albumentations as A


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/796341250.py in <cell line: 0>()
----> 1 from efficientnet_pytorch import EfficientNet
      2 from albumentations.pytorch import ToTensorV2
      3 from albumentations import Rotate
      4 import albumentations as A

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

## === cell 10
if torch.cuda.is_available():
    torch.cuda.device_count()

## === cell 11
writer = SummaryWriter('logs1')

## === cell 12
SEED = 42
N_FOLDS = 10
N_EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5


## === cell 13
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)

## === cell 14
base_path = '../input/cassava-leaf-disease-classification/'


## === cell 15
train_path =base_path + 'train_images/'
test_path = base_path + 'test_images/'

train_csv = pd.read_csv(base_path + 'train.csv')
sample = pd.read_csv(base_path + 'sample_submission.csv')

## === cell 16
train_csv.head()

## === cell 17

train_csv_disease = train_csv.label.map({0:"Cassava Bacterial Blight (CBB)",
1:"Cassava Brown Streak Disease (CBSD)",
2:"Cassava Green Mottle (CGM)",
3:"Cassava Mosaic Disease (CMD)",
4:"Healthy"})
diseases = train_csv_disease.value_counts()

## === cell 18
diseases

## === cell 19
diseases.plot.pie()

## === cell 21
class MyDataset(Dataset):
    
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe
        self.transforms = transforms
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.iloc[idx].label
        p = self.df.iloc[idx].image_id
        
        if self.test == False:
            p_path = train_path + p
        else:
            p_path = test_path + p
            
        image =  Image.open(p_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return image, label

## === cell 22
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

transforms_train = Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE),
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
        ], p=1.)
  
        
transforms_valid = Compose([
            CenterCrop(IMG_SIZE, IMG_SIZE, p=1.),
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/902508482.py in <cell line: 0>()
----> 1 from albumentations import (
      2     HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
      3     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
      4     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
      5     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize

ImportError: cannot import name 'IAAPerspective' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 23
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


## === cell 24
train_csv.shape

## === cell 25
model_name = 'efficientnet-b7'

if Training:
    model = EfficientNet.from_pretrained(model_name, num_classes=5) 
else:
    model = EfficientNet.from_name(model_name, num_classes=5) 
    weights_file = '../input/ramki-cassava-weights/weight.pt'
    model.load_state_dict(torch.load(weights_file)) 
    model.to(device)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1676513948.py in <cell line: 0>()
      4     model = EfficientNet.from_pretrained(model_name, num_classes=5)
      5 else:
----> 6     model = EfficientNet.from_name(model_name, num_classes=5)
      7     weights_file = '../input/ramki-cassava-weights/weight.pt'
      8     model.load_state_dict(torch.load(weights_file))

NameError: name 'EfficientNet' is not defined

## === cell 26
trainset      = MyDataset(train_csv, transforms=transforms_train, test=False)
train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset      = MyDataset(sample, transforms=transforms_valid, test=True)
test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2140704965.py in <cell line: 0>()
----> 1 trainset      = MyDataset(train_csv, transforms=transforms_train, test=False)
      2 train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
      3 
      4 testset      = MyDataset(sample, transforms=transforms_valid, test=True)
      5 test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

NameError: name 'transforms_train' is not defined

## === cell 27
len(train_loader)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006441096.py in <cell line: 0>()
----> 1 len(train_loader)

NameError: name 'train_loader' is not defined

## === cell 28
BATCH_SIZE

## === cell 29
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

## === cell 30
def train_model(model, epoch, dataloader_train, criterion, optimizer ):
    model.train() 
    
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        
        accs.update((output_train.argmax(1)==labels_train).sum().item()/imgs_train.size(0),imgs_train.size(0))
        losses.update(loss.item(), imgs_train.size(0))

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
        
    return losses.avg


def test_model(model, dataloader_valid, criterion):    
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = imgs.to(device), labels.to(device).long()
            output_valid = model(imgs_valid)
            
            loss = criterion(output_valid, labels_valid)

            losses.update(loss.item(), imgs_valid.size(0))
            accs.update((output_valid.argmax(1)==labels_valid).sum().item()/imgs_valid.size(0),imgs_valid.size(0))
            
            tk.set_postfix(loss=losses.avg,acc=accs.avg)
    

            
    return losses.avg,accs.avg

## === cell 31
X = train_csv.iloc[:,:-1]
y =  train_csv.iloc[:,-1:]

## === cell 32
if Training:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X,y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        valid = train_csv.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_csv.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)    


        dataset_train = MyDataset( train, transforms=transforms_train)
        dataset_valid = MyDataset( valid, transforms=transforms_valid)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, \
                                                               patience=1, verbose=True, min_lr=1e-5)

        best_acc = 0

        for epoch in range(N_EPOCHS):
            train_loss = train_model(model, epoch, dataloader_train, criterion, optimizer)
            val_loss, acc = test_model(model, dataloader_valid, criterion)

            writer.add_scalar('training loss',
                                train_loss,
                                i_fold * N_EPOCHS + epoch+1)
            writer.add_scalar('validation loss',
                                val_loss,
                                i_fold * N_EPOCHS + epoch+1)
            writer.add_scalar('validation Acc',
                                acc,
                                i_fold * N_EPOCHS + epoch+1)

            writer.flush()

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'weight.pt')

            print('current_val_acc:', acc, 'best_val_acc:', best_acc)



## === cell 33
test_pred = []

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        images, _ = data
        images = images.to(device)

        pred = model(images)

        pred = pred.argmax(1).cpu().detach().numpy().astype('int')

        test_pred.extend(pred)

sample.label = test_pred
sample.to_csv('submission.csv',index=False)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2930662760.py in <cell line: 0>()
      2 
      3 with torch.no_grad():
----> 4     for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
      5         images, _ = data
      6         images = images.to(device)

NameError: name 'test_loader' is not defined

## === cell 34
sample

## === cell 35
!cat submission.csv
