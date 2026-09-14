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

0.8482925355092172

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
!ls -lrt ../input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth


## === cell 4
!ls ../input/cassava-leaf-disease-classification

## === cell 6
!ls ../input/cassava-leaf-disease-classification

## === cell 8
TRAINING = False
WEIGHT_FILE='../input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth'

## === cell 9
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
from torch.utils.data.sampler import WeightedRandomSampler


## --- ERROR in cell 9, traceback:
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

## === cell 10
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate 
import albumentations as A


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/796341250.py in <cell line: 0>()
----> 1 from efficientnet_pytorch import EfficientNet
      2 from albumentations.pytorch import ToTensorV2
      3 from albumentations import Rotate
      4 import albumentations as A

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 11
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

## === cell 12
if torch.cuda.is_available():
    print(torch.cuda.device_count())

## === cell 13
writer = SummaryWriter('logs/sampler-loss-aug')

## === cell 14
SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5


## === cell 15
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

## === cell 16
base_path = '../input/cassava-leaf-disease-classification/'


## === cell 17
train_path =base_path + 'train_images/'
test_path = base_path + 'test_images/'

train_csv = pd.read_csv(base_path + 'train.csv')
sample = pd.read_csv(base_path + 'sample_submission.csv')

## === cell 18
train_csv.head()

## === cell 19

train_csv_disease = train_csv.label.map({0:"Cassava Bacterial Blight (CBB)",
1:"Cassava Brown Streak Disease (CBSD)",
2:"Cassava Green Mottle (CGM)",
3:"Cassava Mosaic Disease (CMD)",
4:"Healthy"})
diseases = train_csv_disease.value_counts()

## === cell 20
diseases

## === cell 21
diseases.plot.pie()

## === cell 23
class CasavaDataset(Dataset):
    
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

## === cell 25
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

transforms_train = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            HorizontalFlip(p=0.3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
transforms_valid = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2373318471.py in <cell line: 0>()
----> 1 from albumentations import (
      2     HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
      3     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
      4     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
      5     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize

ImportError: cannot import name 'IAAPerspective' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 27
train_csv.shape

## === cell 28
model_name = 'efficientnet-b7'

if TRAINING:
    model = EfficientNet.from_pretrained(model_name, num_classes=5) 
else:
    model = EfficientNet.from_name(model_name, num_classes=5) 
    model.load_state_dict(torch.load(WEIGHT_FILE, map_location='cuda:0')) 
    model.to(device)
    print('model is loaded from weight file')

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/783004086.py in <cell line: 0>()
      4     model = EfficientNet.from_pretrained(model_name, num_classes=5)
      5 else:
----> 6     model = EfficientNet.from_name(model_name, num_classes=5)
      7     model.load_state_dict(torch.load(WEIGHT_FILE, map_location='cuda:0'))
      8     model.to(device)

NameError: name 'EfficientNet' is not defined

## === cell 30
layer =0
for child in model.children():
    layer+=1
print(layer)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824757308.py in <cell line: 0>()
      1 layer =0
----> 2 for child in model.children():
      3     layer+=1
      4 print(layer)

NameError: name 'model' is not defined

## === cell 35
trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/564013970.py in <cell line: 0>()
----> 1 trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
      2 train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
      3 
      4 testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
      5 test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

NameError: name 'transforms_train' is not defined

## === cell 36
len(train_loader)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4006441096.py in <cell line: 0>()
----> 1 len(train_loader)

NameError: name 'train_loader' is not defined

## === cell 37
len(test_loader)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2774873003.py in <cell line: 0>()
----> 1 len(test_loader)

NameError: name 'test_loader' is not defined

## === cell 38
BATCH_SIZE

## === cell 39
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

## === cell 40
def show_metrics(model, epoch, dataloader, criterion, optimizer ):
    model.train() 
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    complete_outputs = []
    complete_labels = []
    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        
        
        predicted_classes = output_train.argmax(1)
        complete_outputs.append(predicted_classes)
        complete_labels.append(labels_train)
        
        correctly_identified_sum = predicted_classes==labels_train_.sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum/number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
                    
                    
    cf_matrix = confusion_matrix(complete_outputs, complete_labels)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="YlGnBu")

    from sklearn.metrics import classification_report
    target_names = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)","Cassava Mosaic Disease (CMD)","Healthy"]
    print(classification_report(complete_outputs, complete_labels, target_names=target_names))

    return losses.avg, complete_outputs, complete_labels

## === cell 41
"Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)","Cassava Mosaic Disease (CMD)","Healthy"

## === cell 42
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
        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes==labels_train).sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum/number_of_images, number_of_images)
        
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
        
    return losses.avg, accs.avg


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

## === cell 43
X = train_csv.iloc[:,:-1]
y =  train_csv.iloc[:,-1:]

## === cell 44
if TRAINING:
    
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X,y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        train = train_csv.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)    

        valid = train_csv.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)









        dataset_train = CasavaDataset( train, transforms=transforms_train)
        dataset_valid = CasavaDataset( valid, transforms=transforms_valid)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True) #sampler=sampler
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, \
                                                               patience=1, verbose=True, min_lr=1e-5)

        best_acc = 0

        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_model(model, epoch, dataloader_train, criterion, optimizer)
            val_loss, acc = test_model(model, dataloader_valid, criterion)

            writer.add_scalar('training acc',
                                train_acc,
                                i_fold * N_EPOCHS + epoch+1)
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
                torch.save(model.state_dict(), 'weights1/weight-at-fold-{}-epoch-{}-acc-{:.5}.pth'.format(i_fold+1, epoch, best_acc))

            print('current_val_acc:', acc, 'best_val_acc:', best_acc)



## === cell 45
test_pred = []

model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        images, _ = data
        images = images.cuda()

        pred = model(images)

        pred = pred.argmax(1).cpu().detach().numpy().astype('int')

        test_pred.extend(pred)

sample.label = test_pred
sample.to_csv('submission.csv',index=False)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2939362416.py in <cell line: 0>()
      1 test_pred = []
      2 
----> 3 model.eval()
      4 
      5 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 46
sample

## === cell 47
!cat submission.csv
