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

No external packages required in the script and installed.

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

0.9028407373828952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

OUTPUT_DIR = "./"
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'

## === cell 1
import sys
sys.path.append('../input/pytorchimagemodels/')
sys.path.append('../input/pretrainedmodels/')
sys.path.append('../input/facebook/')

import time
import random
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, 
    )
from albumentations.pytorch import ToTensorV2

import timm
from timm.models.vision_transformer import VisionTransformer
from pretrainedmodels import se_resnext101_32x4d
from models import DistilledVisionTransformer

import warnings 
warnings.filterwarnings('ignore')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1076266598.py in <cell line: 0>()
     22 from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
     23 from torch.utils.data import DataLoader, Dataset
---> 24 from albumentations import (
     25     Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
     26     RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,

ImportError: cannot import name 'RandomBrightness' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## === cell 3
def seed_torch(seed=1006):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch()

## === cell 4
test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
test.head()

## === cell 5
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

## === cell 6
def get_transforms(*, data, vit=False):
    
    if vit:
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    
    if data == 'train':
        return Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])

## === cell 7
class NetVit(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if no_att:
                pass
            else:
                self.att_layer = nn.Linear(n_features, 1)
            
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output

## === cell 8
class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)
            
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])

        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)

        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)

        h = h1 * w[:, 0].unsqueeze(-1) + h2 * w[:, 1].unsqueeze(-1) + h3 * w[:, 2].unsqueeze(-1) + h4 * w[:, 3].unsqueeze(-1)
        output = self.head(h)
        return output

## === cell 9
from collections import OrderedDict

def inference(model, states, test_loader, device, temp=1):
    model.to(device)
    preds = []
    for state in states:
        pred = []
        model.load_state_dict(state)
        model.eval()
        for i, image in enumerate(test_loader):
            with torch.no_grad():
                pred.append((model(image.to(device))*temp).softmax(1).to('cpu'))
        pred = torch.cat(pred, dim=0)
        preds.append(pred.numpy())
    return np.mean(preds, axis=0)

## === cell 10
def multi2single(path, se=False):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('se_module', 'dummy')
            k = k.replace('module.', '')
            k = k.replace('dummy', 'se_module')
        if 'attention_linear' in k:
            k = k.replace('attention_linear', 'att_layer')
        new_state_dict[k] = v
    return new_state_dict

## === cell 11
temp = 1.0

## === cell 12
    MODEL_NAME = "vit_base_patch16_384"
    MODEL_NUM = "No3001"
    MODEL_DIR = "../input/cassavamodels/"
    IMG_SIZE = 384
    TTA = 5
    BATCH = 32

    model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit_predictions = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit_predictions += inference(model, states, test_loader, device, temp) / TTA

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/221323523.py in <cell line: 0>()
      6 BATCH = 32
      7 
----> 8 model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
      9 states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
     10 if TTA == 1:

/tmp/ipykernel_11/363392660.py in __init__(self, model_name, pretrained, n_class, att_activate, no_att)
      2     def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False):
      3         super().__init__()
----> 4         self.model = timm.create_model(model_name, pretrained=pretrained)
      5         n_features = self.model.head.in_features
      6         self.model.head = nn.Identity()

NameError: name 'timm' is not defined

## === cell 13
if True:
    MODEL_NAME = "vit_base_patch16_224"
    MODEL_NUM = "vit4_ex"
    MODEL_DIR = "../input/cassavamodels/"
    IMG_SIZE = 448
    TTA = 5
    BATCH = 32
    att_activate = False

    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)    
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit4_predictions_a = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit4_predictions_a += inference(model, states, test_loader, device, temp) / TTA

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1633024503.py in <cell line: 0>()
      8     att_activate = False
      9 
---> 10     model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
     11     states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
     12     if TTA == 1:

/tmp/ipykernel_11/2285671094.py in __init__(self, model_name, pretrained, n_class, att_activate)
      2     def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
      3         super().__init__()
----> 4         self.model = timm.create_model(model_name, pretrained=pretrained)
      5         n_features = self.model.head.in_features
      6         self.model.head = nn.Identity()

NameError: name 'timm' is not defined

## === cell 14
if True:
    MODEL_NAME = "vit_base_patch16_224"
    MODEL_NUM = "vit4_ex_smooth001_att_act"
    MODEL_DIR = "../input/cassavamymodels/"
    IMG_SIZE = 448
    TTA = 5
    BATCH = 32
    att_activate = True

    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)    
    states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
    if TTA == 1:
        test_dataset = TestDataset(test, transform=get_transforms(data='valid', vit=True))
    else:
        test_dataset = TestDataset(test, transform=get_transforms(data='train', vit=True))

    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    vit4_predictions_b = np.zeros((len(test), 5))
    for _ in range(TTA):
        vit4_predictions_b += inference(model, states, test_loader, device, temp) / TTA

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1690152143.py in <cell line: 0>()
      8     att_activate = True
      9 
---> 10     model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
     11     states = [multi2single(MODEL_DIR+f'{MODEL_NUM}_{fold+1}.pth') for fold in range(5)]
     12     if TTA == 1:

/tmp/ipykernel_11/2285671094.py in __init__(self, model_name, pretrained, n_class, att_activate)
      2     def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
      3         super().__init__()
----> 4         self.model = timm.create_model(model_name, pretrained=pretrained)
      5         n_features = self.model.head.in_features
      6         self.model.head = nn.Identity()

NameError: name 'timm' is not defined

## === cell 15
predictions = (vit_predictions * 0.45 + vit4_predictions_a * 0.55) / 9 * 10 + vit4_predictions_b * 0.08

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3844701991.py in <cell line: 0>()
----> 1 predictions = (vit_predictions * 0.45 + vit4_predictions_a * 0.55) / 9 * 10 + vit4_predictions_b * 0.08

NameError: name 'vit_predictions' is not defined

## === cell 16
test['label'] = predictions.argmax(1)
test[['image_id', 'label']].to_csv(OUTPUT_DIR+'submission.csv', index=False)
test.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4126994870.py in <cell line: 0>()
      1 # submission
----> 2 test['label'] = predictions.argmax(1)
      3 test[['image_id', 'label']].to_csv(OUTPUT_DIR+'submission.csv', index=False)
      4 test.head()

NameError: name 'predictions' is not defined
