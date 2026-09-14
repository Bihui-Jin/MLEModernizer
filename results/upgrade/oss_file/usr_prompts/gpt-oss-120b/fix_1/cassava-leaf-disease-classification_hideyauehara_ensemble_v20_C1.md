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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8919613176186159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import glob

## === cell 2











pretrained_models = glob.glob(f'../input/eb7slseed70/efficientnet-b7sl_SEED70.orig/*.pth')


print(f'{len(pretrained_models)} models found.')
print('\n'.join(np.sort(pretrained_models)))

## === cell 3
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms  # 学習済みモデル、画像変換
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle


from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

SEED = 42
seed_everything(seed=SEED)

## === cell 4
import sys
sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
from efficientnet_pytorch import EfficientNet

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1153845020.py in <cell line: 0>()
      1 import sys
      2 sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
----> 3 from efficientnet_pytorch import EfficientNet

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 5
SIZE = 512        # image size
num_classes = 5

## === cell 6
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用デバイス: {device}')

## === cell 7
if os.getenv('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
    print("Test run in Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
    TEST_PATH = f'{BASE_DIR}/train_images'
    test_files = os.listdir(f'{TEST_PATH}/')[:32]
elif os.getenv('KAGGLE_KERNEL_RUN_TYPE') == 'Batch':
    print("In Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
    TEST_PATH = f'{BASE_DIR}/test_images'
    test_files = os.listdir(f'{TEST_PATH}/')
else:
    print("In the local environment.")
    BASE_DIR = "data"
    TEST_PATH = f'{BASE_DIR}/train_images'
    test_files = os.listdir(f'{TEST_PATH}/')[:32]     # glob.glob() と違ってファイル名だけが入る
    PRETRAINED_MODEL_PATH="1610594484mf/bak/00個別5fold計算"

print(f"Number of test  images: {len(test_files)}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1495821646.py in <cell line: 0>()
     16     BASE_DIR = "data"
     17     TEST_PATH = f'{BASE_DIR}/train_images'
---> 18     test_files = os.listdir(f'{TEST_PATH}/')[:32]     # glob.glob() と違ってファイル名だけが入る
     19     PRETRAINED_MODEL_PATH="1610594484mf/bak/00個別5fold計算"
     20 

FileNotFoundError: [Errno 2] No such file or directory: 'data/train_images/'

## === cell 8
df_test = pd.DataFrame(test_files, columns=['image_id'])
df_test['label'] = 1

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2191186498.py in <cell line: 0>()
      1 # 予測結果を格納する DataFrame
----> 2 df_test = pd.DataFrame(test_files, columns=['image_id'])
      3 df_test['label'] = 1

NameError: name 'test_files' is not defined

## === cell 9
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/310267853.py in <cell line: 0>()
----> 1 if len(df_test) == 1:
      2     df_test.loc[1] = df_test.loc[0]
      3     print(df_test)

NameError: name 'df_test' is not defined

## === cell 10

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = {
    'test': [
        Compose([
            A.CenterCrop(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
        Compose([
            A.HorizontalFlip(p=1),
            A.CenterCrop(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
        Compose([
            A.RandomResizedCrop(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
        Compose([
            A.RandomResizedCrop(SIZE, SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
        Compose([
            A.RandomResizedCrop(SIZE, SIZE),
            A.VerticalFlip(p=1),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
        Compose([
            A.Rotate(p=1),
            A.RandomResizedCrop(SIZE, SIZE),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.),
    ]
}

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/987991875.py in <cell line: 0>()
     18         ], p=1.),
     19         Compose([
---> 20             A.RandomResizedCrop(SIZE, SIZE),
     21             A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
     22             ToTensorV2(p=1.0),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 12

class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        '''
        model: 学習済みモデルを指定
        '''
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha
        
    def forward(self, inputs, labels, phase):
        if phase == 'val':
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            
            return outputs, loss

        if phase == 'test':
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            
            return outputs
        
        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs             # torch.Size([64, 3, 256, 256])
        x2 = inputs[index]      # torch.Size([64, 3, 256, 256])

        x1 = self.convlayer(x1) # torch.Size([64, 512, 1, 1])
        x2 = self.convlayer(x2) # torch.Size([64, 512, 1, 1])
        
        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 512, 1, 1])
        mixed_x = mixed_x.squeeze()          # torch.Size([64, 512])
        outputs = self.fc(mixed_x)           # torch.Size([64, 5])
        
        labels_a = labels
        labels_b = labels[index]
        
        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(pred, labels_b)
        
        return outputs, loss, labels_a, labels_b, lam

## === cell 13

class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        '''
        model: 学習済みモデルを指定
        '''
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha
        
    def forward(self, inputs, labels, phase):
        if phase == 'val':
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == 'test':
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)

            return outputs
        
        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs             # torch.Size([12, 3, 512, 512])
        x2 = inputs[index]      # torch.Size([12, 3, 512, 512])

        x1 = self.convlayer(x1) # torch.Size([12, 1920, 16, 16])
        x2 = self.convlayer(x2) # torch.Size([12, 1920, 16, 16])

        x1 = self.AdaptiveAvgPool2d(x1)  # torch.Size([12, 1920, 1, 1])
        x2 = self.AdaptiveAvgPool2d(x2)  # torch.Size([12, 1920, 1, 1])

        mixed_x = lam * x1 + (1 - lam) * x2  # torch.Size([64, 1920, 1, 1])
        mixed_x = mixed_x.squeeze()          # torch.Size([64, 1920])
        outputs = self.fc(mixed_x)           # torch.Size([64, 5])

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(pred, labels_b)

        return outputs, loss, labels_a, labels_b, lam

## === cell 14
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
            
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)   
        
        self.model = model
        self.criterion = criterion
        
    def forward(self, inputs, labels, phase):
        if phase == 'val':
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == 'test':
            outputs = self.model(inputs)

            return outputs
        

        print('ここにきてはいけない')
        sys.exit()

## === cell 16

class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.image_ids = df.image_id.tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def load_image(self, image_id):
        img = cv2.imread(f'{TEST_PATH}/{image_id}')  # (H, W, C) の numpy.ndarray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # BGR => RGB に変換
        return img
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        
        img = self.load_image(image_id)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, image_id

## === cell 17
def predict_model (basename, net, dataloader):
    '''
    basename: 学習済みモデル名
    net     : 学習済みモデル
    '''

    model_start_time = time.time()
    
    net.to(device)
    net.eval()    # 検証モード
    torch.set_grad_enabled(False)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
    
    probability = []
    
    
    for phase in ['test']:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            outputs = net(inputs, False, 'test')
            _, preds = torch.max(outputs, 1)  # ラベルを予測

            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())


    
    print(f'{basename} time: {time.time() - model_start_time:.2f}[sec]')

    return np.concatenate(probability)

## === cell 18
probability = []

start_time = time.time()

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]


    criterion = nn.CrossEntropyLoss()

    if 'resnet18' in basename:
        MODEL_NAME = 'resnet18'
        net = models.resnet18(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
    elif 'resnet50' in basename:
        MODEL_NAME = 'resnet50'
        net = models.resnet50(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
    elif 'resnet152' in basename:
        MODEL_NAME = 'resnet152'
        net = models.resnet152(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    elif 'resnext101' in basename:
        MODEL_NAME = 'resnext101'
        net = models.resnext101_32x8d(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12    #  3063MiB
    elif 'densenet201' in basename:
        MODEL_NAME = 'densenet201'
        net = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif 'efficientnet-b7' in basename:
        MODEL_NAME = 'efficientnet-b7'
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        print(f'{basename} is not supported.')
        sys.exit()
    
    print(f'{basename}: {MODEL_NAME}')    
    
    if MODEL_NAME == 'efficientnet-b7':
        net.load_state_dict(torch.load(pretrained_model))
    else:
        net.load_state_dict(torch.load(pretrained_model))
        
    
    for param in net.parameters():
        param.requires_grad = False
        
    for tid, transform_ in enumerate(transform['test']):
        print(f'transform loop={tid}')
        dataset = {
            'test': TestDataset(df_test, transform=transform_),
        }
        dataloader = {
            'test': torch.utils.data.DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True),
        }

        proba = predict_model(basename, net, dataloader)
        probability.append(proba)
    
    
    del net
    torch.cuda.empty_cache()
    
    
df_test['mean'] = np.array(probability).mean(axis=0).argmax(axis=1)

print(f'total time: {time.time() - start_time:.2f}[sec]')

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/316170966.py in <cell line: 0>()
     77     #break
     78 
---> 79 df_test['mean'] = np.array(probability).mean(axis=0).argmax(axis=1)
     80 
     81 print(f'total time: {time.time() - start_time:.2f}[sec]')

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 19
if len(df_test) == 2 and df_test.loc[0, 'image_id'] == df_test.loc[1, 'image_id']:
    df_test = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
else:
    df_test['label'] = df_test['mean']

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3063178855.py in <cell line: 0>()
----> 1 if len(df_test) == 2 and df_test.loc[0, 'image_id'] == df_test.loc[1, 'image_id']:
      2     df_test = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
      3 else:
      4     df_test['label'] = df_test['mean']

NameError: name 'df_test' is not defined

## === cell 20
df_test

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1186045382.py in <cell line: 0>()
----> 1 df_test

NameError: name 'df_test' is not defined

## === cell 21
df_test[['image_id', 'label']].to_csv('submission.csv', index=False)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3176383531.py in <cell line: 0>()
      1 # submission.csv に保存
----> 2 df_test[['image_id', 'label']].to_csv('submission.csv', index=False)

NameError: name 'df_test' is not defined
