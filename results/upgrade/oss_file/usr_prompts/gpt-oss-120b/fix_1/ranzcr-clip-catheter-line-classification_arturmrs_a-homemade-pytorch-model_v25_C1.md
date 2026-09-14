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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.6238850720798981

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2

import os
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip
)
from albumentations.pytorch import ToTensorV2

import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


## === cell 3
seed = 42
np.random.seed(42)
torch.cuda.manual_seed_all(seed)

## === cell 5
class Config:
    img_train_path = '../input/ranzcr-clip-catheter-line-classification/train'
    img_test_path = '../input/ranzcr-clip-catheter-line-classification/test'
    
    batch_size = 64
    img_width= 256
    img_height = 256
    
    target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']
    
    n_classes = len(target_cols)
    n_workers = 8

## === cell 7
class ImageDataset(Dataset):
    
    def __init__(self, df, mode):
        super().__init__()
        self.filenames = df['StudyInstanceUID'].values
        self.labels = df[Config.target_cols].values
        self.len = len(df)
        self.transform = self.train_transforms() if mode == 'train' else self.valid_transforms() if mode == 'valid' else None
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = f'{Config.img_train_path}/{filename}.jpg'
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(self.labels[idx]).float()
        return image, label
    
    def __len__(self):
        return self.len
    
    def train_transforms(self):
        return Compose([
            Resize(Config.img_width, Config.img_height),
            RandomResizedCrop(Config.img_width, Config.img_height, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def valid_transforms(self):
        return Compose([
            Resize(Config.img_width, Config.img_height),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

## === cell 9
class SimpleModel(nn.Module):
    def __init__(self, n_classes=Config.n_classes):
        super(SimpleModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.dense(x)
        return x

## === cell 11
catherer_path = '/kaggle/input/ranzcr-clip-catheter-line-classification'
train_path = os.path.join(catherer_path,'train')
test_path = os.path.join(catherer_path,'test')

## === cell 12
train_csv_path = os.path.join(catherer_path,'train.csv')
train_df = pd.read_csv(train_csv_path)

classes = [col for col in train_df.columns if col not in ['StudyInstanceUID','PatientID']]
print(f'There are {len(classes)} to predict')

print(f"Shape of train dataframe : {train_df.shape}")
print(f"check for null values: {train_df.isnull().sum().sum()}")

test_csv_path = os.path.join(catherer_path,'sample_submission.csv')
test_df = pd.read_csv(test_csv_path)
test_filenames = test_df.StudyInstanceUID

print(f"Shape of train dataframe : {test_df.shape}")
print(f"check for null values: {test_df.isnull().sum().sum()}")

## === cell 13
train_df.head()

## === cell 15
img_path = Config.img_train_path + '/' + train_df['StudyInstanceUID'].iloc[0] + '.jpg'
img_example = Image.open(img_path)
print(f"Image size = {img_example.size}")
plt.figure(figsize=(12,8))
plt.imshow(img_example,cmap='Greys');

## === cell 16
img_example_red = img_example.resize((Config.img_width, Config.img_height))
plt.figure(figsize=(12,8))
plt.imshow(img_example_red,cmap='Greys');

## === cell 18

lim = True
if lim:
    red_train_df = train_df.sample(frac=0.04)
else:
    red_train_df = train_df.copy()
print(red_train_df.shape)

## === cell 19

n_min = 100
count_classes = red_train_df[classes].sum()
ext_train_df = [red_train_df]
for pred_class in classes:
    if count_classes[pred_class] < n_min:
        new_df = red_train_df[red_train_df[pred_class]==1].sample(n_min,replace=True)
        ext_train_df.append(new_df)
        
ext_train_df = pd.concat(ext_train_df)
print(ext_train_df.shape)

## === cell 20
plt.figure(figsize=(10,6))
graph = sns.barplot(x=classes,y=ext_train_df[classes].sum())
graph.set_xticklabels(graph.get_xticklabels(), rotation=90);

## === cell 22
X_train, X_valid = train_test_split(ext_train_df, test_size=0.2, shuffle=True)
print(f' X_train shape: {X_train.shape} and X_valid shape: {X_valid.shape}')

## === cell 23
train_dataset = ImageDataset(df=X_train, mode='train')
valid_dataset = ImageDataset(df=X_valid, mode='valid')
test_dataset = ImageDataset(df=test_df, mode='valid')

train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=Config.batch_size, num_workers=Config.n_workers, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, pin_memory=True, batch_size=Config.batch_size, num_workers=Config.n_workers, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, num_workers=Config.n_workers, shuffle=False)


model = SimpleModel()
optimizer = torch.optim.AdamW(model.parameters())
loss = nn.BCELoss()



## --- ERROR in cell 23, traceback:
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

ValidationError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1696501975.py in <cell line: 0>()
----> 1 train_dataset = ImageDataset(df=X_train, mode='train')
      2 valid_dataset = ImageDataset(df=X_valid, mode='valid')
      3 test_dataset = ImageDataset(df=test_df, mode='valid')
      4 
      5 train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=Config.batch_size, num_workers=Config.n_workers, shuffle=True)

/tmp/ipykernel_11/4291815628.py in __init__(self, df, mode)
      6         self.labels = df[Config.target_cols].values
      7         self.len = len(df)
----> 8         self.transform = self.train_transforms() if mode == 'train' else self.valid_transforms() if mode == 'valid' else None
      9 
     10     def __getitem__(self, idx):

/tmp/ipykernel_11/4291815628.py in train_transforms(self)
     27         return Compose([
     28             Resize(Config.img_width, Config.img_height),
---> 29             RandomResizedCrop(Config.img_width, Config.img_height, scale=(0.85, 1.0)),
     30             HorizontalFlip(p=0.5),
     31             Normalize(

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

ValueError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 24
train_history = []
valid_history = []

def train(model, loss, optimizer, train_dataloader, valid_dataloader, device='cuda', epochs=50):
    for e in range(1, epochs+1):
        for train_values, train_target in train_dataloader:
            train_values, train_target = train_values.to(device), train_target.to(device)
            
            optimizer.zero_grad()   
            model.train()
            output = model(train_values.float())
            loss_train = loss(output, train_target.float())
            loss_train.backward()
            optimizer.step()
        
        valid(model, loss, optimizer, valid_dataloader, device, epochs=1)
        train_history.append(loss_train)
        print(f'Epoch {e}: \ttrain loss {loss_train.item():.2f}')
            
def valid(model, loss, optimizer, valid_dataloader, device, epochs=50):
    for e in range(1, epochs+1):
        for val_values, val_target in valid_dataloader:
            val_values, val_target = val_values.to(device), val_target.to(device)
            
            model.eval()
            val_output = model(val_values.float())
            loss_val = loss(val_output, val_target.float())
        
        valid_history.append(loss_val)
        print(f'Epoch {e}: \tvalidation loss {loss_val.item():.2f}')
            
def predict_probs(filenames, model, device='cuda'):
    model.eval()

    transform = Compose([
            Resize(Config.img_width, Config.img_height),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
    ])
    
    predictions = []
    for filename in tqdm(filenames):
        filepath = f'{Config.img_test_path}/{filename}.jpg'
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image=image)['image']
        image = image.unsqueeze(0).to(device)
        predictions.append(model(image).detach().cpu().numpy())
    return predictions

## === cell 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Current device is {device}')
print(model)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/677910484.py in <cell line: 0>()
      1 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
----> 2 model.to(device)
      3 print(f'Current device is {device}')
      4 print(model)

NameError: name 'model' is not defined

## === cell 27
train(model, loss, optimizer, train_dataloader, valid_dataloader, device, epochs=1)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2257497873.py in <cell line: 0>()
      1 # Train the model
----> 2 train(model, loss, optimizer, train_dataloader, valid_dataloader, device, epochs=1)
      3 # valid(model, loss, optimizer, valid_dataloader, device, epochs=10)

NameError: name 'model' is not defined

## === cell 29
torch.save(model.state_dict(), 'simple_model.pt')

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/95516468.py in <cell line: 0>()
----> 1 torch.save(model.state_dict(), 'simple_model.pt')

NameError: name 'model' is not defined

## === cell 30
plt.plot(range(len(train_history)), train_history)
plt.plot(range(len(valid_history)), valid_history)
plt.show();

## === cell 32
predictions = predict_probs(test_filenames, model)
print(f'Predictions type = {type(predictions)}')
print(f'Predictions size = {len(predictions)}')
with open("preds.pkl", "wb") as fp:
    pickle.dump(predictions, fp)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522030064.py in <cell line: 0>()
----> 1 predictions = predict_probs(test_filenames, model)
      2 print(f'Predictions type = {type(predictions)}')
      3 print(f'Predictions size = {len(predictions)}')
      4 with open("preds.pkl", "wb") as fp:
      5     pickle.dump(predictions, fp)

NameError: name 'model' is not defined

## === cell 35
print(predictions[0])


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2987579472.py in <cell line: 0>()
      1 # Some info
----> 2 print(predictions[0])
      3 # print(pred_probabilities[0])
      4 # print(pred_probabilities[100])

NameError: name 'predictions' is not defined

## === cell 37
preds_one_len = []
for pred in predictions:
    preds_one_len.append(pred.squeeze())

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2238615318.py in <cell line: 0>()
      1 preds_one_len = []
----> 2 for pred in predictions:
      3     preds_one_len.append(pred.squeeze())

NameError: name 'predictions' is not defined

## === cell 38
pred_df = pd.DataFrame(columns=Config.target_cols, data=preds_one_len, index=test_df.index)
pred_df = pd.concat([test_df['StudyInstanceUID'],pred_df],axis=1)
pred_df

## === cell 39
pred_df.to_csv('submission.csv',index=False)
