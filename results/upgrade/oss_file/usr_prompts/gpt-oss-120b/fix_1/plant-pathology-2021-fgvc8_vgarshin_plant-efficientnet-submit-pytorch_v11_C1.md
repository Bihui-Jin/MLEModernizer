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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.831154201292706

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
%%time
!pip install ../input/efficientnet-pytorch/EfficientNet-PyTorch-1.0 -f ./ --no-index

## === cell 3
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from torch.utils.data.sampler import SequentialSampler
from efficientnet_pytorch import model as enet

KAGGLE = True
if not KAGGLE: os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
else: pass
DEVICE = torch.device('cuda')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1461565398.py in <cell line: 0>()
     14 from torchvision import models, transforms
     15 from torch.utils.data.sampler import SequentialSampler
---> 16 from efficientnet_pytorch import model as enet
     17 
     18 KAGGLE = True

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 4
TEST = True
VER = 'v4'
if KAGGLE:
    DATA_PATH = '../input/plant-pathology-2021-fgvc8'
    MDLS_PATH = f'../input/plant-models-{VER}'
else:
    DATA_PATH = './data'
    MDLS_PATH = f'./models_{VER}'
TH = .4
TTAS = [0, 1, 2]
FOLDS = [3, 4]
IMGS_PATH = f'{DATA_PATH}/test_images' if TEST else f'{DATA_PATH}/train_images'

start_time = time.time()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2633566556.py in <cell line: 0>()
      1 TEST = True
      2 VER = 'v4'
----> 3 if KAGGLE:
      4     DATA_PATH = '../input/plant-pathology-2021-fgvc8'
      5     MDLS_PATH = f'../input/plant-models-{VER}'

NameError: name 'KAGGLE' is not defined

## === cell 5
with open(f'{MDLS_PATH}/params.json') as file:
    params = json.load(file)
LABELS_ = params['labels_']
LABELS = params['labels']
WORKERS = 2 if KAGGLE else params['workers']
print('loaded params:', params)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2335846894.py in <cell line: 0>()
----> 1 with open(f'{MDLS_PATH}/params.json') as file:
      2     params = json.load(file)
      3 LABELS_ = params['labels_']
      4 LABELS = params['labels']
      5 WORKERS = 2 if KAGGLE else params['workers']

NameError: name 'MDLS_PATH' is not defined

## === cell 6
df_sub = pd.DataFrame(os.listdir(IMGS_PATH)) if TEST else pd.DataFrame(os.listdir(IMGS_PATH)[:100])
df_sub.columns = ['image']
df_sub['labels'] = 'healthy'
display(df_sub.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4034570782.py in <cell line: 0>()
----> 1 df_sub = pd.DataFrame(os.listdir(IMGS_PATH)) if TEST else pd.DataFrame(os.listdir(IMGS_PATH)[:100])
      2 df_sub.columns = ['image']
      3 df_sub['labels'] = 'healthy'
      4 display(df_sub.head())

NameError: name 'IMGS_PATH' is not defined

## === cell 7
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, ]
    elif axis == 2:
        return img[:, ::-1, ]
    elif axis == 3:
        return img[::-1, ::-1, ]
    else:
        return img

class PlantDataset(data.Dataset):
    
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f'{IMGS_PATH}/{img_name}'
        img = cv2.imread(img_path)
        if not np.any(img):
            print('no img file read:', img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())

class EffNet(nn.Module):
    
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.EfficientNet.from_name(params['backbone'])
        nc = self.enet._fc.in_features
        self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params['dropout']),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params['dropout']),
            nn.Linear(int(nc / 4), out_dim)
        )
        
    def extract(self, x):
        return self.enet(x)
    
    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

## === cell 8
models = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = '{}/model_best_{}.pth'.format(MDLS_PATH, n_fold)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.cuda()
    models.append(model)
    print('loaded:', path)
del state_dict, model
gc.collect();

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344335796.py in <cell line: 0>()
      1 models = []
----> 2 for n_fold in FOLDS:
      3     model = EffNet(params, out_dim=len(LABELS_))
      4     path = '{}/model_best_{}.pth'.format(MDLS_PATH, n_fold)
      5     state_dict = torch.load(path, map_location=torch.device('cpu'))

NameError: name 'FOLDS' is not defined

## === cell 9
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=params['img_size'],
        labels=None,
        transform=None,
        tta=tta)
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=params['batch_size'], 
        sampler=SequentialSampler(dataset), 
        num_workers=WORKERS)
    loaders.append(loader)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2787113230.py in <cell line: 0>()
      1 datasets, loaders = [], []
----> 2 for tta in TTAS:
      3     dataset = PlantDataset(
      4         df=df_sub,
      5         size=params['img_size'],

NameError: name 'TTAS' is not defined

## === cell 10
def get_labels(row, labels, th):
    try:
        row = [i for i, x in enumerate(row) if x > th]
        row = [labels[str(i)] for i in row]
        row = 'healthy' if ('healthy' in row or len(row) == 0) else ' '.join(row)
    except:
        print(row)
    return row

logits = []
with torch.no_grad():
    for i, model in enumerate(models):
        for j, loader in enumerate(loaders):
            logits_tta = []
            for img_data in loader:
                img_data = img_data.to(DEVICE)
                preds = np.squeeze(model(img_data).sigmoid().cpu().numpy())
                logits_tta.append(preds)
            print('model {} | loader {} -> done'.format(i, j))
            logits.append(logits_tta)
logits = np.mean(logits, axis=0)
logits = np.squeeze(np.vstack(logits))
df_sub['labels'] = [get_labels(x, LABELS, TH) for x in list(logits)]

elapsed_time = time.time() - start_time
print(f'time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec')

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3285020994.py in <cell line: 0>()
     20             logits.append(logits_tta)
     21 logits = np.mean(logits, axis=0)
---> 22 logits = np.squeeze(np.vstack(logits))
     23 df_sub['labels'] = [get_labels(x, LABELS, TH) for x in list(logits)]
     24 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in _vhstack_dispatcher(tup, dtype, casting)
    214 
    215 def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
--> 216     return _arrays_for_stack_dispatcher(tup)
    217 
    218 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in _arrays_for_stack_dispatcher(arrays)
    210                         'such as list or tuple.')
    211 
--> 212     return tuple(arrays)
    213 
    214 

TypeError: 'numpy.float64' object is not iterable

## === cell 11
print('value counts:')
print(df_sub.labels.value_counts())
df_sub.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2106293653.py in <cell line: 0>()
      1 print('value counts:')
----> 2 print(df_sub.labels.value_counts())
      3 df_sub.head()

NameError: name 'df_sub' is not defined

## === cell 12
df_sub.to_csv('submission.csv', index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/205353710.py in <cell line: 0>()
----> 1 df_sub.to_csv('submission.csv', index=False)

NameError: name 'df_sub' is not defined
