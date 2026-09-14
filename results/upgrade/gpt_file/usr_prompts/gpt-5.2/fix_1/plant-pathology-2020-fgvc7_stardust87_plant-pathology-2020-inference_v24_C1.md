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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9648988014890484

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install timm

## === cell 2
import os,time

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import timm

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import warnings  
warnings.filterwarnings('ignore')



## === cell 3
DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'
IMAGE_INPUT = '/kaggle/input/plant-pathology-2020-resized-images'

SEED = 42
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 8
IMAGE_SIZE = (409,273)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
device

## === cell 5
class PlantDataset(Dataset):
    
    def __init__(self, df, transforms=None, test_set=False):
    
        self.df = df
        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms  = A.Compose([ToTensorV2(p=1.0)])
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = IMAGE_INPUT + '/images_409_273/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transforms(image=image)
        image = transformed['image']
        
        if not self.test_set:
            labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
            labels = torch.from_numpy(labels.astype(np.int8))
            labels = labels.unsqueeze(-1)
                
            return image, labels
        else:
            return image


## === cell 7
def trim_network_at_index(network,index=-1):
    assert index <0, f'Param index must be negative. Received {index}'
    return nn.Sequential(*list(network.children())[:index])

## === cell 8
class PlantModel(nn.Module):
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('resnest269e',pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone,-1) 
        self.logit = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)

        x = self.logit(x)

        return  x

## === cell 11
transforms_valid = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0)
])

test_df = pd.read_csv(DIR_INPUT + '/test.csv')
dataset_test = PlantDataset(df=test_df,test_set = True, transforms=transforms_valid)
testloader = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=4)

## === cell 13
def test_model(model_name,testloader):
    model_path = f'/kaggle/input/plant-pathology-2020-training/{model_name}.pth'
    model = PlantModel()
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)
    
    test_probs = []
    model.eval()

    test_iter = iter(testloader)

    with torch.no_grad():
        for i in tqdm(range(len(testloader))):        
            image = next(test_iter)
            probs = F.softmax(model(image.to(device)))
            test_probs.append(probs.cpu().numpy())


    test_probs = np.concatenate(test_probs, axis=0)
    
    return test_probs

## === cell 15
test_probs = []
start = time.perf_counter()
for i_fold in range(N_FOLDS):
    test_probs_fold = test_model(f'modelF{i_fold}',testloader)
    test_probs.append(test_probs_fold)
print(f'Finished Inference in {(time.perf_counter() - start):.2f} seconds')

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2091015252.py in <cell line: 0>()
      2 start = time.perf_counter()
      3 for i_fold in range(N_FOLDS):
----> 4     test_probs_fold = test_model(f'modelF{i_fold}',testloader)
      5     test_probs.append(test_probs_fold)
      6 print(f'Finished Inference in {(time.perf_counter() - start):.2f} seconds')

/tmp/ipykernel_11/2371201730.py in test_model(model_name, testloader)
      2     model_path = f'/kaggle/input/plant-pathology-2020-training/{model_name}.pth'
      3     model = PlantModel()
----> 4     model.load_state_dict(torch.load(model_path,map_location=device))
      5     model = model.to(device)
      6 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/plant-pathology-2020-training/modelF0.pth'

## === cell 16
test_probs_mean = [np.mean(k,axis=0) for k in zip(*test_probs)]


## === cell 17
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
submission_df.iloc[:, 1:] = 0
submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_probs_mean
submission_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1647523360.py in <cell line: 0>()
      1 submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
      2 submission_df.iloc[:, 1:] = 0
----> 3 submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_probs_mean
      4 submission_df.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4356 
   4357             else:
-> 4358                 self._iset_not_inplace(key, value)
   4359 
   4360     def _iset_not_inplace(self, key, value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _iset_not_inplace(self, key, value)
   4375         if self.columns.is_unique:
   4376             if np.shape(value)[-1] != len(key):
-> 4377                 raise ValueError("Columns must be same length as key")
   4378 
   4379             for i, col in enumerate(key):

ValueError: Columns must be same length as key

## === cell 18
submission_df
