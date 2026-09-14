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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.12

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
numpy==1.26.4
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
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.679672632190417

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import cuml
from cuml.svm import SVR

from sklearn.decomposition import PCA

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/64805199.py in <cell line: 0>()
     10 from PIL import Image
     11 
---> 12 import cuml
     13 from cuml.svm import SVR
     14 

/usr/local/lib/python3.11/dist-packages/cuml/__init__.py in <module>
     25     del libcuml
     26 
---> 27 from cuml.internals.base import Base, UniversalBase
     28 from cuml.internals.available_devices import is_cuda_available
     29 

/usr/local/lib/python3.11/dist-packages/cuml/internals/__init__.py in <module>
     16 
     17 from cuml.internals.available_devices import is_cuda_available
---> 18 from cuml.internals.base_helpers import BaseMetaClass, _tags_class_and_instance
     19 from cuml.internals.api_decorators import (
     20     _deprecate_pos_args,

/usr/local/lib/python3.11/dist-packages/cuml/internals/base_helpers.py in <module>
     18 import typing
     19 
---> 20 from cuml.internals.api_decorators import (
     21     api_base_return_generic,
     22     api_base_return_array,

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in <module>
     22 
     23 # TODO: Try to resolve circular import that makes this necessary:
---> 24 from cuml.internals import input_utils as iu
     25 from cuml.internals.api_context_managers import BaseReturnAnyCM
     26 from cuml.internals.api_context_managers import BaseReturnArrayCM

/usr/local/lib/python3.11/dist-packages/cuml/internals/input_utils.py in <module>
     18 from typing import Literal
     19 
---> 20 from cuml.internals.array import CumlArray
     21 from cuml.internals.array_sparse import SparseCumlArray
     22 from cuml.internals.global_settings import GlobalSettings

/usr/local/lib/python3.11/dist-packages/cuml/internals/array.py in <module>
     19 import pickle
     20 
---> 21 from cuml.internals.global_settings import GlobalSettings
     22 from cuml.internals.logger import debug
     23 from cuml.internals.mem_type import MemoryType, MemoryTypeError

/usr/local/lib/python3.11/dist-packages/cuml/internals/global_settings.py in <module>
     18 import threading
     19 from cuml.internals.available_devices import is_cuda_available
---> 20 from cuml.internals.device_type import DeviceType
     21 from cuml.internals.mem_type import MemoryType
     22 from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

/usr/local/lib/python3.11/dist-packages/cuml/internals/device_type.py in <module>
     17 
     18 from enum import Enum, auto
---> 19 from cuml.internals.mem_type import MemoryType
     20 
     21 

/usr/local/lib/python3.11/dist-packages/cuml/internals/mem_type.py in <module>
     20 from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
     21 
---> 22 cudf = gpu_only_import("cudf")
     23 cp = gpu_only_import("cupy")
     24 cpx_sparse = gpu_only_import("cupyx.scipy.sparse")

/usr/local/lib/python3.11/dist-packages/cuml/internals/safe_imports.py in gpu_only_import(module, alt)
    360     """
    361     if GPU_ENABLED:
--> 362         return importlib.import_module(module)
    363     else:
    364         return safe_import(

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/cudf/__init__.py in <module>
     18 
     19 _setup_numba()
---> 20 validate_setup()
     21 
     22 import cupy

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:
---> 55             raise e
     56         # If there is no GPU detected, set `gpus_count` to -1
     57         gpus_count = -1

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     50 
     51     try:
---> 52         gpus_count = getDeviceCount()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:

/usr/local/lib/python3.11/dist-packages/rmm/_cuda/gpu.py in getDeviceCount()
    100     status, count = runtime.cudaGetDeviceCount()
    101     if status != runtime.cudaError_t.cudaSuccess:
--> 102         raise CUDARuntimeError(status)
    103     return count
    104 

CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version

## === cell 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, 'train.csv'))

test_df = pd.read_csv(os.path.join(directory, 'test.csv'))

print('Train samples: ', len(train_df),
      '\nTrain samples: ', len(test_df), '\n')


## === cell 3
def ExtractModelFeature(Dataloader, model, Train_PCA=False):
    X = []
    for img in tqdm(Dataloader):
        with torch.no_grad():
            if model.__class__.__name__ == 'EfficientNet':
                x = model(img.to(device))
            elif model.__class__.__name__ == 'CLIPModel':
                x = model.get_image_features(**img.to(device))
            elif model.__class__.__name__ == 'PCA':
                x = img
            else:
                raise Exception("Check if model is implimented !")
        
        X.append(x.cpu().detach().numpy())
                
    X = np.concatenate(X, axis=0)
    
    if model.__class__.__name__ == 'PCA':
        if Train_PCA:
            model.fit(X)
            X = model.transform(X)
            return X, model
        else:
            return model.transform(X)
        
    return X

## === cell 4
'''
class dataset_EfficientNet:
    def __init__(self, df, directory, transform, test=False):
        self.df = df
        self.directory = directory
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        img = Image.open(address).convert('RGB')
        image = self.transform(img) # transform and add batch dimension

            
        return image
'''

## === cell 5
'''
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

ckp_path = "/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b6/1/tf_efficientnet_b6_aa-80ba17e4.pth"
model_eff = timm.create_model('tf_efficientnet_b6', checkpoint_path=ckp_path)

config = resolve_data_config({}, model=model_eff)
transform = create_transform(**config)

model_eff = model_eff.to(device)
'''

## === cell 6
'''
train_dataset = dataset_EfficientNet(train_df, directory, transform, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

test_dataset = dataset_EfficientNet(test_df, directory, transform, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
'''

## === cell 7
'''
X1 = ExtractModelFeature(train_dataloader, model_eff)
X1_test = ExtractModelFeature(test_dataloader, model_eff)

print('train: ', X1.shape)
print('test: ', X1_test.shape)
'''

## === cell 8

from transformers import CLIPModel, CLIPProcessor

model_path_1 = "/kaggle/input/clip-vit/pytorch/b-32-laion2b-s34b-b79k/1"
model_path_2 = "/kaggle/input/clip-vit/pytorch/l-14-datacomp-xl-s13b-b90k/1"

model_clip_1 = CLIPModel.from_pretrained(model_path_1)
processor_clip_1 = CLIPProcessor.from_pretrained(model_path_1)


model_clip_1 = model_clip_1.to(device)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9

class dataset_Clip:
    def __init__(self, df, directory, processor, test=False):
        self.df = df
        self.directory = directory
        self.processor = processor
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        image = self.processor(images=Image.open(address), return_tensors="pt", padding=True)
        for key, val in image.items():
            image[key] = val.squeeze()
            
        return image


## === cell 10
'''
train_dataset = dataset_Clip(train_df, directory, processor_clip, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = dataset_Clip(test_df, directory, processor_clip, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
'''

## === cell 11
'''
X2 = ExtractModelFeature(train_dataloader, model_clip)
X2_test = ExtractModelFeature(test_dataloader, model_clip)

print('train: ', X2.shape)
print('test: ', X2_test.shape)
'''

## === cell 12

train_dataset = dataset_Clip(train_df, directory, processor_clip_1, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = dataset_Clip(test_df, directory, processor_clip_1, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

X2 = ExtractModelFeature(train_dataloader, model_clip_1)
X2_test = ExtractModelFeature(test_dataloader, model_clip_1)


print('train: ', X2.shape)
print('test: ', X2_test.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/568742063.py in <cell line: 0>()
----> 1 train_dataset = dataset_Clip(train_df, directory, processor_clip_1, test=False)
      2 train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
      3 
      4 test_dataset = dataset_Clip(test_df, directory, processor_clip_1, test=True)
      5 test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

NameError: name 'processor_clip_1' is not defined

## === cell 13
'''
train_dataset = dataset_Clip(train_df, directory, processor_clip_2, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

test_dataset = dataset_Clip(test_df, directory, processor_clip_2, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

X3 = ExtractModelFeature(train_dataloader, model_clip_2)
X3_test = ExtractModelFeature(test_dataloader, model_clip_2)

print('train: ', X3.shape)
print('test: ', X3_test.shape)

'''

## === cell 14

x_meta = train_df.iloc[:,1:13].values
x_test_meta = test_df.iloc[:,1:13].values


## === cell 15
'''
class ImageExtract:
    def __init__(self, df, directory, test=False):
        self.df = df
        self.directory = directory
        self.test = test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        image = np.array(Image.open(address).convert('L').resize((128,128))).flatten()
        return image
'''

## === cell 16
'''
train_dataset = ImageExtract(train_df, directory, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

test_dataset = ImageExtract(test_df, directory, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

pca = PCA(n_components=512)
X4, pca = ExtractModelFeature(train_dataloader, pca, Train_PCA=True)
X4_test = ExtractModelFeature(test_dataloader, pca)

print('train: ', X4.shape)
print('test: ', X4_test.shape)

'''

## === cell 17
'''
train_dataset = ImageExtract(train_df, directory, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

test_dataset = ImageExtract(test_df, directory, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

pca = PCA(n_components=512)
X4 = ExtractModelFeature(train_dataloader, 'ImageExtract')
X4_test = ExtractModelFeature(test_dataloader, 'ImageExtract')

print('train: ', X4.shape)
print('test: ', X4_test.shape)

'''

## === cell 18
'''
X = np.hstack((X1,X2, X3, X4, x_meta))
X_test = np.hstack((X1_test, X2_test, X3_test, X4_test, x_test_meta))

print('train_stacked: ', X.shape)
print('test_stacked: ', X_test.shape)
'''

## === cell 19

X = np.hstack((X2, x_meta))
X_test = np.hstack((X2_test, x_test_meta))

print('train_stacked: ', X.shape)
print('test_stacked: ', X_test.shape)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/744985439.py in <cell line: 0>()
----> 1 X = np.hstack((X2, x_meta))
      2 X_test = np.hstack((X2_test, x_test_meta))
      3 
      4 print('train_stacked: ', X.shape)
      5 print('test_stacked: ', X_test.shape)

NameError: name 'X2' is not defined

## === cell 20
y = train_df.Pawpularity.values
print(y)

## === cell 22

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
pt.fit(np.vstack((X, X_test)))
X = pt.transform(X)
X_test = pt.transform(X_test)
print(X.shape)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4014257592.py in <cell line: 0>()
      1 from sklearn.preprocessing import PowerTransformer
      2 pt = PowerTransformer()
----> 3 pt.fit(np.vstack((X, X_test)))
      4 X = pt.transform(X)
      5 X_test = pt.transform(X_test)

NameError: name 'X' is not defined

## === cell 23

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(np.vstack((X, X_test)))
X = scaler.transform(X)
X_test = scaler.transform(X_test)
print(X.shape)
print(X_test.shape)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3120152453.py in <cell line: 0>()
      1 from sklearn.preprocessing import StandardScaler
      2 scaler = StandardScaler()
----> 3 scaler.fit(np.vstack((X, X_test)))
      4 X = scaler.transform(X)
      5 X_test = scaler.transform(X_test)

NameError: name 'X' is not defined

## === cell 24
reg = SVR(C=10.0, kernel='rbf', degree=3, max_iter=400000, output_type='numpy')
reg.fit(X, y)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1055984139.py in <cell line: 0>()
----> 1 reg = SVR(C=10.0, kernel='rbf', degree=3, max_iter=400000, output_type='numpy')
      2 reg.fit(X, y)

NameError: name 'SVR' is not defined

## === cell 25

print("Predicted values:", reg.predict(X[:10, :])) 
print("True values: ", y[:10])


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1485669702.py in <cell line: 0>()
----> 1 print("Predicted values:", reg.predict(X[:10, :]))
      2 print("True values: ", y[:10])

NameError: name 'reg' is not defined

## === cell 26
def RSME(y, y_pred):
    rsme = np.sqrt(np.mean( (y-y_pred)**2) )
    return rsme
    
print(RSME(y, reg.predict(X)))

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2475922339.py in <cell line: 0>()
      3     return rsme
      4 
----> 5 print(RSME(y, reg.predict(X)))

NameError: name 'reg' is not defined

## === cell 27
'''
X_test = scaler.transform(X2_test)
print('X shape: ', X_test.shape)
'''

## === cell 28
y_pred = reg.predict(X_test)
print("Predicted values:", y_pred) 


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2045394328.py in <cell line: 0>()
----> 1 y_pred = reg.predict(X_test)
      2 print("Predicted values:", y_pred)

NameError: name 'reg' is not defined

## === cell 29
submit = pd.DataFrame()
submit['Id'] = test_df['Id']
submit['Pawpularity'] = y_pred

submit.to_csv('submission.csv', index=False)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4010081601.py in <cell line: 0>()
      2 submit = pd.DataFrame()
      3 submit['Id'] = test_df['Id']
----> 4 submit['Pawpularity'] = y_pred
      5 
      6 # Save the submission to a CSV file

NameError: name 'y_pred' is not defined

## === cell 30
submit
