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

3.10

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
model-signing==1.1.1
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
sigstore-models==0.0.5
sklearn-pandas==2.2.0
statsmodels==0.14.5
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

17.30894069590541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
sys.path.append('../input/saved-weights/ConvNeXt/ConvNeXt')
sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")

## === cell 1
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from torch import nn
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations.pytorch
import torchvision
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastai.vision.all import *
from fastai.data.core import *
import gc
import random
import cuml, pickle
from cuml.svm import SVR
import models.convnext as cn_next

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/2792035412.py in <cell line: 0>()
     17 import gc
     18 import random
---> 19 import cuml, pickle
     20 from cuml.svm import SVR
     21 import models.convnext as cn_next

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

## === cell 3
def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))

## === cell 4
base_dir = '/kaggle/input'
model_weights = os.path.join(base_dir, 'saved-weights')
test_folder = os.path.join(base_dir, 'petfinder-pawpularity-score', 'test')
test_file = os.path.join(base_dir, 'petfinder-pawpularity-score', 'test.csv')

## === cell 5
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5
embed_dim = 128
num_of_hidden = 2
batch_size = 64
hidden_dimension = [256, 64]
input_shape_224 = (224, 224, 3)
input_shape_384 = (384, 384, 3)
max_size_384 = 480
max_size_224 = 384
models_list = {'convnext_base': 128,
               'conv_next': 64,
              }
save_name = '/kaggle/working/'

weights_svr = {'beit_base_patch16_224_in22k': [0.5,0.5], 
               'swin_large_patch4_window12_384_in22k': [0.1, 0.9],
               'swin_base_patch4_window7_224_in22k': [0.5,0.5],
               'swin_large_patch4_window7_224_in22k': [0.4, 0.6],
               'conv_next': [0.2, 0.8],
               'convnext_base': [0.2, 0.8]}

model_weights = {'beit_base_patch16_224_in22k': 0.0, 
               'swin_large_patch4_window12_384_in22k': 0.0,
               'swin_base_patch4_window7_224_in22k': 0.0,
               'swin_large_patch4_window7_224_in22k': 0.0,
               'convnext_base': 0.3,
               'conv_next': 0.7,}

## === cell 7
test_csv = pd.read_csv(test_file)
test_csv.head()

## === cell 8
test_csv['path_img'] = list(map(lambda x: os.path.join(test_folder, x+'.jpg'), test_csv['Id']))
test_csv['Pawpularity'] = [1]*len(test_csv)

## === cell 9
class PetsDataset(Dataset):
    
    def __init__(self, df, transform = None):
        self.transform = transform
        self.df = df

        self.cat = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
       'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df['path_img'].iloc[idx]
        label_1 = self.df['Pawpularity'].iloc[idx]
        img = Image.open(img_path)
        if self.transform:
            album = self.transform(image = np.array(img))
            img = album['image']


        df_data = self.df[self.cat].iloc[idx].values
        
        return (img, df_data, label_1)

## === cell 10
def get_data(batch_size, max_size, input_shape):
    
    test_transform = A.Compose([A.LongestMaxSize(max_size=max_size, interpolation=1),
                             A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1], border_mode=0, 
                                           value=(0,0,0)),
                             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                             A.transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, 
                                                      hue = 0.1, p =0.6),
                             A.CenterCrop(height = input_shape[0], width = input_shape[1]),
                             A.HorizontalFlip(p=0.6),
                             A.Normalize(mean, std_dev),
                             ToTensorV2(),
                             ])
    test_dataset = PetsDataset(test_csv, test_transform)
    testloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 4, shuffle=False)
    
    dls = DataLoaders.from_dsets(test_dataset, bs = batch_size)
    
    return dls, testloader


## === cell 12
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Network(nn.Module):
    def __init__(self, base, number_of_hidden, hidden, regression_out, output_categories, freeze_layer):
        super(Network, self).__init__()
        if (number_of_hidden != len(hidden)):
            raise "Number of Hidden layer and length of hidden dim must be same"
            
        
        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim+12)
        
        base.head = nn.Linear(base.head.in_features, embed_dim)
        nn.init.kaiming_normal_(base.head.weight)
        nn.init.constant_(base.head.bias, 0)
        
        self.p = 0.5

        self.regression = self.__fully_connected(number_of_hidden, hidden_dim, regression_out)
        
        self.network = self.__freeze_layer(base, freeze_layer)

        self.__initialise_weights()

    def __initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
              
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def __freeze_layer(self, base, freeze_layer):
        cnt = 0
        for child in base.children():
            cnt+=1
            if cnt > freeze_layer:
                break

            for param in child.parameters():
                param.requires_grad = False
        return base


    def __fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        layers.append(nn.Dropout(self.p))
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i+1]))
            if i != (number_of_hidden-1):
                layers.append(nn.Dropout(self.p))

        layers.append(nn.Linear(hidden_dim[-1], output_categories))
    
        return nn.Sequential(*layers)
        
    def forward(self,x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg,x

## === cell 13
def get_learner(model_name, batch_size, loss, metric, max_size, input_size, conv_next, save_path):
    dls, testloader = get_data(batch_size,max_size, input_size)
    dls = dls.to(device)
    if conv_next:
        if model_name == 'convnext_base':
            network = cn_next.convnext_base(pretrained = False)
            model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
        else:
            network = cn_next.convnext_large(pretrained = False)
            model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    else:
        network = timm.create_model(model_name, pretrained = False)
        model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
        
    model = model.to(device)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, 
                    model_dir = save_path).to_fp16()
    return learn, testloader

## === cell 14
def tta(testloader, learn, svr, svr_weight, tta_steps = 4):
    tta_outputs = []
    tta_steps = 4
    learn.model.eval()
    for i in range(tta_steps):
        final_outputs = []
        svr_data = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                output_list = []
                images, tabular = images.to(device), tabular.to(device)
                reg_output, embed = learn.model(images, tabular)
                reg_output = 100*torch.sigmoid(reg_output)
                output = reg_output.detach().to('cpu').numpy().reshape(-1,).tolist()
                final_outputs.extend(output)
                svr_data.extend(embed.cpu().numpy())
                
            final_outputs = np.array(final_outputs)
            svr_data = np.array(svr_data)
            svr_preds = clf.predict(svr_data)
            final_outputs = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs

        tta_outputs.append(final_outputs)
    tta_outputs_arr = np.mean(np.array(tta_outputs), axis = 0)
    return tta_outputs_arr

## === cell 15
final_predictions = []
for model_name, bs in models_list.items():
    m_name = model_name[:model_name.find('_', 5)]
    
    fold_tta = []
    svr_weights = weights_svr[model_name]
    w = model_weights[model_name]
    m_384 = False
    conv_next = False
    if model_name.find('384')!=-1:
        m_384 = True
        m_name = m_name+'_384'
        
    if model_name.find('conv') != -1:
        conv_next = True
        m_name = model_name
        m_384 = True

    print("#################################")
    print("Testing Model: {}".format(m_name))
    print("#################################")
    for fold in range(N_FOLDS):
        print("Fold: {}".format(fold))
        saved_name = m_name+"_fold_{}_full.pth".format(fold)
        if conv_next:
            svr_name = m_name+"_svr_fold_{}_full.pkl".format(fold)
        else:
            svr_name = m_name+"_svr_fold_{}_full.pkl".format(fold)
        if m_384:
            learn, testloader = get_learner(model_name, bs, BCEWithLogitsLossFlat(), 
                            petfinder_rmse, max_size_384, input_shape_384, conv_next, save_name) 
            
        else:
            learn, testloader = get_learner(model_name, bs, BCEWithLogitsLossFlat(), 
                            petfinder_rmse, max_size_224, input_shape_224, conv_next, save_name) 
            
        learn.load(os.path.join('/kaggle/input/saved-weights', saved_name))
        clf = pickle.load(open(os.path.join('/kaggle/input/saved-weights', svr_name), "rb"))
        
        fold_tta.append(tta(testloader, learn, clf,svr_weights))
        
        del learn
        torch.cuda.empty_cache()
        gc.collect()
        
    fold_tta = np.array(fold_tta)
    fold_tta = np.mean(np.array(fold_tta), axis = 0)
    final_predictions.append(w*fold_tta)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3661616316.py in <cell line: 0>()
     28             svr_name = m_name+"_svr_fold_{}_full.pkl".format(fold)
     29         if m_384:
---> 30             learn, testloader = get_learner(model_name, bs, BCEWithLogitsLossFlat(), 
     31                             petfinder_rmse, max_size_384, input_shape_384, conv_next, save_name) 
     32 

/tmp/ipykernel_11/3494549799.py in get_learner(model_name, batch_size, loss, metric, max_size, input_size, conv_next, save_path)
      1 def get_learner(model_name, batch_size, loss, metric, max_size, input_size, conv_next, save_path):
----> 2     dls, testloader = get_data(batch_size,max_size, input_size)
      3     dls = dls.to(device)
      4     if conv_next:
      5         if model_name == 'convnext_base':

/tmp/ipykernel_11/1217904063.py in get_data(batch_size, max_size, input_shape)
      1 def get_data(batch_size, max_size, input_shape):
      2 
----> 3     test_transform = A.Compose([A.LongestMaxSize(max_size=max_size, interpolation=1),
      4                              A.PadIfNeeded(min_height=input_shape[0], min_width=input_shape[1], border_mode=0, 
      5                                            value=(0,0,0)),

AttributeError: 'functools.partial' object has no attribute 'Compose'

## === cell 16
final_predictions = np.array(final_predictions)

## === cell 17
final_predictions = np.sum(final_predictions, axis = 0)

## === cell 18
test_csv['Pawpularity'] = final_predictions

## === cell 19
test_csv = test_csv[["Id", "Pawpularity"]]

## === cell 20
test_csv.head()

## === cell 21
test_csv.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
