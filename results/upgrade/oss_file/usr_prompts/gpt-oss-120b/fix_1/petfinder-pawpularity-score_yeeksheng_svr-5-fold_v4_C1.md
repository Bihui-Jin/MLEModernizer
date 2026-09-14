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

3.13

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
geopandas==0.14.4
joblib==1.5.2
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

23.26740549171348

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore')

import pickle
from tqdm.auto import tqdm
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import random
import gc

from PIL import Image

import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torch
import timm
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import glob
import joblib

from cuml.svm import SVR as cumlSVR

gc.enable()

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/2066922464.py in <cell line: 0>()
     27 import joblib
     28 
---> 29 from cuml.svm import SVR as cumlSVR
     30 
     31 gc.enable()

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
models = {
    'beit_large_patch16_512': {
        'model_path': '/kaggle/input/beit_large__512_5fold/transformers/default/1',
        'im_size': 512,
        'train_emb': [],
        'test_emb': []
    },
    'deit_base_distilled_patch16_384': {
        'model_path': '/kaggle/input/deit-base-5-folds/pytorch/default/1',
        'im_size': 384,
        'train_emb': [],
        'test_emb': []
    },
    'maxvit_xlarge_tf_512': {
        'model_path': '/kaggle/input/maxvit_5folds/transformers/default/1',
        'im_size':512,
        'train_emb': [],
        'test_emb': []
    },
    'tf_efficientnet_l2': {
        'model_path': '/kaggle/input/tf_efficientnet-5-folds/pytorch/default/1',
        'im_size': 475,
        'train_emb': [],
        'test_emb': []
    }
}


class Config:
    data_dir = "/kaggle/input/petfinder-pawpularity-score"
    embedding_dir = "/kaggle/input/training-embeddings/pytorch/default/1"
    svr_dir = "/kaggle/input/svr-4-models-5-folds/pytorch/default/1"
    random_seed = 555
    tta_times = 1 # 1: no TTA
    tta_beta = 1 / tta_times
    pretrained = False
    inp_channels = 3
    batch_size = 1
    num_workers = 0 # >0: OS Error
    out_features = 1
    dropout = 0.1
    scheduler_name = "OneCycleLR"

## === cell 2
def seed_everything(seed=Config.random_seed):
    os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark =True

seed_everything()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

## === cell 3
train = pd.read_csv(f'{Config.data_dir}/train.csv')
test = pd.read_csv(f'{Config.data_dir}/test.csv')

train['path'] = train['Id'].map(lambda x: f'{Config.data_dir}/train/{x}.jpg')
test['path'] = test['Id'].map(lambda x: f'{Config.data_dir}/test/{x}.jpg')


print(train.shape, test.shape)

## === cell 4
def get_train_transforms(dim):
    return A.Compose(
        [             
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p = 0.5),
            A.HorizontalFlip(p = 0.5)
        ]
  )

def get_inference_fixed_transforms(dim):
    return A.Compose([
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.CenterCrop(height=dim, width=dim, p=1.0),
        ], p=1.0)


## === cell 5
class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets=None, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        image = image / 255  # Convert to 0-1
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return image, target
        else:
            return image


## === cell 6
class PetNet(nn.Module):
    def __init__(
        self,
        model_name,
        out_features = Config.out_features,
        inp_channels=Config.inp_channels,
        pretrained=Config.pretrained
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3, num_classes=0)
        self.fc1 = nn.Linear(self.model.num_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)
    
    def get_image_embedding(self, image):
        return self.model(image)

    def forward(self, image):
        output = self.model(image)
        x= self.fc1(output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

## === cell 7
def extract_embeddings(model_arch, model_path, im_size, batch_size, dataset):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)

    all_embeddings = None

    with torch.no_grad():
        print(f'Extract Embeddings for {model_path}')
        for model_file in glob.glob(f'{model_path}/*.pth'):
            current_embeddings = []

            model = PetNet(model_name=model_arch)
            model.load_state_dict(torch.load(model_file))
            model = model.to(device)
            model.eval()

            for batch in tqdm(dataloader):
                images = batch  # Targets are not used for embedding extraction
                images = images.to('cuda')
                outputs = model.model(images)
                current_embeddings.append(outputs.cpu().numpy())


            current_embeddings = np.concatenate(current_embeddings, axis=0)

            if all_embeddings is None:
                all_embeddings = current_embeddings
            else:
                all_embeddings = np.concatenate((all_embeddings, current_embeddings), axis=1)

            del model
            torch.cuda.empty_cache()
            gc.collect()

    return all_embeddings


## === cell 8
for model_name, model_info in models.items():

    train_emb_path = f'{Config.embedding_dir}/{model_name}_train_embeddings.pkl'
    train_embeddings = joblib.load(train_emb_path)
    models[model_name]['train_emb'] = train_embeddings
    

    
    model_path = model_info['model_path']
    im_size = model_info['im_size']

    test_dataset = PetDataset(
        image_filepaths=test['path'].values,
        targets=None,
        transform=get_inference_fixed_transforms(im_size)
    )
    
    test_embeddings = extract_embeddings(model_name, model_path, im_size,Config.batch_size, test_dataset)

    models[model_name]['test_emb'] = test_embeddings

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3379708923.py in <cell line: 0>()
      2 
      3     train_emb_path = f'{Config.embedding_dir}/{model_name}_train_embeddings.pkl'
----> 4     train_embeddings = joblib.load(train_emb_path)
      5     models[model_name]['train_emb'] = train_embeddings
      6 

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/training-embeddings/pytorch/default/1/beit_large_patch16_512_train_embeddings.pkl'

## === cell 9
TRAIN = np.concatenate([models[name]['train_emb'] for name in models.keys()], axis=1)
TEST = np.concatenate([models[name]['test_emb'] for name in models.keys()], axis=1)
targets_train = train['Pawpularity']

print(TRAIN.shape,TEST.shape)
print(len(targets_train))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1631458201.py in <cell line: 0>()
----> 1 TRAIN = np.concatenate([models[name]['train_emb'] for name in models.keys()], axis=1)
      2 TEST = np.concatenate([models[name]['test_emb'] for name in models.keys()], axis=1)
      3 targets_train = train['Pawpularity']
      4 
      5 print(TRAIN.shape,TEST.shape)

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 10
def fit_gpu_svr(TRAIN, TEST, train_targets, n_splits=5):
    ypredtrain_ = np.zeros(TRAIN.shape[0])
    ypredtest_ = np.zeros(TEST.shape[0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(tqdm(kf.split(TRAIN))):
        X_train, X_valid = TRAIN[train_index], TRAIN[valid_index]
        y_train, y_valid = train_targets[train_index], train_targets[valid_index]

        model = cumlSVR(C=16.0, kernel='rbf', degree=3, max_iter=4000)

        model.fit(X_train, y_train.clip(1, 85))

        ypredtrain_[valid_index] = np.clip(model.predict(X_valid), 1, 100)
        ypredtest_ += np.clip(model.predict(TEST), 1, 100)
        

        del model
        gc.collect()

    ypredtest_ /= n_splits

    return ypredtrain_, ypredtest_



def svr_predict(TEST, n_splits=5):
    ypredtest_ = np.zeros(TEST.shape[0])

    for index in range(n_splits):

        model = joblib.load(f'{Config.svr_dir}/svr_model_fold_{index}.joblib')

        ypredtest_ += np.clip(model.predict(TEST), 1, 100)

        del model
        gc.collect()

    ypredtest_ /= n_splits

    return ypredtest_


## === cell 11
ypred_train, ypred_test = fit_gpu_svr(TRAIN, TEST, targets_train)
ypred_test

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518390324.py in <cell line: 0>()
----> 1 ypred_train, ypred_test = fit_gpu_svr(TRAIN, TEST, targets_train)
      2 # ypred_test = svr_predict(TEST)
      3 ypred_test

NameError: name 'TRAIN' is not defined

## === cell 13
test['Pawpularity'] = np.array(ypred_test)

output = test[['Id','Pawpularity']]

output

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/494980609.py in <cell line: 0>()
----> 1 test['Pawpularity'] = np.array(ypred_test)
      2 
      3 output = test[['Id','Pawpularity']]
      4 
      5 output

NameError: name 'ypred_test' is not defined

## === cell 14
output.to_csv('submission.csv', index=None)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518803861.py in <cell line: 0>()
----> 1 output.to_csv('submission.csv', index=None)

NameError: name 'output' is not defined
