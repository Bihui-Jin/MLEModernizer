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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.6894357242271227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install --upgrade pip
!pip install -qU timm albumentations

## === cell 2
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tqdm.notebook import tqdm
import timm
import glob

import warnings
warnings.filterwarnings('ignore')

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2426939683.py in <cell line: 0>()
      8 from torch.utils.data import Dataset, DataLoader
      9 import cv2
---> 10 import albumentations as A
     11 from albumentations.pytorch import ToTensorV2
     12 from sklearn.model_selection import StratifiedKFold

/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py in <module>
     16 from albumentations.check_version import check_for_updates
     17 
---> 18 from .augmentations import *
     19 from .core.composition import *
     20 from .core.serialization import *

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/__init__.py in <module>
     17 from .other.lambda_transform import *
     18 from .other.type_transform import *
---> 19 from .pixel.transforms import *
     20 from .spectrogram.transform import *
     21 from .text.transforms import *

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/pixel/transforms.py in <module>
     37     model_validator,
     38 )
---> 39 from scipy import special
     40 from typing_extensions import Literal, Self
     41 

/usr/lib/python3.11/importlib/_bootstrap.py in _handle_fromlist(module, fromlist, import_, recursive)

/usr/local/lib/python3.11/dist-packages/scipy/__init__.py in __getattr__(name)
    132 def __getattr__(name):
    133     if name in submodules:
--> 134         return _importlib.import_module(f'scipy.{name}')
    135     else:
    136         try:

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 3
class CFG:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = '/kaggle/input/alaska2-image-steganalysis/'
    subset_size = 75000
    image_size = 512
    
    model_name = 'tf_efficientnet_b2_ns' 
    num_classes = 1
    
    n_folds = 5
    fold_to_train = 0
    epochs = 1 # Can set a higher number of epochs for long training
    train_batch_size = 16 # MUST be reduced to fit B5 in 16GB VRAM. Start with 4.
    valid_batch_size = 32 # Also reduce validation batch size.
    weight_decay = 1e-6
    
    lr = 1e-4
    T_0 = 5 # Cosine Annealing: Number of epochs for the first restart.
    eta_min = 1e-6
    
    checkpoint_save_path = '/kaggle/working/latest_checkpoint.pth'
    best_model_save_path = f'/kaggle/working/best_model_fold_{fold_to_train}.pth'
    
    checkpoint_load_path = None # '/kaggle/input/alaska2-steg-resume/pytorch/default/1/latest_checkpoint.pth'


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)

## === cell 4
def alaska_weighted_auc(y_true, y_pred):
    """
    Calculates the weighted AUC score for the ALASKA2 competition.
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2: return 0.5
    
    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i+1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            mask_indices = np.where(mask)[0]
            start_idx, end_idx = mask_indices[0], mask_indices[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx:end_idx+1], [upper]])
            fpr_slice = np.concatenate([[np.interp(lower, tpr, fpr)], fpr[start_idx:end_idx+1], [np.interp(upper, tpr, fpr)]])
            tpr_slice, unique_indices = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[unique_indices]
            areas[i] = np.trapz(fpr_slice, tpr_slice)
            
    return np.sum(areas * weights) / np.sum(weights)

## === cell 5
image_folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
subset_files = []

print(f"Creating a subset of {CFG.subset_size} images from each folder...")
for folder in image_folders:
    folder_path = os.path.join(CFG.data_path, folder)
    files_in_folder = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    random.shuffle(files_in_folder) # Shuffle to get a random subset
    subset_files.extend(files_in_folder[:CFG.subset_size])
    print(f"  - Took {len(files_in_folder[:CFG.subset_size])} images from {folder}")

df = pd.DataFrame({'image_path': subset_files})
df['label'] = df['image_path'].apply(lambda x: 0 if 'Cover' in x else 1)
df['image_id'] = df['image_path'].apply(os.path.basename)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = fold

print("\nSubset dataset distribution:")
print(df['label'].value_counts())
print("\nFold distribution:")
print(df.groupby('fold')['label'].value_counts())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2232272557.py in <cell line: 0>()
     11     print(f"  - Took {len(files_in_folder[:CFG.subset_size])} images from {folder}")
     12 
---> 13 df = pd.DataFrame({'image_path': subset_files})
     14 df['label'] = df['image_path'].apply(lambda x: 0 if 'Cover' in x else 1)
     15 df['image_id'] = df['image_path'].apply(os.path.basename)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    478     else:
    479         keys = list(data.keys())
--> 480         columns = Index(keys) if keys else default_index(0)
    481         arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    482 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in __new__(cls, data, dtype, copy, name, tupleize_cols)
    563 
    564         try:
--> 565             arr = sanitize_array(data, None, dtype=dtype, copy=copy)
    566         except ValueError as err:
    567             if "index must be specified when data is not list-like" in str(err):

/usr/local/lib/python3.11/dist-packages/pandas/core/construction.py in sanitize_array(data, index, dtype, copy, allow_2d)
    652 
    653         else:
--> 654             subarr = maybe_convert_platform(data)
    655             if subarr.dtype == object:
    656                 subarr = cast(np.ndarray, subarr)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/cast.py in maybe_convert_platform(values)
    136     if arr.dtype == _dtype_obj:
    137         arr = cast(np.ndarray, arr)
--> 138         arr = lib.maybe_convert_objects(arr)
    139 
    140     return arr

lib.pyx in pandas._libs.lib.maybe_convert_objects()

TypeError: Cannot convert numpy.ndarray to numpy.ndarray

## === cell 6
def get_transforms(data_type='train'):
    if data_type == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=CFG.image_size, width=CFG.image_size, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=CFG.image_size, width=CFG.image_size, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class AlaskaDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms: image = self.transforms(image=image)['image']
        return image, label

## === cell 7
def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    return running_loss / len(loader)

def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    val_loss = running_loss / len(loader)
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score

## === cell 8
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

## === cell 9
def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")
    
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    train_dataset = AlaskaDataset(train_df, transforms=get_transforms('train'))
    valid_dataset = AlaskaDataset(valid_df, transforms=get_transforms('valid'))
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=CFG.num_classes).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0 * len(train_loader), eta_min=CFG.eta_min)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_score = 0.0
    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming training from checkpoint: {CFG.checkpoint_load_path}")
        checkpoint = torch.load(CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = 10 # checkpoint['epoch'] + 1 # Start from the next epoch
        best_score = checkpoint['best_score']
        print(f"Loaded model from epoch {start_epoch-1} with best score: {best_score:.4f}")
    else:
        print("No checkpoint found, starting training from scratch.")

    for epoch in range(start_epoch, CFG.epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
        train_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, CFG.device)
        val_loss, val_score = eval_fn(valid_loader, model, criterion, CFG.device)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Weighted AUC: {val_score:.4f}")

        if val_score > best_score:
            print(f"Validation score improved! ({best_score:.4f} -> {val_score:.4f}). Saving best model...")
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_score': best_score,
        }
        torch.save(checkpoint, CFG.checkpoint_save_path)
        print(f"Epoch {epoch+1} state saved to checkpoint: {CFG.checkpoint_save_path}")

    print(f"\n========== Finished Training for Fold {fold}. Best Score: {best_score:.4f} ==========")

run_training(CFG.fold_to_train)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/899071029.py in <cell line: 0>()
     60 
     61 # Start the training process
---> 62 run_training(CFG.fold_to_train)

/tmp/ipykernel_11/899071029.py in run_training(fold)
      4 
      5     # --- Data Setup ---
----> 6     train_df = df[df['fold'] != fold].reset_index(drop=True)
      7     valid_df = df[df['fold'] == fold].reset_index(drop=True)
      8     train_dataset = AlaskaDataset(train_df, transforms=get_transforms('train'))

NameError: name 'df' is not defined

## === cell 10
def create_submission():
    print("\nStarting inference on the test set...")
    
    test_folder = os.path.join(CFG.data_path, 'Test')
    test_image_ids = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    
    test_df = pd.DataFrame({'image_id': test_image_ids})
    test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(test_folder, x))
    test_df['label'] = 0 # Dummy label
    
    test_dataset = AlaskaDataset(test_df, transforms=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=2)
    
    model = timm.create_model(CFG.model_name, pretrained=False, num_classes=CFG.num_classes)
    model.load_state_dict(torch.load(CFG.best_model_save_path))
    model.to(CFG.device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting")
        for images, _ in pbar:
            images = images.to(CFG.device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
            
    submission_df = pd.DataFrame({'Id': test_image_ids, 'Label': predictions}).sort_values('Id')
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nSubmission file created successfully!")
    print(submission_df)

create_submission()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1647485009.py in <cell line: 0>()
     33     print(submission_df)
     34 
---> 35 create_submission()

/tmp/ipykernel_11/1647485009.py in create_submission()
      6     test_image_ids = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
      7 
----> 8     test_df = pd.DataFrame({'image_id': test_image_ids})
      9     test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(test_folder, x))
     10     test_df['label'] = 0 # Dummy label

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    478     else:
    479         keys = list(data.keys())
--> 480         columns = Index(keys) if keys else default_index(0)
    481         arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    482 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in __new__(cls, data, dtype, copy, name, tupleize_cols)
    563 
    564         try:
--> 565             arr = sanitize_array(data, None, dtype=dtype, copy=copy)
    566         except ValueError as err:
    567             if "index must be specified when data is not list-like" in str(err):

/usr/local/lib/python3.11/dist-packages/pandas/core/construction.py in sanitize_array(data, index, dtype, copy, allow_2d)
    652 
    653         else:
--> 654             subarr = maybe_convert_platform(data)
    655             if subarr.dtype == object:
    656                 subarr = cast(np.ndarray, subarr)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/cast.py in maybe_convert_platform(values)
    136     if arr.dtype == _dtype_obj:
    137         arr = cast(np.ndarray, arr)
--> 138         arr = lib.maybe_convert_objects(arr)
    139 
    140     return arr

lib.pyx in pandas._libs.lib.maybe_convert_objects()

TypeError: Cannot convert numpy.ndarray to numpy.ndarray
