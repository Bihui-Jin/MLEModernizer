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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.0991236022967664

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the unused `pretrainedmodels` import that caused a ModuleNotFoundError, fix the Albumentations API usage (replace `RandomResizedCrop` and `CenterCrop` with the current signature), and correct the dataset class to inherit from `torch.utils.data.Dataset` and return the label as a tensor. These changes eliminate the runtime errors, allow the transforms to be built, and let the training‑validation loop run, ultimately producing a valid `submission.csv` file.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/1942864320.py", line 1
    I will remove the unused `pretrainedmodels` import that caused a ModuleNotFoundError, fix the Albumentations API usage (replace `RandomResizedCrop` and `CenterCrop` with the current signature), and correct the dataset class to inherit from `torch.utils.data.Dataset` and return the label as a tensor. These changes eliminate the runtime errors, allow the transforms to be built, and let the training‑validation loop run, ultimately producing a valid `submission.csv` file.
                            ^
SyntaxError: invalid non-printable character U+202F


## === cell 1
!nvidia-smi



## === cell 2
import os

!pip install ../input/timmmodels/dist/timm-0.3.4.tar
!cp ../input/timm-pretrained-efficientnet/efficientnet/tf_efficientnet_b4_ns-d6313a46.pth /root/.cache/torch/hub/checkpoints/



## === cell 3
import cv2
import copy
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

from tqdm.notebook import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm



## === cell 4
ROOT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"



## === cell 5
class CFG:
    model_name = 'tf_efficientnet_b4_ns'
    img_size = 512
    scheduler = 'CosineAnnealingWarmRestarts'
    T_max = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    seed = 42
    num_classes = 5
    num_epochs = 10
    n_fold = 5
    smoothing = 0.2
    t1 = 0.8
    t2 = 1.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    savemodel = True
    savemodelpath = '/kaggle/working/bitemp-21.pth'
    dosubmission = True
    crop_size = 256
    crop_p = 0.9



## === cell 6
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CFG.seed)



## === cell 7
df = pd.read_csv(f"{ROOT_DIR}/train.csv")



## === cell 8
skf = StratifiedKFold(n_splits=CFG.n_fold)
for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_ , "kfold"] = int(fold)
    
df['kfold'] = df['kfold'].astype(int)



## === cell 9
class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name = self.df.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.df.iloc[index, 1])
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return img, torch.tensor(label, dtype=torch.long)



## === cell 10
data_transforms = {
    "train": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.CenterCrop(CFG.crop_size, CFG.crop_size, p=CFG.crop_p),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        ToTensorV2()], p=1.0),
    
    "valid": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.CenterCrop(CFG.crop_size, CFG.crop_size, p=CFG.crop_p),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.0)
}



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3117575737.py in <cell line: 0>()
     25             ),
     26         A.CoarseDropout(p=0.5),
---> 27         A.Cutout(p=0.5),
     28         ToTensorV2()], p=1.0),
     29 

AttributeError: module 'albumentations' has no attribute 'Cutout'

## === cell 11
def log_t(u, t):
    """Compute log_t for `u'."""
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(
                exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                logt_partition.pow(1.0-t)
    logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu
    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    effective_dim = \
        torch.sum(
                (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)
    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)
    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
                exp_t(normalized_activations - logt_partition, t),
                dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
                lower * update + (1.0-update) * logt_partition,
                shape_partition)
        upper = torch.reshape(
                upper * (1.0 - update) + update * logt_partition,
                shape_partition)
    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants 
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_softmax(activations, t, num_iters=5):
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_logistic_loss(activations,
        labels,
        t1,
        t2,
        label_smoothing=0.0,
        num_iters=5,
        reduction = 'mean'):
    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels
    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)
    probabilities = tempered_softmax(activations, t2, num_iters)
    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1)
    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    return loss_values.mean()



## === cell 12
def train_model(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)
    scaler = amp.GradScaler()

    for epoch in range(1,num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train','valid']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)
                        loss = bi_tempered_logistic_loss(outputs, labels, t1=CFG.t1, t2=CFG.t2, label_smoothing=CFG.smoothing)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels).double().item()

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_sizes[phase]
            history[f'{phase} loss'].append(epoch_loss)
            history[f'{phase} acc'].append(epoch_acc)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase=='valid' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Fold{fold}_{best_acc}_epoch{epoch}.bin"

        print()
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        (time.time()-start)//3600, ((time.time()-start)%3600)//60, (time.time()-start)%60))
    print("Best Accuracy ",best_acc)
    model.load_state_dict(best_model_wts)
    return model, history



## === cell 13
def run_fold(model, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]
    
    train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
    valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"])
    
    dataset_sizes = {'train': len(train_data), 'valid': len(valid_data)}
    
    train_loader = DataLoader(dataset=train_data, batch_size=CFG.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=CFG.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    dataloaders = {'train': train_loader, 'valid': valid_loader}
    model, history = train_model(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    return model, history



## === cell 14
model = timm.create_model(CFG.model_name, pretrained=True)



## === cell 15
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, CFG.num_classes)
model.to(CFG.device)



## === cell 16
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)



## === cell 17
def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    if CFG.scheduler == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr)
    return None



## === cell 18
scheduler = fetch_scheduler(optimizer)



## === cell 19
model, history = run_fold(model, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1147226466.py in <cell line: 0>()
----> 1 model, history = run_fold(model, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs)
      2 

/tmp/ipykernel_55/4167243089.py in run_fold(model, optimizer, scheduler, device, fold, num_epochs)
      3     train_df = df[df.kfold != fold]
      4 
----> 5     train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
      6     valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"])
      7 

NameError: name 'data_transforms' is not defined

## === cell 20
if CFG.savemodel:
    torch.save(model, CFG.savemodelpath)



## === cell 21
if CFG.dosubmission:
    t_df = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
    test_data = CassavaLeafDataset(TEST_DIR, t_df, transforms=data_transforms["valid"])
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    submit_df = t_df.copy(deep=True)
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(CFG.device)
            outputs = model(inputs)
            pred_label = torch.argmax(outputs, dim=1).cpu().item()
            submit_df.at[i, 'label'] = pred_label
    submit_df.to_csv("/kaggle/working/submission.csv", index=False)
```

## --- ERROR in cell 21, traceback:
  File "/tmp/ipykernel_55/2213874823.py", line 14
    ```
    ^
SyntaxError: invalid syntax
