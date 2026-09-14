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

0.8370631752274111

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import gc
import os
import cv2
from time import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42

## === cell 3
def breaker():
    print("\n" + 50*"-" + "\n")

def head(x, no_of_ele=5):
    print(x[:no_of_ele])
    
def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        try:
            image = cv2.imread(file_path + name + ".jpg", cv2.IMREAD_GRAYSCALE).astype("float64")
        except AttributeError:
            print(file_path + name)
        if size:
            image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4)
        cv2.normalize(src=image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        images.append(image.reshape(1, size, size))
    return np.array(images)

## === cell 6
start_time = time()

ss = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/sample_submission.csv")

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages("../input/ranzcr-clip-catheter-line-classification/test/", 
                      ts_img_names, 
                      size=144)

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time)/60))
breaker()

## === cell 8
class Dataset(Dataset):
    def __init__(this, X=None, y=None, mode="train"):
        this.mode = mode
        this.X = X
        if mode == "train":
            this.y = y
            
    def __len__(this):
        return this.X.shape[0]
    
    def __getitem__(this, idx):
        if this.mode == "train":
            return torch.FloatTensor(this.X[idx]), torch.FloatTensor(this.y[idx])
        else:
            return torch.FloatTensor(this.X[idx])

## === cell 11
class CFG():
    tr_batch_size = 128 # Alos va_batch_size
    ts_batch_size = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_channels = 1
    OL = 11
    
    def __init__(this, filter_sizes=[64, 128, 256, 512], HL=[4096, 4096], epochs=50, n_folds=5):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds

## === cell 13
class CNN(nn.Module):
    def __init__(this, in_channels=1, filter_sizes=None, HL=None, OL=None, use_DP=False, DP1=0.2, DP2=0.5):
        super(CNN, this).__init__()
        
        this.use_DP = use_DP
        
        this.DP1 = nn.Dropout(p=0.2)
        this.DP2 = nn.Dropout(p=0.5)
        
        this.MP_ = nn.MaxPool2d(kernel_size=2)
        
        this.CN1 = nn.Conv2d(in_channels=in_channels, out_channels=filter_sizes[0], kernel_size=3, stride=1, padding=1)
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)
        
        this.CN2 = nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=3, stride=1, padding=1)
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)
        
        this.CN3 = nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=1, padding=1)
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)
    
        this.CN4 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=1, padding=1)
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        
        this.CN5 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1, padding=1)
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        
        this.CN6 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1, padding=1)
        this.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        
        this.FC1 = nn.Linear(in_features=filter_sizes[3]*2*2, out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        this.FC3 = nn.Linear(in_features=HL[1], out_features=OL)
        
    def getOptimizer(this, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(this.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getStepLR(this, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
        
    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)
    
    def forward(this, x):
        if not this.use_DP:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            
            x = x.view(x.shape[0], -1)
            
            x = F.relu(this.FC1(x))
            x = F.relu(this.FC2(x))
            x = this.FC3(x)
            
            return x
        else:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            
            x = x.view(x.shape[0], -1)
            
            x = F.relu(this.DP2(this.FC1(x)))
            x = F.relu(this.DP2(this.FC2(x)))
            x = this.FC3(x)
            
            return x

## === cell 15
def predict_(model=None, dataloader=None, device=None, path=None):
    if path:
        model.load_state_dict(torch.load(path))
    else:
        pass
    model.to(device)
    model.eval()

    y_pred = torch.zeros(1, 11).to(device)

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)
    
    return y_pred[1:].detach().cpu().numpy()

## === cell 17
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[4096, 4096], epochs=50, n_folds=5)

ts_data_setup = Dataset(ts_images, None, "test")
ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

model = CNN(in_channels=cfg.in_channels, filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL)

y_pred_e19 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_19.pt")
y_pred_e20 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_20.pt")
y_pred_e23 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_23.pt")
y_pred_e25 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_25.pt")

y_pred = (y_pred_e19 + y_pred_e20 + y_pred_e23 + y_pred_e25)/4
y_pred = np.clip(y_pred, 1e-15, 1-1e-15)

ss.iloc[:, 1:] = y_pred
ss.to_csv("./submission.csv", index=False)
ss.head(5)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1684126811.py in <cell line: 0>()
      6 model = CNN(in_channels=cfg.in_channels, filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL)
      7 
----> 8 y_pred_e19 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_19.pt")
      9 y_pred_e20 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_20.pt")
     10 y_pred_e23 = predict_(model=model, dataloader=ts_data, device=cfg.device, path="../input/rccl-1x144-f-train/Epoch_23.pt")

/tmp/ipykernel_11/2641274988.py in predict_(model, dataloader, device, path)
      1 def predict_(model=None, dataloader=None, device=None, path=None):
      2     if path:
----> 3         model.load_state_dict(torch.load(path))
      4     else:
      5         pass

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/rccl-1x144-f-train/Epoch_19.pt'
