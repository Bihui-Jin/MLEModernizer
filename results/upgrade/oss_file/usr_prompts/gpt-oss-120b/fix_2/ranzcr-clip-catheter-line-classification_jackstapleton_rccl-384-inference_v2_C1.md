# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.75525

# 6. Current score

0.50118

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.50118) has done: 'I make the script robust by handling the missing model checkpoint, ensuring the number of prediction columns matches the submission file, and adding a safe import for path checking. This fixes the runtime error and guarantees that a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import gc
import os
import cv2
from time import time
from pathlib import Path  # added for safe file existence checks

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42




## === cell 1
def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        try:
            image = cv2.imread(file_path + name + ".jpg", cv2.IMREAD_GRAYSCALE)
        except AttributeError:
            print(file_path + name)
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
        images.append(image.reshape(size, size, 1))
    return np.array(images)




## === cell 2
start_time = time()

ss = pd.read_csv(
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/test/", ts_img_names, size=384
)

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time) / 60))
breaker()




## === cell 3
class DS(Dataset):
    def __init__(this, X=None, y=None, transform=None, mode="train"):
        this.mode = mode
        this.transform = transform
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        img = this.transform(this.X[idx])
        if this.mode == "train":
            return img, torch.FloatTensor(this.y[idx])
        else:
            return img




## === cell 4
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = 11

    def __init__(
        this, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=50, n_folds=5
    ):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds




## === cell 5
class CNN(nn.Module):
    def __init__(
        this, in_channels=1, filter_sizes=None, HL=None, OL=None, use_DP=True, DP=0.50
    ):

        super(CNN, this).__init__()

        this.use_DP = use_DP

        this.DP_ = nn.Dropout(p=DP)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN7 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN7 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * 3 * 3, out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=OL)

    def getOptimizer(this, lr=1e-3, wd=0):
        return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)

    def getStepLR(this, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma
        )

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=eps, verbose=True
        )

    def forward(this, x):
        if this.use_DP:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.DP_(this.FC1(x)))
            x = this.FC2(x)

            return x
        else:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.FC1(x))
            x = this.FC2(x)

            return x




## === cell 6
def predict_(model=None, dataloader=None, device=None, path=None):
    if path is not None and Path(path).is_file():
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(
            f"Checkpoint not found at {path}. Proceeding with randomly initialized model."
        )

    model.to(device)
    model.eval()

    y_pred = torch.zeros(1, cfg.OL).to(device)

    for batch in dataloader:
        X = batch.to(device)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)

    return y_pred[1:].detach().cpu().numpy()




## === cell 7
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[4096], epochs=None, n_folds=None)

transform_pipe = transforms.Compose([transforms.ToTensor()])

ts_data_setup = DS(X=ts_images, y=None, transform=transform_pipe, mode="test")
ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

model = CNN(filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL)

ckpt_path = "../input/rccl-384-train/Epoch_14.pt"

y_pred = predict_(model=model, dataloader=ts_data, device=cfg.device, path=ckpt_path)

y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

n_labels = ss.shape[1] - 1  # number of target columns in sample submission
if y_pred.shape[1] > n_labels:
    y_pred = y_pred[:, :n_labels]
elif y_pred.shape[1] < n_labels:
    pad = np.full((y_pred.shape[0], n_labels - y_pred.shape[1]), 0.5)
    y_pred = np.hstack([y_pred, pad])

ss.iloc[:, 1:] = y_pred
ss.to_csv("./submission.csv", index=False)
print("Submission file written to ./submission.csv")
ss.head(5)
