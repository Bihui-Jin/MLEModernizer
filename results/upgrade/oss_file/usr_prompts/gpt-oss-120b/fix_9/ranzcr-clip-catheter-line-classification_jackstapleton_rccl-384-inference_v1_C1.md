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

0.9088197075841224

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5376) has done: 'I rename the cells to start at 1, keep the original logic, and fix the shape mismatch when writing the submission: after obtaining the predictions I truncate or pad them so that the number of columns exactly matches the columns present in the sample‑submission file. This prevents the `ValueError` and ensures a valid CSV is written.'
- What this solution (achieved 0.5127) has done: 'I fixed the runtime error caused by using a NumPy view with negative strides when flipping test images. The flip operation now creates a contiguous copy, which `torchvision.transforms.ToTensor` can handle. I also renumbered the notebook cells to start at 1 as required and kept the original logic unchanged, ensuring the script runs end‑to‑end and writes a valid `submission.csv` with the correct number of columns.'
- What this solution (achieved 0.47817) has done: 'I add a simple vertical‑flip test‑time augmentation and include its predictions in the ensemble average. This keeps the original model and training unchanged, only expands inference with an extra deterministic augmentation that usually improves AUC, moving the score upward toward the target.'
- What this solution (achieved 0.49365) has done: 'I add a proper vertical‑flip test‑time augmentation (axis 0) and include its predictions in the ensemble average. This keeps the original model and training untouched while providing an extra informative view of each image, which should raise the AUC toward the target score.'
- What this solution (achieved 0.50357) has done: 'I rename the cells so they start at 1 (required for proper execution) and add a fourth test‑time augmentation – a horizontal + vertical flip (equivalent to a 180° rotation). Its predictions are averaged together with the original, horizontal‑flip and vertical‑flip predictions, which usually yields a modest AUC boost while keeping the core model unchanged. This small change moves the score closer to the target without altering training logic.'

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
import torch.nn.functional as F

import gc
import os
import cv2
from time import time

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
    def __init__(self, X=None, y=None, transform=None, mode="train"):
        self.mode = mode
        self.transform = transform
        self.X = X
        if mode == "train":
            self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.transform(self.X[idx])
        if self.mode == "train":
            return img, torch.FloatTensor(self.y[idx])
        else:
            return img




## === cell 4
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = 11  # number of output logits (matches training labels)

    def __init__(
        self, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=50, n_folds=5
    ):
        self.filter_sizes = filter_sizes
        self.HL = HL
        self.epochs = epochs
        self.n_folds = n_folds




## === cell 5
class CNN(nn.Module):
    def __init__(
        self, in_channels=1, filter_sizes=None, HL=None, OL=None, use_DP=True, DP=0.50
    ):
        super(CNN, self).__init__()
        self.use_DP = use_DP
        self.DP_ = nn.Dropout(p=DP)
        self.MP_ = nn.MaxPool2d(kernel_size=2)

        self.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        self.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        self.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        self.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        self.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        self.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        self.CN7 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.BN7 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        self.FC1 = nn.Linear(in_features=filter_sizes[3] * 3 * 3, out_features=HL[0])
        self.FC2 = nn.Linear(in_features=HL[0], out_features=OL)

    def getOptimizer(self, lr=1e-3, wd=0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        x = F.relu(self.MP_(self.BN1(self.CN1(x))))
        x = F.relu(self.MP_(self.BN2(self.CN2(x))))
        x = F.relu(self.MP_(self.BN3(self.CN3(x))))
        x = F.relu(self.MP_(self.BN4(self.CN4(x))))
        x = F.relu(self.MP_(self.BN5(self.CN5(x))))
        x = F.relu(self.MP_(self.BN6(self.CN6(x))))
        x = F.relu(self.MP_(self.BN7(self.CN7(x))))

        x = x.view(x.shape[0], -1)
        if self.use_DP:
            x = F.relu(self.DP_(self.FC1(x)))
        else:
            x = F.relu(self.FC1(x))
        x = self.FC2(x)
        return x




## === cell 6
def predict_(model=None, dataloader=None, device=None, path=None, use_dropout=False):
    """
    Load checkpoint if available, then run inference.
    If use_dropout=True, the model is kept in train mode so dropout layers stay active.
    """
    if path and os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"[INFO] Checkpoint not found at '{path}'. Using untrained model.")
    model.to(device)
    model.eval()
    if use_dropout:
        model.train()  # enables dropout; gradients are still not computed because of torch.no_grad

    y_pred = torch.zeros(1, 11).to(device)  # placeholder for concatenation

    for batch in dataloader:
        X = batch.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, pred), dim=0)

    return y_pred[1:].cpu().numpy()




## === cell 7
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=30, n_folds=5)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

ts_data_setup = DS(X=ts_images, y=None, transform=transform, mode="test")
ts_loader = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

ts_images_hflip = np.flip(ts_images, axis=1).copy()
ts_data_hflip = DS(X=ts_images_hflip, y=None, transform=transform, mode="test")
ts_loader_hflip = DL(ts_data_hflip, batch_size=cfg.ts_batch_size, shuffle=False)

ts_images_vflip = np.flip(ts_images, axis=0).copy()
ts_data_vflip = DS(X=ts_images_vflip, y=None, transform=transform, mode="test")
ts_loader_vflip = DL(ts_data_vflip, batch_size=cfg.ts_batch_size, shuffle=False)

ts_images_hvflip = np.flip(np.flip(ts_images, axis=0), axis=1).copy()
ts_data_hvflip = DS(X=ts_images_hvflip, y=None, transform=transform, mode="test")
ts_loader_hvflip = DL(ts_data_hvflip, batch_size=cfg.ts_batch_size, shuffle=False)

ts_images_rot90 = np.rot90(ts_images, k=1, axes=(0, 1)).copy()
ts_data_rot90 = DS(X=ts_images_rot90, y=None, transform=transform, mode="test")
ts_loader_rot90 = DL(ts_data_rot90, batch_size=cfg.ts_batch_size, shuffle=False)

ts_images_rot270 = np.rot90(ts_images, k=3, axes=(0, 1)).copy()
ts_data_rot270 = DS(X=ts_images_rot270, y=None, transform=transform, mode="test")
ts_loader_rot270 = DL(ts_data_rot270, batch_size=cfg.ts_batch_size, shuffle=False)

model = CNN(filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL)
ckpt_path = "../input/rccl-384-train/Epoch_14.pt"

pred_orig = predict_(model, ts_loader, cfg.device, ckpt_path, use_dropout=True)
pred_hflip = predict_(model, ts_loader_hflip, cfg.device, ckpt_path, use_dropout=True)
pred_vflip = predict_(model, ts_loader_vflip, cfg.device, ckpt_path, use_dropout=True)
pred_hvflip = predict_(model, ts_loader_hvflip, cfg.device, ckpt_path, use_dropout=True)
pred_rot90 = predict_(model, ts_loader_rot90, cfg.device, ckpt_path, use_dropout=True)
pred_rot270 = predict_(model, ts_loader_rot270, cfg.device, ckpt_path, use_dropout=True)

y_pred = (
    pred_orig + pred_hflip + pred_vflip + pred_hvflip + pred_rot90 + pred_rot270
) / 6.0

num_target_cols = ss.shape[1] - 1  # columns after the ID column
if y_pred.shape[1] > num_target_cols:
    y_pred = y_pred[:, :num_target_cols]  # truncate excess logits
elif y_pred.shape[1] < num_target_cols:
    pad_width = num_target_cols - y_pred.shape[1]
    y_pred = np.concatenate([y_pred, np.zeros((y_pred.shape[0], pad_width))], axis=1)

y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

ss.iloc[:, 1:] = y_pred
submission_path = "./submission.csv"
ss.to_csv(submission_path, index=False)
print(f"Submission saved to '{submission_path}'")
ss.head(5)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1446011184.py in <cell line: 0>()
     42 pred_vflip = predict_(model, ts_loader_vflip, cfg.device, ckpt_path, use_dropout=True)
     43 pred_hvflip = predict_(model, ts_loader_hvflip, cfg.device, ckpt_path, use_dropout=True)
---> 44 pred_rot90 = predict_(model, ts_loader_rot90, cfg.device, ckpt_path, use_dropout=True)
     45 pred_rot270 = predict_(model, ts_loader_rot270, cfg.device, ckpt_path, use_dropout=True)
     46 

/tmp/ipykernel_55/1070308378.py in predict_(model, dataloader, device, path, use_dropout)
     18         X = batch.to(device)
     19         with torch.no_grad():
---> 20             pred = torch.sigmoid(model(X))
     21         y_pred = torch.cat((y_pred, pred), dim=0)
     22 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3403282548.py in forward(self, x)
     86         x = x.view(x.shape[0], -1)
     87         if self.use_DP:
---> 88             x = F.relu(self.DP_(self.FC1(x)))
     89         else:
     90             x = F.relu(self.FC1(x))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x35328 and 4608x2048)
