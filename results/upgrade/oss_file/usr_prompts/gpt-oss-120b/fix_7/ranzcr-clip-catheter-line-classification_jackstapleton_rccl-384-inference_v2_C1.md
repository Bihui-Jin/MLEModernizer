# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0

import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as DL
import torchvision.transforms as transforms


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
            continue
        if image is None:
            image = np.zeros((size, size), dtype=np.uint8)
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
        images.append(image.reshape(size, size, 1))
    return np.array(images)




## === cell 1
start_time = time.time()

ss = pd.read_csv(
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)

ts_img_names = ss["StudyInstanceUID"].values
ts_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/test/", ts_img_names, size=384
)

breaker()
print(
    "Time Taken to read data : {:.2f} minutes".format((time.time() - start_time) / 60)
)
breaker()




## === cell 2
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




## === cell 3
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = 11

    def __init__(
        self, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=2, n_folds=5
    ):
        self.filter_sizes = filter_sizes
        self.HL = HL
        self.epochs = epochs
        self.n_folds = n_folds




## === cell 4
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
        if self.use_DP:
            x = F.relu(self.MP_(self.BN1(self.CN1(x))))
            x = F.relu(self.MP_(self.BN2(self.CN2(x))))
            x = F.relu(self.MP_(self.BN3(self.CN3(x))))
            x = F.relu(self.MP_(self.BN4(self.CN4(x))))
            x = F.relu(self.MP_(self.BN5(self.CN5(x))))
            x = F.relu(self.MP_(self.BN6(self.CN6(x))))
            x = F.relu(self.MP_(self.BN7(self.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(self.DP_(self.FC1(x)))
            x = self.FC2(x)
            return x
        else:
            x = F.relu(self.MP_(self.BN1(self.CN1(x))))
            x = F.relu(self.MP_(self.BN2(self.CN2(x))))
            x = F.relu(self.MP_(self.BN3(self.CN3(x))))
            x = F.relu(self.MP_(self.BN4(self.CN4(x))))
            x = F.relu(self.MP_(self.BN5(self.CN5(x))))
            x = F.relu(self.MP_(self.BN6(self.CN6(x))))
            x = F.relu(self.MP_(self.BN7(self.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(self.FC1(x))
            x = self.FC2(x)
            return x




## === cell 5
def predict_(model=None, dataloader=None, device=None, path=None):
    if path is not None and Path(path).is_file():
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(
            f"Checkpoint not found at {path}. Proceeding with current model parameters."
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




## === cell 6
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[4096], epochs=5, n_folds=None)

train_csv_path = "../input/ranzcr-clip-catheter-line-classification/train.csv"
train_df = pd.read_csv(train_csv_path)

subset_df = train_df.reset_index(drop=True)

train_names = subset_df["StudyInstanceUID"].values
train_labels = subset_df.iloc[:, 1:12].values  # 11 label columns

train_images = getImages(
    "../input/ranzcr-clip-catheter-line-classification/train/", train_names, size=384
)

val_ratio = 0.2
val_size = int(len(train_images) * val_ratio)
train_X, val_X = train_images[val_size:], train_images[:val_size]
train_y, val_y = train_labels[val_size:], train_labels[:val_size]

train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5)]
)
val_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DS(X=train_X, y=train_y, transform=train_transform, mode="train")
val_dataset = DS(X=val_X, y=val_y, transform=val_transform, mode="train")

train_loader = DL(
    train_dataset, batch_size=cfg.tr_batch_size, shuffle=True, num_workers=0
)
val_loader = DL(val_dataset, batch_size=cfg.tr_batch_size, shuffle=False, num_workers=0)

model = CNN(filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL)
model = model.to(cfg.device)

criterion = nn.BCEWithLogitsLoss()
optimizer = model.getOptimizer(lr=1e-3)

print("Starting lightweight training...")
for epoch in range(cfg.epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(cfg.device), yb.to(cfg.device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{cfg.epochs} - Train loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"            - Val loss: {val_loss:.4f}")

print("Training completed.")

label_cols = train_df.columns[1:12]  # same order as training labels
for col in label_cols:
    if col not in ss.columns:
        ss[col] = 0.5  # placeholder if missing
ss = ss[["StudyInstanceUID"] + list(label_cols)]

transform_pipe = transforms.Compose([transforms.ToTensor()])

ts_data_setup = DS(X=ts_images, y=None, transform=transform_pipe, mode="test")
ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

ckpt_path = (
    "../input/rccl-384-train/Epoch_14.pt"  # kept for compatibility; may be missing
)
y_pred = predict_(model=model, dataloader=ts_data, device=cfg.device, path=ckpt_path)

y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

n_labels = len(label_cols)
if y_pred.shape[1] > n_labels:
    y_pred = y_pred[:, :n_labels]
elif y_pred.shape[1] < n_labels:
    pad = np.full((y_pred.shape[0], n_labels - y_pred.shape[1]), 0.5)
    y_pred = np.hstack([y_pred, pad])

ss.iloc[:, 1:] = y_pred
ss.to_csv("./submission.csv", index=False)
print("Submission file written to ./submission.csv")
ss.head(5)
