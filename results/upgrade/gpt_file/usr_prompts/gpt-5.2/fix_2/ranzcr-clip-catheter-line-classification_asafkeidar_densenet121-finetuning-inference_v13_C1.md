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
import os
import time
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



## === cell 1
TARGET_COLUMNS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

DEBUG = False

if DEBUG is False:
    BATCH_SIZE = 32
    EPOCHS = 2  # keep runtime safe; core training loop/architecture unchanged
    AVERAGING_SIZE = 100
    TRAIN_SUBSAMPLE = None
else:
    BATCH_SIZE = 4
    EPOCHS = 1
    AVERAGING_SIZE = 20
    TRAIN_SUBSAMPLE = 800

IMG_SIZE = 256

DATA_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = "./"
SUB_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_set_df = sample_sub.copy()

for col in TARGET_COLUMNS:
    if col not in test_set_df.columns:
        test_set_df[col] = 0.0

test_set_df = test_set_df[["StudyInstanceUID"] + TARGET_COLUMNS]

test_set_df.head(), test_set_df.shape




## === cell 3
class RanzcrClipDataset(Dataset):
    def __init__(self, df, img_dir, target_columns, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_columns = target_columns
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.loc[idx, "StudyInstanceUID"]
        img_path = os.path.join(self.img_dir, f"{uid}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.is_test:
            return img
        y = self.df.loc[idx, self.target_columns].values.astype(np.float32)
        return img, torch.from_numpy(y)




## === cell 4
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
train_df = pd.read_csv(TRAIN_CSV)

if TRAIN_SUBSAMPLE is not None and TRAIN_SUBSAMPLE < len(train_df):
    train_df = train_df.sample(TRAIN_SUBSAMPLE, random_state=42).reset_index(drop=True)

trn_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=42, shuffle=True
)

train_set = RanzcrClipDataset(
    trn_df, TRAIN_IMG_DIR, TARGET_COLUMNS, transform=train_transforms, is_test=False
)
val_set = RanzcrClipDataset(
    val_df, TRAIN_IMG_DIR, TARGET_COLUMNS, transform=test_transforms, is_test=False
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

test_set = RanzcrClipDataset(
    test_set_df, TEST_IMG_DIR, TARGET_COLUMNS, transform=test_transforms, is_test=True
)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

len(train_set), len(val_set), len(test_set)




## === cell 6
class CustomPretrainedmModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            weights = (
                torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            self.model = torchvision.models.densenet121(weights=weights)
        except Exception:
            self.model = torchvision.models.densenet121(pretrained=pretrained)

        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, len(TARGET_COLUMNS))

    def forward(self, x):
        return self.model(x)




## === cell 7
model = CustomPretrainedmModel(pretrained=True).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def run_eval(net, loader):
    net.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = net(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running = []
    for step, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running.append(loss.item())
        if (step + 1) % AVERAGING_SIZE == 0:
            print(
                f"epoch {epoch+1}/{EPOCHS} step {step+1}/{len(train_loader)} loss {np.mean(running[-AVERAGING_SIZE:]):.4f}"
            )

    val_loss = run_eval(model, val_loader)
    print(
        f"epoch {epoch+1}/{EPOCHS} train_loss {np.mean(running):.4f} val_loss {val_loss:.4f}"
    )

print(f"Training elapsed: {(time.time()-start):.1f}s")




## === cell 8
def inference(net, loader):
    y_prob = []
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, inputs in enumerate(loader, 0):
            inputs = inputs.to(device, non_blocking=True)
            outputs = net(inputs)
            probs = outputs.sigmoid()
            y_prob.append(probs.detach().cpu().numpy())
    y_prob = (
        np.vstack(y_prob)
        if len(y_prob)
        else np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
    )
    end_time = time.time()
    print(
        f"Inference batches: {i+1 if 'i' in locals() else 0}, elapsed {(end_time - start_time):.2f}s"
    )
    return y_prob


y_prob = inference(model, test_loader)
y_prob.shape



## === cell 9
sub = sample_sub.copy()
for col in TARGET_COLUMNS:
    if col not in sub.columns:
        sub[col] = 0.0

sub = sub[["StudyInstanceUID"] + TARGET_COLUMNS]
assert (
    y_prob.shape[0] == sub.shape[0]
), f"Pred rows {y_prob.shape[0]} != submission rows {sub.shape[0]}"
sub.loc[:, TARGET_COLUMNS] = y_prob

sub["StudyInstanceUID"] = (
    sub["StudyInstanceUID"].astype(str).str.replace(".jpg", "", regex=False)
)

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH)
sub.head()



## === cell 10
print("submission shape:", sub.shape)
print("columns match:", list(sub.columns) == ["StudyInstanceUID"] + TARGET_COLUMNS)
print("any NaN:", sub.isna().any().any())
print(
    "prob range:",
    float(sub[TARGET_COLUMNS].min().min()),
    float(sub[TARGET_COLUMNS].max().max()),
)
