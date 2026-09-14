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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7485687903970452

# 6. Current score

0.15488

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'I make the notebook run end-to-end and reliably produce a valid `submission.csv` by removing the dependency on missing external inputs (`torchcontrib`, `../input/train-labeled/train.csv`) and by fixing the inference bug where `torch.exp(logits)` is incorrectly used instead of a proper softmax (this can silently harm predictions and F1). I keep the exact same model (ResNeXt101_32x8d with a linear head) and the same overall approach (single-label classification mapped back to the original space-delimited label strings), only adjusting the minimum required pieces to execute and to align prediction probabilities with the classifier output. I also ensure image loading is robust (convert to RGB) and that the submission is aligned to `sample_submission.csv` order to avoid any indexing/order-related scoring issues. These are minimal, score-relevant fixes that should move you from “no score” to a valid submission and improve the expected F1 versus the current inference/post-processing.'
- What this solution (achieved 0.15488) has done: 'I (1) make the checkpoint loading robust so the notebook runs even when `/kaggle/input/resnet-model/ResNext16.pth` is missing, by falling back to ImageNet weights (same architecture) instead of crashing. I (2) fix the device/type mismatch during inference by ensuring the model parameters are moved to `DEVICE` after any weight loading, so CUDA tensors and model weights match. I (3) keep the rest of your pipeline intact (label encoding, single-label prediction mapped back to space-delimited strings, and submission alignment), only adding minimal safety checks so a valid `submission.csv` is always written. This should both unblock execution and substantially improve score versus the current failing/weak run by using a sensible pretrained initialization instead of random weights.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import copy

import sys
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df



## === cell 3
train_df["labels"].value_counts()



## === cell 4
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL



## === cell 5
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
print(train_df)



## === cell 6
class_map = dict(
    train_df[["label_id", "labels"]]
    .drop_duplicates()
    .sort_values("label_id")
    .values.tolist()
)
len(class_map), list(class_map.items())[:3]



## === cell 7
tr_df = train_df[:trainnum]
print(len(tr_df))
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values



## === cell 8
Transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 9
Transformval = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 10
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index])).convert("RGB")

        if "train_images" in self.dir:
            return self.transform(x), self.labels[index]
        elif "test_images" in self.dir:
            return self.transform(x), self.fnames[index]
        else:
            if self.labels is None:
                return self.transform(x), self.fnames[index]
            return self.transform(x), self.labels[index]




## === cell 11
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 12
val_df = train_df[-valnum:]
print(len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 13
next(iter(trainloader))[0].shape




## === cell 14
def PREDS(l):
    word = train_df.loc[train_df["label_id"] == l].values[0][1]  # finds label's name
    words = word.split(" ")
    return words




## === cell 15
def metrics(preds, labels, tp, fn, fp):
    a = preds.tolist()
    b = labels.tolist()
    for i in range(len(preds)):
        tp += len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fn += len(PREDS(b[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fp += len(PREDS(a[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
    return tp, fn, fp




## === cell 16
ckpt_path = "/kaggle/input/resnet-model/ResNext16.pth"

model = torchvision.models.resnext101_32x8d(weights=None)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

loaded = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    loaded = True
else:
    try:
        w = torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        model = torchvision.models.resnext101_32x8d(weights=w)
        model.fc = nn.Linear(2048, NUM_CL, bias=True)
        loaded = True
        print(
            f"Checkpoint not found at {ckpt_path}. Using torchvision ImageNet weights instead."
        )
    except Exception as e:
        print(
            f"Checkpoint not found and could not load torchvision weights due to: {e}. Using random init."
        )

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)



## === cell 17
testnum = 3700
test_df = train_df[-testnum:]
X_test, Y_test = test_df["image"].values, test_df["label_id"].values
testset_dbg = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
testloader_dbg = DataLoader(
    testset_dbg,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 18
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
X_Test = sample_sub["image"].tolist()
len(X_Test), X_Test[:3]



## === cell 19
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 20
pred_rows = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader:
        image = image.to(DEVICE, non_blocking=True)
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        top_class = torch.argmax(probs, dim=1)
        pred_rows.append([fname[0], int(top_class.item())])



## === cell 21
pred_df = pd.DataFrame.from_records(pred_rows, columns=["image", "label_id"])
pred_df.head(), len(pred_df)



## === cell 22
pred_df["labels"] = pred_df["label_id"].map(class_map)
pred_df.head()



## === cell 23
sub = sample_sub[["image"]].merge(pred_df[["image", "labels"]], on="image", how="left")
if sub["labels"].isna().any():
    fallback = train_df["labels"].mode().iloc[0]
    sub["labels"] = sub["labels"].fillna(fallback)

sub.head(), sub.shape



## === cell 24
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
