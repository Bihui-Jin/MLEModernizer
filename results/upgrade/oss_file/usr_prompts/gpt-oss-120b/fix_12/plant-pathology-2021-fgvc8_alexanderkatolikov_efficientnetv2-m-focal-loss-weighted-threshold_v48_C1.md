# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, copy, time, sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn import preprocessing
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True  # enable TF‑32 for faster matmul on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True  # same for CUDA matmul
torch.set_float32_matmul_precision("high")
torch.manual_seed(42)  # deterministic seed for reproducibility



## === cell 1
BATCH = 256  # larger batch → fewer optimizer steps per epoch
EPOCHS = 5
WEIGHT_DECAY = 0.0
LR = 1e-5
IM_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
try:
    labeled_train_df = pd.read_csv("../input/train-labeled/train.csv")
except Exception:
    labeled_train_df = None



## === cell 3
le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])

if labeled_train_df is not None:
    my_dict = {}
    NUM_CL = len(train_df["labels"].value_counts())
    for i in range(NUM_CL):
        eq = (
            labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
            != train_df.loc[train_df["label_id"] == i].values[0][1]
        )
        if eq:
            for l in range(NUM_CL):
                eq2 = (
                    labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
                    == train_df.loc[train_df["label_id"] == l].values[0][1]
                )
                if eq2:
                    my_dict[l] = i
    train_df = train_df.replace({"label_id": my_dict})

class_map = {row[0]: row[1] for row in train_df[["label_id", "labels"]].values.tolist()}
NUM_CL = len(class_map)



## === cell 4
train_transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
val_transform = train_transform  # same steps for validation & test




## === cell 5
class GetData(Dataset):
    def __init__(self, root_dir, filenames, labels, transform):
        self.root = root_dir
        self.fnames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return img, self.fnames[idx]  # test mode
        else:
            return img, self.labels[idx]  # train/val mode




## === cell 6
full_dataset = GetData(
    TRAIN_DIR, train_df["image"].values, train_df["label_id"].values, train_transform
)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

num_workers = min(16, os.cpu_count() or 1)  # more workers for faster I/O
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)



## === cell 7
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(model.fc.in_features, NUM_CL, bias=True)

pretrained_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.isfile(pretrained_path):
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    except Exception:
        pass  # fall back to random initialization

model = model.to(DEVICE)

if DEVICE.type == "cuda":
    model = torch.compile(model, mode="max-autotune")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler()



## === cell 8
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(
            DEVICE, non_blocking=True
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} – Train loss: {epoch_loss:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

val_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Final validation F1: {val_f1:.4f}")



## === cell 9
test_fnames = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_dataset = GetData(TEST_DIR, test_fnames, None, val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)

model.eval()
pred_records = []
with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred_ids = torch.argmax(probs, dim=1).cpu().numpy()
        for fname, pid in zip(fnames, pred_ids):
            pred_records.append([fname, pid])

pred_df = pd.DataFrame(pred_records, columns=["image", "label_id"])
pred_df["labels"] = pred_df["label_id"].map(class_map)

submission = pred_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with', len(submission), "rows.")
