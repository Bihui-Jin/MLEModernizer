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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tqdm==4.67.1

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
import os
import csv
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from tqdm import tqdm

torch.set_num_threads(1)

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test_images")

with open(TRAIN_CSV_PATH, "r") as f:
    train_lines = list(csv.reader(f))[1:]  # list of [image, labels]

train_lines_split, valid_lines_split = train_test_split(
    train_lines,
    test_size=0.2,
    random_state=42,
    stratify=[row[1].split()[0] for row in train_lines],
)

train_csv = ["{},{}".format(row[0], row[1]) for row in train_lines_split]
valid_csv = ["{},{}".format(row[0], row[1]) for row in valid_lines_split]



## === cell 1
base_transform_train = transforms.Compose(
    [
        transforms.Resize(256),  # deterministic resize for caching
        transforms.ToTensor(),  # convert once and store
    ]
)

train_transform_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
class LazyTrainDataset(Dataset):
    """Training dataset that loads images on‑demand, applying a deterministic
    base transform followed by stochastic augmentations."""

    def __init__(
        self,
        img_root,
        csv_lines,
        label2idx=None,
        base_transform=None,
        aug_transform=None,
    ):
        self.img_root = img_root
        self.records = [line.strip().split(",") for line in csv_lines]
        self.labels = [rec[1].split()[0] for rec in self.records]

        if label2idx is None:
            uniq = sorted(set(self.labels))
            self.label2idx = {lbl: idx for idx, lbl in enumerate(uniq)}
        else:
            self.label2idx = label2idx
        self.idx2label = {idx: lbl for lbl, idx in self.label2idx.items()}

        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.img_root, rec[0])
        img = Image.open(img_path).convert("RGB")
        if self.base_transform:
            img = self.base_transform(img)
        if self.aug_transform:
            img = self.aug_transform(img)
        label = self.label2idx[self.labels[idx]]
        return img, label


class LazyDataset(Dataset):
    """Validation / test dataset that loads images on‑demand."""

    def __init__(self, img_root, csv_lines, label2idx, transform=None):
        self.img_root = img_root
        self.records = [line.strip().split(",") for line in csv_lines]
        self.labels = [rec[1].split()[0] for rec in self.records]
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.img_root, rec[0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label2idx[self.labels[idx]]
        return img, label


class TestDataset(Dataset):
    """Dataset for the test set (no labels)."""

    def __init__(self, img_root, csv_lines, transform=None):
        self.img_root = img_root
        self.records = [line.strip().split(",") for line in csv_lines]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.img_root, rec[0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec[0]  # return image tensor and filename




## === cell 3
train_dataset = LazyTrainDataset(
    TRAIN_IMG_ROOT,
    train_csv,
    base_transform=base_transform_train,
    aug_transform=train_transform_aug,
)

valid_dataset = LazyDataset(
    TRAIN_IMG_ROOT,
    valid_csv,
    label2idx=train_dataset.label2idx,
    transform=transform_valid,
)

worker_count = min(4, os.cpu_count() or 1)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)



## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_dataset.label2idx)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler


def train_one_epoch():
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(train_loader, desc="Training", leave=False, disable=True):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in tqdm(
            valid_loader, desc="Validating", leave=False, disable=True
        ):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


for epoch in range(2):
    loss = train_one_epoch()
    acc = evaluate()
    print(f"Epoch {epoch+1}: Train loss {loss:.4f}, Validation acc {acc:.4f}")



## === cell 5
with open(TEST_CSV_PATH, "r") as f:
    test_lines = list(csv.reader(f))[1:]  # header skip

test_csv = [
    "{},{}".format(row[0], row[1]) for row in test_lines
]  # second column is placeholder
test_dataset = TestDataset(TEST_IMG_ROOT, test_csv, transform=transform_valid)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    prefetch_factor=2,
)

model.eval()
predictions = []
with torch.no_grad():
    for imgs, filenames in tqdm(test_loader, desc="Predicting", disable=True):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        for fn, p in zip(filenames, preds.cpu().numpy()):
            label = train_dataset.idx2label[p]
            predictions.append((fn, label))

submission_path = "/kaggle/working/submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "labels"])
    for img_name, label in predictions:
        writer.writerow([img_name, label])

print(f"Submission written to {submission_path}")
