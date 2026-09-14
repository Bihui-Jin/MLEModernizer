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
import gc
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_PATH = _first_existing(
    [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../data/plant-pathology-2021-fgvc8",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
        "./data/plant-pathology-2021-fgvc8",
    ]
)

if DATA_PATH is None:
    raise FileNotFoundError("Could not find competition data directory.")


def _resolve_file(fname):
    cands = [
        os.path.join(DATA_PATH, fname),
        os.path.join(DATA_PATH, "plant-pathology-2021-fgvc8", fname),
    ]
    out = _first_existing(cands)
    if out is None:
        raise FileNotFoundError(f"Could not find required file {fname} in {cands}")
    return out


def _resolve_dir(dname):
    cands = [
        os.path.join(DATA_PATH, dname),
        os.path.join(DATA_PATH, "plant-pathology-2021-fgvc8", dname),
    ]
    out = _first_existing(cands)
    if out is None:
        raise FileNotFoundError(f"Could not find required directory {dname} in {cands}")
    return out


TRAIN_CSV = _resolve_file("train.csv")
SAMPLE_SUB = _resolve_file("sample_submission.csv")
TRAIN_IMGS_PATH = _resolve_dir("train_images")
TEST_IMGS_PATH = _resolve_dir("test_images")

print("DATA_PATH:", DATA_PATH)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMGS_PATH:", TRAIN_IMGS_PATH)
print("TEST_IMGS_PATH:", TEST_IMGS_PATH)
print("DEVICE:", DEVICE)

TTAS = [0, 1, 2, 3]
FOLDS = [0]



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
df_sub = pd.read_csv(SAMPLE_SUB)

all_labels = sorted(
    {lbl for s in df_train["labels"].astype(str).values for lbl in s.split()}
)
LABELS_ = all_labels[:]  # index -> label
LABELS = {l: i for i, l in enumerate(LABELS_)}  # label -> index

params = {
    "img_size": 256,
    "batch_size": 32,
    "dropout": 0.5,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
WORKERS = 2

print("train rows:", len(df_train), "| test rows:", len(df_sub))
print("num classes:", len(LABELS_))
print("labels:", LABELS_)



## === cell 2
df_sub = df_sub[["image", "labels"]].copy()
df_sub["labels"] = "healthy"
print(df_sub.head())
print("test images:", len(df_sub))




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(
        self,
        df,
        img_dir,
        size,
        labels,
        transform=None,
        tta=0,
        train_mode=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = size
        self.labels = labels  # dict label->index or None
        self.transform = transform
        self.tta = tta
        self.train_mode = train_mode
        self.mean = np.array(mean, dtype=np.float32)[None, None, :]
        self.std = np.array(std, dtype=np.float32)[None, None, :]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.train_mode:
            r = np.random.rand()
            if r < 0.33:
                img = flip(img, axis=1)
            elif r < 0.66:
                img = flip(img, axis=2)

        if self.labels is not None:
            img = ((img - self.mean) / self.std).copy()
            img = img.transpose(2, 0, 1).copy()
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = ((img - self.mean) / self.std).copy()
            img = img.transpose(2, 0, 1).copy()
            return torch.tensor(img)


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def forward(self, x):
        return self.rsnxt(x)




## === cell 4
idx = np.arange(len(df_train))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)

train_ds = PlantDataset(
    df_trn,
    TRAIN_IMGS_PATH,
    params["img_size"],
    LABELS,
    train_mode=True,
    mean=params["mean"],
    std=params["std"],
)
val_ds = PlantDataset(
    df_val,
    TRAIN_IMGS_PATH,
    params["img_size"],
    LABELS,
    train_mode=False,
    mean=params["mean"],
    std=params["std"],
)

train_loader = DataLoader(
    train_ds,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=params["batch_size"],
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=torch.cuda.is_available(),
)

model = ResNext(params, out_dim=len(LABELS_)).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

EPOCHS = 2

for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)

    tr_loss /= len(train_loader.dataset) if len(train_loader.dataset) else 1

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            va_loss += loss.item() * xb.size(0)
    va_loss /= len(val_loader.dataset) if len(val_loader.dataset) else 1

    print(
        f"epoch {epoch+1}/{EPOCHS} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}"
    )

models_list = [model]
gc.collect()



## === cell 5
loaders = []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        img_dir=TEST_IMGS_PATH,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=tta,
        train_mode=False,
        mean=params["mean"],
        std=params["std"],
    )
    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    loaders.append(loader)

print("Num TTAs:", len(loaders), "| Num models:", len(models_list))



## === cell 6
TH = 0.5


def get_labels_from_probs(probs_row, labels_list, th=0.5):
    idxs = [i for i, p in enumerate(probs_row) if p > th]
    if not idxs:
        return "healthy"
    lbls = [labels_list[i] for i in idxs]
    if "healthy" in lbls and len(lbls) > 1:
        lbls = [l for l in lbls if l != "healthy"]
    return " ".join(lbls) if lbls else "healthy"


all_preds = []
with torch.no_grad():
    for mi, mdl in enumerate(models_list):
        mdl.eval()
        for tj, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                probs = mdl(img_data).sigmoid().detach().cpu().numpy()
                if probs.ndim == 1:
                    probs = probs[None, :]
                preds_batches.append(probs)
            preds_tta = np.vstack(preds_batches)  # (N, C)
            all_preds.append(preds_tta)
            print(f"model {mi} | tta {tj} -> preds: {preds_tta.shape}")

all_preds = np.stack(all_preds, axis=0)  # (M*T, N, C)
probs_mean = all_preds.mean(axis=0)  # (N, C)

df_sub["labels"] = [get_labels_from_probs(x, LABELS_, th=TH) for x in probs_mean]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 7
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()



## === cell 8
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(df_sub.head())
print("Submission columns:", list(pd.read_csv(sub_path).columns))
