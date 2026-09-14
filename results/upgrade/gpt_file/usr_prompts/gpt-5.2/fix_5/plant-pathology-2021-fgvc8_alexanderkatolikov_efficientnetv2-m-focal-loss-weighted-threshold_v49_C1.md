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
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import copy
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # safe for fixed-size inputs; speeds convs
    torch.backends.cudnn.deterministic = False



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

_NUM_WORKERS = min(4, (os.cpu_count() or 2))
_PIN_MEMORY = torch.cuda.is_available()

_CACHE_TRAIN = True
_CACHE_VAL = True
_CACHE_TEST = True



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train_df



## === cell 3
train_df["labels"].value_counts()



## === cell 4
all_tokens = sorted(
    {t for s in train_df["labels"].astype(str).values for t in s.split(" ") if t}
)
token2id = {t: i for i, t in enumerate(all_tokens)}
id2token = {i: t for t, i in token2id.items()}
NUM_CL = len(all_tokens)
NUM_CL




## === cell 5
def labels_to_multihot(label_str, num_classes=NUM_CL):
    y = np.zeros(num_classes, dtype=np.float32)
    for t in str(label_str).split(" "):
        if t:
            y[token2id[t]] = 1.0
    return y


train_df["y"] = train_df["labels"].apply(labels_to_multihot)
train_df[["image", "labels"]].head()



## === cell 6
tr_df = train_df[:trainnum].reset_index(drop=True)
print(len(tr_df))
X_Train = tr_df["image"].values
Y_Train = np.stack(tr_df["y"].values)



## === cell 7
Transform = transforms.Compose(
    [
        transforms.Resize(
            (IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 8
Transformval = transforms.Compose(
    [
        transforms.Resize(
            (IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 9
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform, cache=False):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels
        self.cache = cache
        self._cache_x = {} if cache else None  # index -> Tensor (CPU)
        self._cache_y = {} if (cache and Labels is not None) else None

    def __len__(self):
        return len(self.fnames)

    def _load_and_transform(self, index):
        fp = os.path.join(self.dir, self.fnames[index])
        with Image.open(fp) as img:
            x = img.convert("RGB")
        x = self.transform(x)
        return x

    def __getitem__(self, index):
        if self.cache and index in self._cache_x:
            x = self._cache_x[index]
        else:
            x = self._load_and_transform(index)
            if self.cache:
                self._cache_x[index] = x

        if "train" in self.dir:
            if self.cache and index in self._cache_y:
                y = self._cache_y[index]
            else:
                y = torch.from_numpy(self.labels[index])
                if self.cache:
                    self._cache_y[index] = y
            return x, y
        elif "test" in self.dir:
            return x, self.fnames[index]




## === cell 10
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform, cache=_CACHE_TRAIN)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 11
val_df = train_df[-valnum:].reset_index(drop=True)
print(len(val_df))
X_val = val_df["image"].values
Y_val = np.stack(val_df["y"].values)

valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval, cache=_CACHE_VAL)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 12
next(iter(trainloader))[0].shape




## === cell 13
def PREDS_FROM_PROBS(probs, thr=0.5):
    idx = (probs >= thr).nonzero(as_tuple=False).view(-1).tolist()
    if len(idx) == 0:
        return ["healthy"]
    toks = [id2token[i] for i in idx]
    return toks




## === cell 14
def micro_f1_from_logits(logits, targets, thr=0.5, eps=1e-9):
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).to(targets.dtype)
    tp = (pred * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall




## === cell 15
model = torchvision.models.resnext101_32x8d(weights=None)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)




## === cell 16
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        stream = self.stream

        def _preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(stream):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    a, b = batch
                    a = a.to(self.device, non_blocking=True)
                    if torch.is_tensor(b):
                        b = b.to(self.device, non_blocking=True)
                    next_batch = (a, b)
                else:
                    next_batch = batch

        next_batch = None
        _preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            _preload()
            yield batch




## === cell 17
train_iterable = (
    CUDAPrefetcher(trainloader, DEVICE) if DEVICE.type == "cuda" else trainloader
)
val_iterable = CUDAPrefetcher(valloader, DEVICE) if DEVICE.type == "cuda" else valloader

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n = 0

    for imgs, ys in train_iterable:
        if DEVICE.type != "cuda":
            imgs = imgs.to(DEVICE)
            ys = ys.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n += bs

    train_loss = running_loss / max(1, n)

    model.eval()
    with torch.no_grad():
        vloss = 0.0
        vn = 0
        for imgs, ys in val_iterable:
            if DEVICE.type != "cuda":
                imgs = imgs.to(DEVICE)
                ys = ys.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, ys)
            bs = imgs.size(0)
            vloss += loss.item() * bs
            vn += bs
        val_loss = vloss / max(1, vn)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}"
    )



## === cell 18
X_Test = sorted(
    [
        name
        for name in os.listdir(TEST_DIR)
        if os.path.isfile(os.path.join(TEST_DIR, name))
    ]
)



## === cell 19
testset = GetData(TEST_DIR, X_Test, None, Transformval, cache=_CACHE_TEST)
testloader = DataLoader(
    testset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 20
test_iterable = (
    CUDAPrefetcher(testloader, DEVICE) if DEVICE.type == "cuda" else testloader
)

s_ls = []
THR = 0.5

with torch.no_grad():
    model.eval()
    for image, fname in test_iterable:
        if DEVICE.type != "cuda":
            image = image.to(DEVICE)
        logits = model(image)
        probs = torch.sigmoid(logits).detach().cpu()
        for fn, pr in zip(fname, probs):
            toks = PREDS_FROM_PROBS(pr, thr=THR)
            s_ls.append([fn, " ".join(toks)])



## === cell 21
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])
pred_df



## === cell 22
pred_df.head()



## === cell 23
sub = pred_df[["image", "labels"]].copy()

sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
sub = sample_sub[["image"]].merge(sub, on="image", how="left")

sub["labels"] = sub["labels"].fillna("healthy").astype(str)
sub.loc[sub["labels"].str.strip().eq(""), "labels"] = "healthy"

sub.head()



## === cell 24
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
