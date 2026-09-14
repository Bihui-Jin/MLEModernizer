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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

albumentations==2.0.8
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"



## === cell 1
import albumentations as A



## === cell 2
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub.copy()
test_df["data_type"] = "test"

labels = labels[
    labels["id"].apply(lambda x: os.path.exists(os.path.join(TRAIN_DIR, x)))
].reset_index(drop=True)
test_df = test_df[
    test_df["id"].apply(lambda x: os.path.exists(os.path.join(TEST_DIR, x)))
].reset_index(drop=True)

print("train rows:", len(labels), "test rows:", len(test_df))



## === cell 3
train_df, valid_df = train_test_split(
    labels, stratify=labels["has_cactus"], test_size=0.2, random_state=SEED
)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("train/valid:", len(train_df), len(valid_df))
print(
    "train pos rate:",
    train_df["has_cactus"].mean(),
    "valid pos rate:",
    valid_df["has_cactus"].mean(),
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1710523120.py in <cell line: 0>()
----> 1 train_df, valid_df = train_test_split(
      2     labels, stratify=labels["has_cactus"], test_size=0.2, random_state=SEED
      3 )
      4 train_df = train_df.reset_index(drop=True)
      5 valid_df = valid_df.reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2234 
   2235     if n_train == 0:
-> 2236         raise ValueError(
   2237             "With n_samples={}, test_size={} and train_size={}, the "
   2238             "resulting train set will be empty. Adjust any of the "

ValueError: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

## === cell 4
def reader_fn(row):
    if row["data_type"] == "train":
        path = os.path.join(TRAIN_DIR, row["id"])
    else:
        path = os.path.join(TEST_DIR, row["id"])
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    image = image[:, :, ::-1]  # BGR -> RGB
    label = float(row.get("has_cactus", -1))
    return image, label


def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        p=p,
    )




## === cell 5
class CactusDataset(Dataset):
    def __init__(self, df, size=32, augment=False):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.augment = augment
        self.aug = augs(p=0.8) if augment else None
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        img, label = reader_fn(row)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.aug is not None:
            img = self.aug(image=img)["image"]

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        x = torch.from_numpy(img).float()
        y = torch.tensor([label], dtype=torch.float32)
        return {"image": x, "label": y, "id": row["id"]}




## === cell 6
batch_size = 64
workers = 0  # keep 0 for Kaggle notebook stability

train_ds = CactusDataset(train_df, size=32, augment=True)
valid_ds = CactusDataset(valid_df, size=32, augment=False)
test_ds = CactusDataset(test_df, size=32, augment=False)

train_dl = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True
)
val_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/965901147.py in <cell line: 0>()
      2 workers = 0  # keep 0 for Kaggle notebook stability
      3 
----> 4 train_ds = CactusDataset(train_df, size=32, augment=True)
      5 valid_ds = CactusDataset(valid_df, size=32, augment=False)
      6 test_ds = CactusDataset(test_df, size=32, augment=False)

NameError: name 'train_df' is not defined

## === cell 7
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)




## === cell 8
class Net(nn.Module):
    def __init__(self, num_classes=1, p=0.2, pooling_size=2):
        super().__init__()
        try:
            weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        self.backbone = torchvision.models.densenet169(weights=weights)
        self.features = self.backbone.features

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(1664 * 2 * pooling_size * pooling_size),
            nn.Dropout(p),
            nn.Linear(1664 * 2 * pooling_size * pooling_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.head(x)
        return x


model = Net(num_classes=1, p=0.2, pooling_size=2).to(device)
criterion = nn.BCEWithLogitsLoss()




## === cell 9
def bce_accuracy(target, logits, thresh=0.5):
    target = target.detach().cpu().numpy().reshape(-1)
    preds = (torch.sigmoid(logits).detach().cpu().numpy().reshape(-1) > thresh).astype(
        np.int32
    )
    return accuracy_score(target, preds)


def roc_auc(target, logits):
    target = target.detach().cpu().numpy().reshape(-1)
    preds = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    try:
        return roc_auc_score(target, preds)
    except Exception:
        return np.nan




## === cell 10
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-2, momentum=0.99, weight_decay=0.0
)

EPOCHS = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_epoch(train=True):
    model.train(train)
    loader = train_dl if train else val_dl
    total_loss = 0.0
    accs, aucs = [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        accs.append(bce_accuracy(y, logits))
        aucs.append(roc_auc(y, logits))

    mean_loss = total_loss / len(loader.dataset)
    mean_acc = float(np.nanmean(accs))
    mean_auc = float(np.nanmean(aucs))
    return mean_loss, mean_acc, mean_auc


start = time.time()
best_val_auc = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_auc = run_epoch(train=True)
    va_loss, va_acc, va_auc = run_epoch(train=False)
    scheduler.step()

    if va_auc > best_val_auc:
        best_val_auc = va_auc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    print(
        f"epoch {epoch:02d}/{EPOCHS} | "
        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.4f} | "
        f"valid loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f}"
    )

print("train time (s):", round(time.time() - start, 1), "best val auc:", best_val_auc)

if best_state is not None:
    model.load_state_dict(best_state)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2395494766.py in <cell line: 0>()
     44 
     45 for epoch in range(1, EPOCHS + 1):
---> 46     tr_loss, tr_acc, tr_auc = run_epoch(train=True)
     47     va_loss, va_acc, va_auc = run_epoch(train=False)
     48     scheduler.step()

/tmp/ipykernel_11/2395494766.py in run_epoch(train)
     12 def run_epoch(train=True):
     13     model.train(train)
---> 14     loader = train_dl if train else val_dl
     15     total_loss = 0.0
     16     accs, aucs = [], []

NameError: name 'train_dl' is not defined

## === cell 11
@torch.no_grad()
def predict_proba(loader):
    model.eval()
    probs = []
    ids = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
        ids.extend(batch["id"])
    probs = np.concatenate(probs, axis=0)
    return ids, probs


test_ids, test_probs = predict_proba(test_dl)
pred_df = pd.DataFrame({"id": test_ids, "has_cactus": test_probs})



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4191033798.py in <cell line: 0>()
     14 
     15 
---> 16 test_ids, test_probs = predict_proba(test_dl)
     17 pred_df = pd.DataFrame({"id": test_ids, "has_cactus": test_probs})
     18 

NameError: name 'test_dl' is not defined

## === cell 12
sub = pd.read_csv(SAMPLE_SUB)
sub = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))

if "has_cactus_pred" in sub.columns:
    sub["has_cactus"] = sub["has_cactus_pred"].fillna(0.5).astype(float)
    sub = sub[["id", "has_cactus"]]
else:
    sub["has_cactus"] = sub["has_cactus"].fillna(0.5).astype(float)

sub["has_cactus"] = sub["has_cactus"].clip(0.0, 1.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1381643282.py in <cell line: 0>()
      1 # Fix submission row-count mismatch by aligning to sample_submission ids exactly and preserving order.
      2 sub = pd.read_csv(SAMPLE_SUB)
----> 3 sub = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
      4 
      5 # If any missing (shouldn't happen), fill with 0.5 (neutral)

NameError: name 'pred_df' is not defined
