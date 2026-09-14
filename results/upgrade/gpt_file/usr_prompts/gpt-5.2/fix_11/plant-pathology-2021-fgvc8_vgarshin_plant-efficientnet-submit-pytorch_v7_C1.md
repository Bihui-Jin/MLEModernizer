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
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
TEST = True
VER = "v1"

if KAGGLE:
    DATA_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"/kaggle/input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
TTAS = [0, 1, 2, 3]
FOLDS = [0, 1]

_primary_imgs = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
_nested_imgs = (
    f"{DATA_PATH}/plant-pathology-2021-fgvc8/test_images"
    if TEST
    else f"{DATA_PATH}/plant-pathology-2021-fgvc8/train_images"
)
if os.path.isdir(_primary_imgs):
    IMGS_PATH = _primary_imgs
elif os.path.isdir(_nested_imgs):
    IMGS_PATH = _nested_imgs
else:
    IMGS_PATH = _primary_imgs  # keep original for error visibility

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 2
params_path = f"{MDLS_PATH}/params.json"
if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    WORKERS = 2 if KAGGLE else params.get("workers", 2)
    print("loaded params:", params)
else:
    default_labels = [
        "scab",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
        "healthy",
    ]
    LABELS_ = default_labels
    LABELS = {str(i): lab for i, lab in enumerate(default_labels)}
    params = {
        "backbone": "efficientnet_b0",
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "workers": 2,
    }
    WORKERS = 2
    print(f"WARNING: {params_path} not found. Using fallback params/labels: {params}")



## === cell 3
sub_path_primary = f"{DATA_PATH}/sample_submission.csv"
sub_path_nested = f"{DATA_PATH}/plant-pathology-2021-fgvc8/sample_submission.csv"
if os.path.exists(sub_path_primary):
    sub_path = sub_path_primary
elif os.path.exists(sub_path_nested):
    sub_path = sub_path_nested
else:
    sub_path = sub_path_primary

df_sub = pd.read_csv(sub_path)
if "labels" not in df_sub.columns:
    df_sub["labels"] = "healthy"
else:
    df_sub["labels"] = "healthy"
print("df_sub shape:", df_sub.shape)
print(df_sub.head())




## === cell 4
class PlantDataset(data.Dataset):
    def __init__(
        self,
        df,
        size,
        labels,
        transform=None,
        tta=0,
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = img.transpose(2, 0, 1)  # CHW
        base = torch.from_numpy(img)  # float32

        if self.labels:
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return base, torch.tensor(label)

        if self.tta == 0:
            return base
        elif self.tta == 1:
            return torch.flip(base, dims=(1,))  # vertical flip (H)
        elif self.tta == 2:
            return torch.flip(base, dims=(2,))  # horizontal flip (W)
        else:
            return torch.flip(base, dims=(1, 2))  # both


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = models.efficientnet_b0(weights=None)
        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
            nn.BatchNorm1d(int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
models_ens = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("loaded:", path)
        del state_dict
    else:
        print(f"WARNING: missing weights {path}. Using untrained model for this fold.")
    model.float()
    model.eval()
    model.to(DEVICE)
    models_ens.append(model)
gc.collect()



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=tta,
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
        drop_last=False,
    )
    loaders.append(loader)
print("Prepared loaders:", len(loaders))




## === cell 7
@torch.no_grad()
def predict_sum_over_models(xb, models_list):
    preds = []
    for m in models_list:
        preds.append(m(xb).sigmoid())
    return torch.stack(preds, dim=0).sum(dim=0)


with torch.no_grad():
    N = len(df_sub)
    C = len(LABELS_)
    all_preds_sum = np.zeros((N, C), dtype=np.float32)
    count = 0  # number of model predictions accumulated per sample

    for j, loader in enumerate(loaders):
        tta_sum = np.zeros((N, C), dtype=np.float32)
        ofs = 0
        for img_data in loader:
            bs = img_data.shape[0]
            img_data = img_data.to(DEVICE, non_blocking=True)
            batch_sum = predict_sum_over_models(img_data, models_ens)
            tta_sum[ofs : ofs + bs] = batch_sum.detach().cpu().numpy()
            ofs += bs

        all_preds_sum += tta_sum
        count += len(models_ens)
        print(
            f"loader {j} -> done, summed over {len(models_ens)} model(s), shape {tta_sum.shape}"
        )

logits = all_preds_sum / float(count)

idx_to_name = [LABELS[str(i)] for i in range(len(LABELS_))]
healthy_idx = idx_to_name.index("healthy") if "healthy" in idx_to_name else None

train_csv_primary = f"{DATA_PATH}/train.csv"
train_csv_nested = f"{DATA_PATH}/plant-pathology-2021-fgvc8/train.csv"
train_csv_path = (
    train_csv_primary if os.path.exists(train_csv_primary) else train_csv_nested
)
df_train = pd.read_csv(train_csv_path)

name_to_idx = {LABELS[str(i)]: i for i in range(len(LABELS_))}

Y = np.zeros((len(df_train), len(LABELS_)), dtype=np.uint8)
labels_list = df_train["labels"].astype(str).str.split().tolist()
for i, labs in enumerate(labels_list):
    idxs = [name_to_idx[lab] for lab in labs if lab in name_to_idx]
    if idxs:
        Y[i, idxs] = 1

_primary_train_imgs = f"{DATA_PATH}/train_images"
_nested_train_imgs = f"{DATA_PATH}/plant-pathology-2021-fgvc8/train_images"
if os.path.isdir(_primary_train_imgs):
    TRAIN_IMGS_PATH = _primary_train_imgs
elif os.path.isdir(_nested_train_imgs):
    TRAIN_IMGS_PATH = _nested_train_imgs
else:
    TRAIN_IMGS_PATH = _primary_train_imgs


def sample_f1_mean(y_true_bin, y_pred_bin):
    tp = (y_true_bin & y_pred_bin).sum(axis=1).astype(np.float32)
    fp = ((1 - y_true_bin) & y_pred_bin).sum(axis=1).astype(np.float32)
    fn = (y_true_bin & (1 - y_pred_bin)).sum(axis=1).astype(np.float32)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(f1.mean())


def make_folds_by_labelcount(df, n_splits=2, seed=0):
    rng = np.random.RandomState(seed)
    label_counts = df["labels"].astype(str).apply(lambda x: len(x.split())).values
    order = np.lexsort((rng.rand(len(df)), label_counts))
    folds = np.zeros(len(df), dtype=np.int64)
    for k, idx in enumerate(order):
        folds[idx] = k % n_splits
    return folds


_old_imgs_path = IMGS_PATH
IMGS_PATH = TRAIN_IMGS_PATH

fold_ids = make_folds_by_labelcount(df_train, n_splits=max(2, len(FOLDS)), seed=0)

oof_pred = np.zeros((len(df_train), len(LABELS_)), dtype=np.float32)
oof_mask = np.zeros((len(df_train),), dtype=bool)

for fold_pos, fold_id in enumerate(FOLDS):
    if fold_id >= fold_ids.max() + 1:
        continue
    val_idx = np.where(fold_ids == fold_id)[0]
    if val_idx.size == 0:
        continue

    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    val_ds = PlantDataset(
        df=df_val,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=0,  # unchanged
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(val_ds),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
        drop_last=False,
    )

    model = models_ens[fold_pos] if fold_pos < len(models_ens) else None
    if model is None:
        continue

    with torch.no_grad():
        ofs = 0
        Pv_fold = np.zeros((len(df_val), len(LABELS_)), dtype=np.float32)
        for xb in val_loader:
            bs = xb.shape[0]
            xb = xb.to(DEVICE, non_blocking=True)
            p = model(xb).sigmoid()
            Pv_fold[ofs : ofs + bs] = p.detach().cpu().numpy()
            ofs += bs

    oof_pred[val_idx] = Pv_fold
    oof_mask[val_idx] = True
    print(f"OOF preds collected for fold {fold_id}: {val_idx.size} rows")

IMGS_PATH = _old_imgs_path

Pv = oof_pred[oof_mask]
Yv = Y[oof_mask].astype(np.uint8)

grid = np.linspace(0.05, 0.95, 37, dtype=np.float32)
best_t, best_score = 0.5, -1.0
for t in grid:
    pred = (Pv > float(t)).astype(np.uint8)
    if healthy_idx is not None:
        other_any = (pred.sum(axis=1) > 1) & (pred[:, healthy_idx] == 1)
        if np.any(other_any):
            pred[other_any, healthy_idx] = 0
    score = sample_f1_mean(Yv, pred)
    if score > best_score:
        best_score, best_t = score, float(t)

thr = np.full(len(LABELS_), best_t, dtype=np.float32)
if healthy_idx is not None:
    thr[healthy_idx] = max(float(thr[healthy_idx]), 0.50)

print(
    "Global threshold (OOF-by-fold fitted for mean samplewise F1):",
    best_t,
    "OOF mean F1:",
    best_score,
)
print("Applied thresholds:", {idx_to_name[i]: float(thr[i]) for i in range(len(thr))})

mask = logits > thr[None, :]
if healthy_idx is not None:
    other_present = (mask.sum(axis=1) > 1) & (mask[:, healthy_idx])
    if np.any(other_present):
        mask[other_present, healthy_idx] = False

out_labels = []
for r in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[r])
    if idxs.size == 0:
        out_labels.append("healthy")
    else:
        out_labels.append(" ".join(idx_to_name[i] for i in idxs))
df_sub["labels"] = out_labels

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 9
out = df_sub[["image", "labels"]].copy()
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
