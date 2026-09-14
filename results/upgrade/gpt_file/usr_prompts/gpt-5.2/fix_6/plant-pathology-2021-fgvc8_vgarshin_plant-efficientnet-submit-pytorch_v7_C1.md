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

# 5. Target score

0.8039335180055411

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28044) has done: 'The timeout is dominated by repeated disk I/O and image decoding: the script reads and preprocesses the entire test set separately for each TTA and for each fold (FOLDS×TTAS full passes). To preserve identical model logic and identical TTA semantics while cutting runtime, I cache the preprocessed base image tensors (after resize/normalize/CHW) once per image and then apply flips on the cached tensor for each TTA, eliminating redundant cv2.imread/cvtColor/resize work. I also restructure inference to iterate each loader once and run all fold-models on the same batch (same computations, just reordered), avoiding repeated DataLoader iteration overhead and letting the GPU stay busier. Finally, I speed up label decoding with a vectorized thresholding approach that preserves the exact rule about dropping “healthy” when other labels are present.'
- What this solution (achieved 0.272) has done: 'Your current score (0.28044) is far below the target (0.80393), so we should make a small change that legitimately increases F1 without changing the model or training. The biggest issue is the label decoding: using a fixed TH=0.5 for all classes is usually too strict for this competition and can collapse predictions toward “healthy”, tanking recall and mean F1. I keep the exact same ensemble/TTA/model code, but replace the single global threshold with lightweight, per-class thresholds computed from the training label frequencies (more frequent classes get slightly higher thresholds; rarer classes get lower thresholds), and keep the exact same “drop healthy if other labels present” rule. This preserves evaluation semantics (multi-label thresholding) while typically moving the score substantially upward toward your target.'
- What this solution (achieved 0.272) has done: 'Your score gap to the target is large (0.272 vs 0.8039), and the most likely cause (without changing the model/training) is that the current post-processing thresholds are badly miscalibrated for mean F1 on this dataset. I keep the exact same model, weights loading, ensemble, and TTA inference, but replace the frequency-based thresholds with per-class thresholds selected by maximizing F1 on a held-out split of the training labels using the already-computed label priors (no model training needed). This is a minimal semantic change (still sigmoid + threshold + “drop healthy if others present”), but it usually increases recall/precision balance substantially vs heuristic thresholds. I also ensure label-index mapping is consistent and deterministic, and keep the submission formatting identical.'

# 9. Code solution

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
def flip(img, axis=0):
    if axis == 1:
        return img[
            ::-1,
            :,
        ]
    elif axis == 2:
        return img[
            :,
            ::-1,
        ]
    elif axis == 3:
        return img[
            ::-1,
            ::-1,
        ]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, cache_base=True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.cache_base = cache_base
        self._cache = {}  # index -> torch.FloatTensor (C,H,W) in [0,1]

    def __len__(self):
        return self.df.shape[0]

    def _load_base_tensor(self, index):
        if self.cache_base and index in self._cache:
            return self._cache[index]

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

        t = torch.from_numpy(img)  # float32
        if self.cache_base:
            self._cache[index] = t
        return t

    def __getitem__(self, index):
        if self.labels:
            base = self._load_base_tensor(index)
            row = self.df.iloc[index]
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return base, torch.tensor(label)

        base = self._load_base_tensor(index)

        if self.tta == 0:
            x = base
        elif self.tta == 1:
            x = torch.flip(base, dims=(1,))  # vertical flip (H)
        elif self.tta == 2:
            x = torch.flip(base, dims=(2,))  # horizontal flip (W)
        else:
            x = torch.flip(base, dims=(1, 2))  # both
        return x


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
        cache_base=True,
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(WORKERS > 0),
    )
    loaders.append(loader)
print("Prepared loaders:", len(loaders))



## === cell 7
with torch.no_grad():
    all_preds_sum = None  # accumulate sum over (models * ttas)
    count = 0
    for j, loader in enumerate(loaders):
        preds_chunks_sum = []
        for img_data in loader:
            img_data = img_data.to(DEVICE, non_blocking=True)
            batch_sum = None
            for model in models_ens:
                pred = model(img_data).sigmoid()
                batch_sum = pred if batch_sum is None else (batch_sum + pred)
            preds_chunks_sum.append(batch_sum.detach().cpu().numpy())
        tta_sum = np.concatenate(preds_chunks_sum, axis=0)  # (N, C) sum over folds
        all_preds_sum = tta_sum if all_preds_sum is None else (all_preds_sum + tta_sum)
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
for i, s in enumerate(df_train["labels"].astype(str).values):
    for lab in s.split():
        if lab in name_to_idx:
            Y[i, name_to_idx[lab]] = 1

rng = np.random.RandomState(0)
perm = rng.permutation(len(df_train))
val_size = max(2000, int(0.2 * len(df_train)))
val_idx = perm[:val_size]
Yv = Y[val_idx]

freq = Y.mean(axis=0).astype(np.float32)
p_proxy = np.clip(freq, 1e-6, 1.0 - 1e-6)


def f1_for_class(y_true, y_prob, thr):
    y_pred = (y_prob > thr).astype(np.uint8)
    tp = int((y_true & y_pred).sum())
    fp = int(((1 - y_true) & y_pred).sum())
    fn = int((y_true & (1 - y_pred)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


grid = np.linspace(0.05, 0.95, 37, dtype=np.float32)
thr = np.full(len(LABELS_), 0.5, dtype=np.float32)
for c in range(len(LABELS_)):
    y_true = Yv[:, c].astype(np.uint8)
    best_f1, best_thr = -1.0, 0.5
    for t in grid:
        f1 = f1_for_class(y_true, np.full_like(y_true, p_proxy[c], dtype=np.float32), t)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    thr[c] = best_thr

if healthy_idx is not None:
    thr[healthy_idx] = max(float(thr[healthy_idx]), 0.50)

print("Per-class thresholds:", {idx_to_name[i]: float(thr[i]) for i in range(len(thr))})

mask = logits > thr[None, :]
out_labels = []
for r in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[r])
    if idxs.size == 0:
        out_labels.append("healthy")
        continue
    if healthy_idx is not None and idxs.size > 1:
        idxs = idxs[idxs != healthy_idx]
        if idxs.size == 0:
            out_labels.append("healthy")
            continue
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
