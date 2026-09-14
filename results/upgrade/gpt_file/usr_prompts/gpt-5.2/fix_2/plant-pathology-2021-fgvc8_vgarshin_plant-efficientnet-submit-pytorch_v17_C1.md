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

0.8065189289012012

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

KAGGLE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
TEST = True
VER = "v101"


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_PATH = _first_existing(
    [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../data/plant-pathology-2021-fgvc8",
        "./data/plant-pathology-2021-fgvc8",
        "../input",
        "../data",
    ]
)

if DATA_PATH is None:
    raise FileNotFoundError("Could not find competition data directory.")

MDLS_PATH = _first_existing(
    [
        f"/kaggle/input/plant-models-{VER}",
        f"../input/plant-models-{VER}",
        f"../input/plant-models-{VER}/plant-models-{VER}",  # in case of nested
        f"./models_{VER}",
        f"/kaggle/working/models_{VER}",
    ]
)

if MDLS_PATH is None:
    raise FileNotFoundError(
        f"Could not find model directory for VER={VER}. "
        f"Expected something like ../input/plant-models-{VER} with params.json and checkpoints."
    )

TTAS = [0, 1, 2, 3]
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
if not os.path.isdir(IMGS_PATH):
    alt = (
        f"{DATA_PATH}/plant-pathology-2021-fgvc8/test_images"
        if TEST
        else f"{DATA_PATH}/plant-pathology-2021-fgvc8/train_images"
    )
    if os.path.isdir(alt):
        IMGS_PATH = alt
    else:
        raise FileNotFoundError(
            f"Could not find images folder at {IMGS_PATH} (or {alt})."
        )

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("DEVICE:", DEVICE)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/779589536.py in <cell line: 0>()
     39 
     40 if MDLS_PATH is None:
---> 41     raise FileNotFoundError(
     42         f"Could not find model directory for VER={VER}. "
     43         f"Expected something like ../input/plant-models-{VER} with params.json and checkpoints."

FileNotFoundError: Could not find model directory for VER=v101. Expected something like ../input/plant-models-v101 with params.json and checkpoints.

## === cell 2
params_path = os.path.join(MDLS_PATH, "params.json")
ths_path = os.path.join(MDLS_PATH, "ths.json")

with open(params_path, "r") as file:
    params = json.load(file)

LABELS_ = params["labels_"]  # list-like (index -> label)
LABELS = params["labels"]  # dict-like (label -> index)
WORKERS = 2 if KAGGLE else params.get("workers", 2)

print("loaded params keys:", list(params.keys()))
print("num classes:", len(LABELS_))

with open(ths_path, "r") as file:
    ths = json.load(file)
print("loaded thresholds:", len(ths))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2703983936.py in <cell line: 0>()
      1 # Load params/thresholds required for exact label mapping/calibration.
----> 2 params_path = os.path.join(MDLS_PATH, "params.json")
      3 ths_path = os.path.join(MDLS_PATH, "ths.json")
      4 
      5 with open(params_path, "r") as file:

/usr/lib/python3.11/posixpath.py in join(a, *p)

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 3
df_sub = pd.DataFrame(sorted(os.listdir(IMGS_PATH)))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"
print(df_sub.head())
print("test images:", len(df_sub))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3210230668.py in <cell line: 0>()
      1 # Build submission frame from test image directory (keeps original behavior).
----> 2 df_sub = pd.DataFrame(sorted(os.listdir(IMGS_PATH)))
      3 df_sub.columns = ["image"]
      4 df_sub["labels"] = "healthy"
      5 print(df_sub.head())

NameError: name 'IMGS_PATH' is not defined

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
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        raise RuntimeError(
            "EffNet is not used in this inference script (efficientnet_pytorch dependency removed)."
        )


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(pretrained=False)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )
        self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 5
models_list = []
for n_fold in FOLDS:
    model = ResNext(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    print("loaded:", path)

del state_dict, model
gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/134569224.py in <cell line: 0>()
      1 # Load model checkpoints.
      2 models_list = []
----> 3 for n_fold in FOLDS:
      4     model = ResNext(params, out_dim=len(LABELS_))
      5     path = f"{MDLS_PATH}/model_best_{n_fold}.pth"

NameError: name 'FOLDS' is not defined

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
        pin_memory=torch.cuda.is_available(),
    )
    loaders.append(loader)

print("Num TTAs:", len(loaders), "| Num models:", len(models_list))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3661084360.py in <cell line: 0>()
      1 # Build TTA loaders.
      2 datasets, loaders = [], []
----> 3 for tta in TTAS:
      4     dataset = PlantDataset(
      5         df=df_sub,

NameError: name 'TTAS' is not defined

## === cell 7
def get_labels(row, labels, ths):
    idxs = [i for i, x in enumerate(row) if x > ths[str(i)]]
    lbls = [labels[str(i)] for i in idxs]
    out = "healthy" if ("healthy" in lbls or len(lbls) == 0) else " ".join(lbls)
    return out


all_preds = []
with torch.no_grad():
    for mi, model in enumerate(models_list):
        for tj, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                probs = model(img_data).sigmoid().detach().cpu().numpy()
                if probs.ndim == 1:
                    probs = probs[None, :]
                preds_batches.append(probs)
            preds_tta = np.vstack(preds_batches)  # (N, C)
            all_preds.append(preds_tta)
            print(f"model {mi} | tta {tj} -> preds: {preds_tta.shape}")

all_preds = np.stack(all_preds, axis=0)
logits = all_preds.mean(axis=0)

df_sub["labels"] = [get_labels(x, LABELS_, ths) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2098252127.py in <cell line: 0>()
     26 
     27 # (M*T, N, C) -> (N, C)
---> 28 all_preds = np.stack(all_preds, axis=0)
     29 logits = all_preds.mean(axis=0)
     30 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 8
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1888263766.py in <cell line: 0>()
      1 print("value counts:")
----> 2 print(df_sub["labels"].value_counts().head(20))
      3 df_sub.head()
      4 

NameError: name 'df_sub' is not defined

## === cell 9
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(df_sub.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2979406471.py in <cell line: 0>()
      1 # Ensure correct submission format and .csv suffix.
      2 sub_path = "submission.csv"
----> 3 df_sub[["image", "labels"]].to_csv(sub_path, index=False)
      4 print("Wrote:", sub_path, "| rows:", len(df_sub))
      5 print(df_sub.head())

NameError: name 'df_sub' is not defined
