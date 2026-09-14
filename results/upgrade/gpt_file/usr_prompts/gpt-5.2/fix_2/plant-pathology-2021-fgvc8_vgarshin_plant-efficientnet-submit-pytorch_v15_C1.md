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

0.8245798707294568

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

try:
    display  # noqa: F821
except NameError:

    def display(x):
        print(x)




## === cell 1
pass



## === cell 2
import torchvision
from torchvision import models

KAGGLE = True
if not KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("device:", DEVICE)



## === cell 3
TEST = True
VER = "v6"

_CANDIDATES = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
DATA_PATH = None
for p in _CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "sample_submission.csv")):
        DATA_PATH = p
        break
if DATA_PATH is None:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"

MDLS_PATH = f"../input/plant-models-{VER}"

TTAS = [0, 1]
FOLDS = [0, 1, 2]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 4
params_path = f"{MDLS_PATH}/params.json"
if not os.path.isfile(params_path):
    raise FileNotFoundError(
        f"Missing {params_path}. This notebook expects pretrained models + params in {MDLS_PATH}."
    )

with open(params_path) as file:
    params = json.load(file)

LABELS_ = params["labels_"]  # list-like (model output order)
LABELS = params["labels"]  # dict-like index->label strings
WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
print("loaded params keys:", sorted(list(params.keys())))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1931113152.py in <cell line: 0>()
      2 params_path = f"{MDLS_PATH}/params.json"
      3 if not os.path.isfile(params_path):
----> 4     raise FileNotFoundError(
      5         f"Missing {params_path}. This notebook expects pretrained models + params in {MDLS_PATH}."
      6     )

FileNotFoundError: Missing ../input/plant-models-v6/params.json. This notebook expects pretrained models + params in ../input/plant-models-v6.

## === cell 5
if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")

df_sub = pd.DataFrame(sorted(os.listdir(IMGS_PATH)))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"
display(df_sub.head())
print("n_test_images:", len(df_sub))




## === cell 6
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
            raise FileNotFoundError(f"Image read failed: {img_path}")
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
    """
    Fix: replace efficientnet_pytorch dependency with torchvision EfficientNet.
    Core logic preserved: EfficientNet backbone -> replace classifier with Identity -> custom FC head.
    """

    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params["backbone"]

        name_map = {
            "efficientnet-b0": "efficientnet_b0",
            "efficientnet-b1": "efficientnet_b1",
            "efficientnet-b2": "efficientnet_b2",
            "efficientnet-b3": "efficientnet_b3",
            "efficientnet-b4": "efficientnet_b4",
            "efficientnet-b5": "efficientnet_b5",
            "efficientnet-b6": "efficientnet_b6",
            "efficientnet-b7": "efficientnet_b7",
        }
        tv_name = name_map.get(backbone, backbone)

        if not hasattr(models, tv_name):
            raise ValueError(
                f"Unknown backbone '{backbone}' (mapped to '{tv_name}'). "
                f"Available torchvision: {[n for n in dir(models) if 'efficientnet' in n]}"
            )

        self.enet = getattr(models, tv_name)(weights=None)

        if not hasattr(self.enet, "classifier") or not isinstance(
            self.enet.classifier, nn.Sequential
        ):
            raise RuntimeError(
                "Unexpected EfficientNet structure in torchvision version."
            )

        last_linear = self.enet.classifier[-1]
        if not isinstance(last_linear, nn.Linear):
            raise RuntimeError(
                "Unexpected classifier last layer type; expected nn.Linear."
            )
        nc = last_linear.in_features

        self.enet.classifier = nn.Identity()

        drop = float(params["dropout"])
        self.myfc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(drop),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 7
models_ens = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model weights: {path}")
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.float()
    model.eval()
    model.to(DEVICE)
    models_ens.append(model)
    print("loaded:", path)

del state_dict, model
gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1200911779.py in <cell line: 0>()
      1 models_ens = []
      2 for n_fold in FOLDS:
----> 3     model = EffNet(params, out_dim=len(LABELS_))
      4     path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
      5     if not os.path.isfile(path):

NameError: name 'params' is not defined

## === cell 8
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
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=(DEVICE.type == "cuda"),
    )
    loaders.append(loader)

print("n_loaders:", len(loaders))




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1003656698.py in <cell line: 0>()
      3     dataset = PlantDataset(
      4         df=df_sub,
----> 5         size=params["img_size"],
      6         labels=None,
      7         transform=None,

NameError: name 'params' is not defined

## === cell 9
def get_labels(row, labels, ths):
    idxs = [i for i, x in enumerate(row) if x > ths[str(i)]]
    row_labels = [labels[str(i)] for i in idxs]
    row_out = (
        "healthy"
        if ("healthy" in row_labels or len(row_labels) == 0)
        else " ".join(row_labels)
    )
    return row_out


all_preds = []  # each element: (N, C)
with torch.no_grad():
    for i, model in enumerate(models_ens):
        for j, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                p = model(img_data).sigmoid().detach().cpu().numpy()
                if p.ndim == 1:
                    p = p[None, :]
                preds_batches.append(p)
            preds_full = np.vstack(preds_batches)  # (N, C)
            all_preds.append(preds_full)
            print(f"model {i} | loader {j} -> done, shape={preds_full.shape}")

probs = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)

df_sub["labels"] = [get_labels(x, LABELS, params["ths"]) for x in probs]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2924162352.py in <cell line: 0>()
     29             print(f"model {i} | loader {j} -> done, shape={preds_full.shape}")
     30 
---> 31 probs = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)
     32 
     33 df_sub["labels"] = [get_labels(x, LABELS, params["ths"]) for x in probs]

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 10
print("value counts:")
print(df_sub.labels.value_counts().head(20))
display(df_sub.head())



## === cell 11
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("wrote:", sub_path, "rows:", len(df_sub))
print(df_sub.dtypes)

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
