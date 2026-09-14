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

## === cell 1
import os, sys

print("Skipping external pip install; using torchvision EfficientNet instead.")



## === cell 2
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
from torch.utils.data.sampler import SequentialSampler
import torchvision
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)



## === cell 3
TEST = True
VER = "v4"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.3
TTAS = [0, 1, 2]
FOLDS = [3, 4]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

if not os.path.isdir(IMGS_PATH):
    alt = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images"
    if TEST and os.path.isdir(alt):
        IMGS_PATH = alt
    else:
        alt2 = "../input/plant-pathology-2021-fgvc8/test_images"
        if os.path.isdir(alt2):
            IMGS_PATH = alt2

print("IMGS_PATH:", IMGS_PATH)



## === cell 4
default_params = {
    "backbone": "efficientnet_b0",
    "img_size": 512,
    "batch_size": 16,
    "dropout": 0.5,
    "workers": 2,
    "labels_": [
        "healthy",
        "scab",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
    ],
    "labels": {
        "0": "healthy",
        "1": "scab",
        "2": "frog_eye_leaf_spot",
        "3": "rust",
        "4": "complex",
        "5": "powdery_mildew",
    },
}

params_path = f"{MDLS_PATH}/params.json"
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)
else:
    params = default_params

LABELS_ = params.get("labels_", default_params["labels_"])
LABELS = params.get("labels", default_params["labels"])
WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
print("loaded params:", params)
print("num labels:", len(LABELS_))



## === cell 5
sub_path = f"{DATA_PATH}/sample_submission.csv"
if not os.path.isfile(sub_path):
    sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

df_sub = pd.read_csv(sub_path)
df_sub = df_sub[["image", "labels"]].copy()
df_sub["labels"] = "healthy"

existing = set([f for f in os.listdir(IMGS_PATH) if f.lower().endswith(".jpg")])
df_sub = df_sub[df_sub["image"].isin(existing)].reset_index(drop=True)

print("df_sub shape:", df_sub.shape)
print(df_sub.head())




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
        backbone = params.get("backbone", "efficientnet_b0")
        if backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            feat_dim = 1280
        elif backbone == "efficientnet_b1":
            base = models.efficientnet_b1(weights=None)
            feat_dim = 1280
        elif backbone == "efficientnet_b2":
            base = models.efficientnet_b2(weights=None)
            feat_dim = 1408
        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=None)
            feat_dim = 1536
        else:
            base = models.efficientnet_b0(weights=None)
            feat_dim = 1280

        self.enet = base
        self.enet.classifier = nn.Identity()

        drop = float(params.get("dropout", 0.5))
        self.myfc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, int(feat_dim / 4)),
            nn.Dropout(drop),
            nn.Linear(int(feat_dim / 4), out_dim),
        )

    def extract(self, x):
        x = self.enet.features(x)
        x = self.enet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 7
models_list = []
loaded_any = False

for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
    if os.path.isfile(path):
        state_dict = torch.load(path, map_location="cpu")
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_sd = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)
        loaded_any = True
        print("loaded:", path)
    else:
        print(
            "checkpoint missing, using untrained weights for fold:",
            n_fold,
            "| expected:",
            path,
        )

    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)

gc.collect()

if not models_list:
    model = EffNet(params, out_dim=len(LABELS_)).float().eval().to(DEVICE)
    models_list = [model]
print("num models:", len(models_list), "| loaded_any:", loaded_any)



## === cell 8
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    loaders.append(loader)

print("num loaders (TTA):", len(loaders))




## === cell 9
def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    out = [labels.get(str(i), labels.get(i, str(i))) for i in idx]
    out = "healthy" if ("healthy" in out or len(out) == 0) else " ".join(out)
    return out


num_imgs = len(df_sub)
num_classes = len(LABELS_)
probs_accum = np.zeros((num_imgs, num_classes), dtype=np.float32)

with torch.no_grad():
    for mi, model in enumerate(models_list):
        for ti, loader in enumerate(loaders):
            all_probs = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True).float()
                preds = model(img_data).sigmoid().detach().cpu().numpy()
                all_probs.append(preds)
            all_probs = np.vstack(all_probs)  # (N, C)
            probs_accum += all_probs.astype(np.float32)
            print(
                f"model {mi} | tta {ti} -> done, batch_probs shape: {all_probs.shape}"
            )

probs = probs_accum / float(len(models_list) * len(loaders))
df_sub["labels"] = [get_labels(p, LABELS, TH) for p in probs]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 10
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 11
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
