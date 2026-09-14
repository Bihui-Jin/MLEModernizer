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



## === cell 1
TEST = True
VER = "v0"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.999

VOTERS = 1
TTAS = [0, 1, 2]
FOLDS = [0, 1]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

LABELS_ = {
    "complex": 0,
    "frog_eye_leaf_spot": 1,
    "healthy": 2,
    "powdery_mildew": 3,
    "rust": 4,
    "scab": 5,
}
LABELS = {v: k for k, v in LABELS_.items()}

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 2
default_params = {
    "img_size": 240,  # efficientnet_b1 default input resolution is 240
    "dropout": 0.3,
    "backbone": "efficientnet-b1",
}

params_path = f"{MDLS_PATH}/params.json"
if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    print("loaded params:", params)
else:
    params = default_params
    print(f"params.json not found at {params_path}; using default params:", params)

for k, v in default_params.items():
    params.setdefault(k, v)



## === cell 3
df_sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
print(df_sub.head())
print("submission rows:", len(df_sub))




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
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.transpose(2, 0, 1)

        if self.labels:
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params.get("backbone", "efficientnet-b1")
        if backbone not in ["efficientnet-b1", "efficientnet_b1"]:
            print(
                f"Warning: requested backbone {backbone}; falling back to efficientnet_b1."
            )
        self.enet = models.efficientnet_b1(weights=None)

        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
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
models_list = []
params["backbone"] = "efficientnet-b1"

for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Strict load failed for {path}: {e}\nTrying strict=False.")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("missing keys:", len(missing), "unexpected keys:", len(unexpected))
        print("loaded:", path)
        del state_dict
    else:
        print("weights not found (will use random init):", path)

    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)

gc.collect()



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        sampler=SequentialSampler(dataset),
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    loaders.append(loader)

print("n_models:", len(models_list), "n_ttas:", len(loaders), "n_test:", len(df_sub))




## === cell 7
def get_labels(row, labels, th):
    idx = [i for i, e in enumerate(row) if e > th]
    row_lbls = [labels[i] for i in idx]
    if len(row_lbls) == 0:
        return "healthy"
    return " ".join(row_lbls)


all_probs = []

with torch.no_grad():
    for i, model in enumerate(models_list):
        for j, loader in enumerate(loaders):
            probs_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                preds = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, 6)
                probs_batches.append(preds)
            probs = np.concatenate(probs_batches, axis=0)  # (N, 6)
            all_probs.append(probs)
            print(f"model {i} | loader {j} -> probs shape {probs.shape}")

all_probs = np.stack(all_probs, axis=0)  # (M*T, N, 6)
mean_probs = all_probs.mean(axis=0)  # (N, 6)

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in mean_probs]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 9
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
