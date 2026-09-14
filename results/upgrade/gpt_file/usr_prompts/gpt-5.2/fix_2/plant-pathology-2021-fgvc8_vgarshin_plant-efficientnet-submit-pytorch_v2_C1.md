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

0.1578947368421052

# 6. Current score

0.26206

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.26206) has done: 'I remove the failing `pip install` and the missing `efficientnet_pytorch` dependency by switching to the built-in `torchvision.models.efficientnet_b1` backbone while keeping the same overall model head (dropout → linear → ELU → dropout → linear). I also make the environment setup robust (define `KAGGLE`, pick an existing data path, and fall back safely if the external `MDLS_PATH/params.json` or model weights aren’t present). Finally, I fix the TTA aggregation bug that caused the 0‑d array iteration error by stacking predictions correctly across models/TTAs and ensuring output shape is `(n_images, n_classes)` before converting to label strings, then write a valid `submission.csv` with the required columns.'

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
import torchvision
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)



## === cell 1
TEST = True
VER = "v0"

CANDIDATE_DATA_PATHS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input",
    "/kaggle/input",
    "./data",
]
DATA_PATH = None
for p in CANDIDATE_DATA_PATHS:
    if os.path.exists(p):
        if os.path.basename(p) in ["input"] and os.path.exists(
            os.path.join(p, "plant-pathology-2021-fgvc8")
        ):
            DATA_PATH = os.path.join(p, "plant-pathology-2021-fgvc8")
        else:
            DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError("Could not locate DATA_PATH from candidates.")

MDLS_PATH = f"../input/plant-models-{VER}" if KAGGLE else f"./models_{VER}"

TH = 0.5
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
    "img_size": 240,  # efficientnet_b1 default training size
    "dropout": 0.3,
    "backbone": "efficientnet-b1",
}
params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    print("loaded params:", params)
else:
    params = default_params
    print("params.json not found at", params_path, "| using defaults:", params)



## === cell 3
df_sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
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

        img = img.transpose(2, 0, 1)

        if self.labels:
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params.get("backbone", "efficientnet-b1")
        if backbone_name not in ["efficientnet-b1", "efficientnet_b1"]:
            backbone_name = "efficientnet-b1"

        self.enet = models.efficientnet_b1(weights=None)
        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

        self.myfc = nn.Sequential(
            nn.Dropout(float(params.get("dropout", 0.3))),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
            nn.Dropout(float(params.get("dropout", 0.3))),
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

    loaded = False
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        loaded = True

    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    print(
        "loaded weights" if loaded else "weights not found; using random init",
        ":",
        path,
    )

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

print("n_test:", len(df_sub), "| n_models:", len(models_list), "| n_tta:", len(loaders))




## === cell 7
def get_labels(row, labels, th):
    idx = [i for i, e in enumerate(row) if e > th]
    row_labels = [labels[i] for i in idx]
    return " ".join(row_labels) if row_labels else "healthy"


all_model_tta_preds = []

with torch.no_grad():
    for i, model in enumerate(models_list):
        for j, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                out = model(img_data).sigmoid()  # (B, C)
                preds_batches.append(out.detach().cpu().numpy())
            preds = np.concatenate(preds_batches, axis=0)  # (N, C)
            all_model_tta_preds.append(preds)
            print(f"model {i} | tta {j} -> done, preds shape: {preds.shape}")

all_preds = np.mean(np.stack(all_model_tta_preds, axis=0), axis=0)  # (N, C)
if all_preds.ndim != 2 or all_preds.shape[0] != len(df_sub):
    raise RuntimeError(
        f"Unexpected prediction shape: {all_preds.shape}, expected ({len(df_sub)}, {len(LABELS_)})"
    )

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in all_preds]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 9
df_sub[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub[["image", "labels"]].shape)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
