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

0.7627146814404437

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I remove the broken `pip install` step and make `efficientnet_pytorch` optional by switching to a torchvision EfficientNet implementation when it’s not available, keeping the same model head and inference semantics. I also harden the environment setup so `KAGGLE`, paths, and `DEVICE` are always defined, even if CUDA is unavailable, and ensure the code can run without external model packs by falling back to a safe “healthy” submission if weights/params aren’t found. Finally, I fix the TTA/model ensembling aggregation bug that produced `numpy.float64 object is not iterable`, replacing it with a correct average over models and TTAs and writing a valid `submission.csv` with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler

warnings.filterwarnings("ignore")

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
pass



## === cell 2
try:
    from efficientnet_pytorch import model as enet  # original dependency

    HAS_EFFNET_PYTORCH = True
    print("Using efficientnet_pytorch")
except Exception as e:
    HAS_EFFNET_PYTORCH = False
    enet = None
    print(
        "efficientnet_pytorch not available; will use torchvision EfficientNet fallback.",
        repr(e),
    )



## === cell 3
TEST = True
VER = "v103"

if KAGGLE:
    if Path("../input/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    elif Path("/kaggle/input/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
    elif Path("/kaggle/data/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "/kaggle/data/plant-pathology-2021-fgvc8"
    else:
        DATA_PATH = "../input/plant-pathology-2021-fgvc8"

    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 4
params_path = Path(MDLS_PATH) / "params.json"
ths_path = Path(MDLS_PATH) / "ths.json"

if params_path.exists():
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    WORKERS = 2 if KAGGLE else params.get("workers", 2)
    print("loaded params:", params)
else:
    sample_path = Path(DATA_PATH) / "sample_submission.csv"
    if not sample_path.exists():
        if Path("/kaggle/data/sample_submission.csv").exists():
            sample_path = Path("/kaggle/data/sample_submission.csv")
        elif Path(
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
        ).exists():
            sample_path = Path(
                "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
            )
    df_sample = pd.read_csv(sample_path)
    LABELS_, LABELS = [], {}
    params = {
        "img_size": 512,
        "batch_size": 8,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
    }
    WORKERS = 2
    print("WARNING: params.json not found; using fallback params:", params)

if ths_path.exists():
    with open(ths_path) as file:
        ths = json.load(file)
    print("thresholds loaded:", list(ths.items())[:3], "...")
else:
    ths = {}
    print(
        "WARNING: ths.json not found; thresholds dict is empty (will default to healthy)."
    )



## === cell 5
sample_path = Path(DATA_PATH) / "sample_submission.csv"
if not sample_path.exists():
    if Path("/kaggle/data/sample_submission.csv").exists():
        sample_path = Path("/kaggle/data/sample_submission.csv")
    elif Path(
        "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    ).exists():
        sample_path = Path(
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
        )

df_sub = pd.read_csv(sample_path)[["image", "labels"]].copy()
df_sub["labels"] = "healthy"
print(df_sub.head())




## === cell 6
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

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params["backbone"]

        if HAS_EFFNET_PYTORCH:
            self.enet = enet.EfficientNet.from_name(backbone)
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
            self._extract = "effnet_pytorch"
        else:
            tv_map = {
                "efficientnet-b0": torchvision.models.efficientnet_b0,
                "efficientnet-b1": torchvision.models.efficientnet_b1,
                "efficientnet-b2": torchvision.models.efficientnet_b2,
                "efficientnet-b3": torchvision.models.efficientnet_b3,
                "efficientnet-b4": torchvision.models.efficientnet_b4,
                "efficientnet-b5": torchvision.models.efficientnet_b5,
                "efficientnet-b6": torchvision.models.efficientnet_b6,
                "efficientnet-b7": torchvision.models.efficientnet_b7,
            }
            if backbone not in tv_map:
                backbone = "efficientnet-b0"
            self.enet = tv_map[backbone](weights=None)
            nc = self.enet.classifier[-1].in_features
            self.enet.classifier = nn.Identity()
            self._extract = "torchvision"

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(weights=None)
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




## === cell 7
models_list = []
loaded_any = False

for n_fold in FOLDS:
    if len(LABELS_) == 0:
        break

    if params["backbone"] == "resnext":
        model = ResNext(params=params, out_dim=len(LABELS_))
    else:
        model = EffNet(params=params, out_dim=len(LABELS_))

    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if not Path(path).exists():
        print("WARNING: model weights not found:", path)
        continue

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    loaded_any = True
    print("loaded:", path)

del_state = locals().get("state_dict", None)
if del_state is not None:
    del state_dict
gc.collect()



## === cell 8
datasets, loaders = [], []
if loaded_any:
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
        )
        loaders.append(loader)
else:
    print("No models loaded; will write all-healthy submission.")




## === cell 9
def get_labels(probs_row, labels, ths):
    idxs = []
    for i, x in enumerate(probs_row):
        th = ths.get(str(i), 0.5)
        if x > th:
            idxs.append(i)
    row_labels = [labels.get(str(i), None) for i in idxs]
    row_labels = [x for x in row_labels if x is not None]
    return (
        "healthy"
        if ("healthy" in row_labels or len(row_labels) == 0)
        else " ".join(row_labels)
    )


if loaded_any:
    all_model_tta = []  # list of arrays [num_images, num_classes]
    with torch.no_grad():
        for mi, model in enumerate(models_list):
            for tj, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True)
                    preds = model(img_data).sigmoid().detach().cpu().numpy()
                    preds_batches.append(preds)
                preds_full = np.vstack(preds_batches)  # [N, C]
                all_model_tta.append(preds_full)
                print(f"model {mi} | tta {tj} -> done; shape={preds_full.shape}")

    logits = np.mean(np.stack(all_model_tta, axis=0), axis=0)  # [N, C]
    df_sub["labels"] = [get_labels(x, LABELS, ths) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 10
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
print(df_sub.head())



## === cell 11
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
