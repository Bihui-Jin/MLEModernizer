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

0.8034164358264089

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

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
from torch.utils.data.sampler import SequentialSampler
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True
TEST = True
VER = "v5"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
TTAS = [0, 1, 2, 3]
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

torch.backends.cudnn.benchmark = True




## === cell 1
print("Using torchvision EfficientNet backbone; no external pip installs required.")




## === cell 2
def build_params_from_train_csv(data_path):
    train_csv = os.path.join(data_path, "train.csv")
    df = pd.read_csv(train_csv)
    all_labels = sorted(
        {l for s in df["labels"].astype(str).tolist() for l in s.split()}
    )
    labels_ = all_labels  # index -> label
    labels = {lbl: i for i, lbl in enumerate(labels_)}  # label -> index
    return {
        "labels_": labels_,
        "labels": labels,
        "img_size": 512,  # used for resize in inference
        "batch_size": 16,
        "dropout": 0.2,
        "backbone": "efficientnet_b0",
        "workers": 2,
    }


params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as file:
        params = json.load(file)
    print("loaded params from:", params_path)
else:
    params = build_params_from_train_csv(DATA_PATH)
    print("WARNING: params.json not found at", params_path)
    print("Using fallback params derived from train.csv")

LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
print("num classes:", len(LABELS_))




## === cell 3
test_files = sorted(os.listdir(IMGS_PATH))
df_sub = pd.DataFrame({"image": test_files})
df_sub["labels"] = "healthy"
df_sub.head()




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params.get("backbone", "efficientnet_b0")
        if backbone != "efficientnet_b0":
            backbone = "efficientnet_b0"

        self.enet = models.efficientnet_b0(weights=None)
        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

        dp = float(params.get("dropout", 0.2))
        self.myfc = nn.Sequential(
            nn.Dropout(dp),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(dp),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
def discover_checkpoints(mdls_path):
    if not os.path.isdir(mdls_path):
        return []
    ckpts = []
    for fn in os.listdir(mdls_path):
        if fn.startswith("model_best_") and fn.endswith(".pth"):
            ckpts.append(os.path.join(mdls_path, fn))
    return sorted(ckpts)


models_list = []
all_missing = True

ckpt_paths = []
for n_fold in FOLDS:
    ckpt_paths.append(os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth"))
if not any(os.path.exists(p) for p in ckpt_paths):
    ckpt_paths = discover_checkpoints(MDLS_PATH)

if len(ckpt_paths) == 0:
    print("WARNING: no checkpoints found under:", MDLS_PATH)
else:
    print("Found checkpoints:", len(ckpt_paths))
    for p in ckpt_paths[:10]:
        print(" -", p)

for ckpt_path in ckpt_paths if len(ckpt_paths) > 0 else [None]:
    model = EffNet(params, out_dim=len(LABELS_))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        all_missing = False
        print("loaded:", ckpt_path)
    else:
        if ckpt_path is not None:
            print("WARNING: checkpoint missing:", ckpt_path)

    model.eval()
    model.to(DEVICE)
    models_list.append(model)

gc.collect()




## === cell 7
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

len(df_sub), len(loaders)




## === cell 8
def get_labels(row_probs, idx2lbl, th):
    idxs = [i for i, x in enumerate(row_probs) if x > th]
    lbls = [idx2lbl[i] for i in idxs]
    lbls = "healthy" if ("healthy" in lbls or len(lbls) == 0) else " ".join(lbls)
    return lbls


IDX2LBL = {i: lbl for i, lbl in enumerate(LABELS_)}

if all_missing:
    print("No checkpoints found; writing default 'healthy' submission.")
    df_sub["labels"] = "healthy"
else:
    probs_accum = None
    n_passes = 0

    with torch.no_grad():
        for i, model in enumerate(models_list):
            for j, loader in enumerate(loaders):
                probs_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True).float()
                    preds = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, C)
                    probs_batches.append(preds)
                probs = np.vstack(probs_batches)  # (N, C)
                if probs_accum is None:
                    probs_accum = probs
                else:
                    probs_accum += probs
                n_passes += 1
                print(f"model {i} | tta {j} -> done")

    probs_mean = probs_accum / float(n_passes)
    df_sub["labels"] = [get_labels(x, IDX2LBL, TH) for x in probs_mean]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")




## === cell 9
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
df_sub.head()




## === cell 10
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())




## === cell 11
assert os.path.exists("submission.csv"), "submission.csv was not created"
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image", "labels"], f"Bad columns: {sub.columns}"
assert sub.shape[0] == df_sub.shape[0], "Row count mismatch"
assert sub["image"].isna().sum() == 0, "Found NaN image ids"
assert sub["labels"].isna().sum() == 0, "Found NaN labels"
print("Submission file is valid.")
