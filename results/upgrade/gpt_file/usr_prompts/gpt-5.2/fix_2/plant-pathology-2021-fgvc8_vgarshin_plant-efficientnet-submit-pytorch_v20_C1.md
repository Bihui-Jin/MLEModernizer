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

0.7925577100646358

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
import torchvision

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

TEST = True
VER = "v0"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-efficientnet-train-pytorch/models_{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)



## === cell 1
pass



## === cell 2
with open(f"{MDLS_PATH}/params.json") as file:
    params = json.load(file)
LABELS_ = params["labels_"]  # list of label names (in model output order)
LABELS = params[
    "labels"
]  # mapping label->index or index->label depending on training code
WORKERS = 2 if KAGGLE else params.get("workers", 2)
print("loaded params:", params)

with open(f"{MDLS_PATH}/ths.json") as file:
    ths = json.load(file)
print("thresholds loaded; n_ths =", len(ths))

if isinstance(LABELS, dict) and ("0" not in LABELS) and (0 in LABELS):
    LABELS = {str(k): v for k, v in LABELS.items()}
elif (
    isinstance(LABELS, dict)
    and ("0" not in LABELS)
    and all(isinstance(k, str) for k in LABELS.keys())
):
    vals = list(LABELS.values())
    if len(vals) and all(isinstance(v, int) for v in vals):
        inv = {str(v): k for k, v in LABELS.items()}
        LABELS = inv

ths = {str(k): float(v) for k, v in ths.items()}



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1573802475.py in <cell line: 0>()
      1 # Load model params and thresholds (same files/paths as original).
----> 2 with open(f"{MDLS_PATH}/params.json") as file:
      3     params = json.load(file)
      4 LABELS_ = params["labels_"]  # list of label names (in model output order)
      5 LABELS = params[

FileNotFoundError: [Errno 2] No such file or directory: '../input/plant-efficientnet-train-pytorch/models_v0/params.json'

## === cell 3
df_sub = pd.DataFrame(sorted(os.listdir(IMGS_PATH)))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"
print("n_test_images:", len(df_sub))
df_sub.head()




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
            raise FileNotFoundError(f"Could not read image: {img_path}")
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




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params["backbone"]

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
            raise ValueError(
                f"Unsupported backbone '{backbone}' without efficientnet_pytorch. "
                f"Supported: {sorted(tv_map.keys())} or use params['backbone']=='resnext'."
            )

        self.enet = tv_map[backbone](weights=None)

        if hasattr(self.enet, "classifier") and isinstance(
            self.enet.classifier, nn.Sequential
        ):
            nc = None
            for m in reversed(self.enet.classifier):
                if isinstance(m, nn.Linear):
                    nc = m.in_features
                    break
            if nc is None:
                raise RuntimeError(
                    "Could not infer EfficientNet feature dim from torchvision classifier."
                )
            self.enet.classifier = nn.Identity()
        else:
            raise RuntimeError(
                "Unexpected torchvision EfficientNet structure; no .classifier Sequential found."
            )

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
        if torch.cuda.device_count() > 1:
            self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 6
models_list = []
for n_fold in FOLDS:
    if params["backbone"] == "resnext":
        model = ResNext(params=params, out_dim=len(LABELS_))
    else:
        model = EffNet(params=params, out_dim=len(LABELS_))

    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
    state_dict = torch.load(path, map_location="cpu")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=True)

    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    print("loaded:", path)

del state_dict, model
gc.collect()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2845590938.py in <cell line: 0>()
      2 models_list = []
      3 for n_fold in FOLDS:
----> 4     if params["backbone"] == "resnext":
      5         model = ResNext(params=params, out_dim=len(LABELS_))
      6     else:

NameError: name 'params' is not defined

## === cell 7
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    loaders.append(loader)

print("n_loaders:", len(loaders), "TTAS:", TTAS)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2993238568.py in <cell line: 0>()
      3 for tta in TTAS:
      4     dataset = PlantDataset(
----> 5         df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
      6     )
      7     datasets.append(dataset)

NameError: name 'params' is not defined

## === cell 8
def get_labels(row, labels, ths):
    idxs = [i for i, x in enumerate(row) if x > ths.get(str(i), 0.5)]
    labs = [labels.get(str(i), str(i)) for i in idxs]
    out = "healthy" if ("healthy" in labs or len(labs) == 0) else " ".join(labs)
    return out


n_imgs = len(df_sub)
n_classes = len(LABELS_)
all_preds = []

with torch.no_grad():
    for i, model in enumerate(models_list):
        for j, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                pred = (
                    model(img_data).sigmoid().detach().cpu().numpy()
                )  # (bs, n_classes)
                preds_batches.append(pred)
            preds = np.concatenate(preds_batches, axis=0)
            if preds.shape[0] != n_imgs or preds.shape[1] != n_classes:
                raise RuntimeError(
                    f"Pred shape mismatch: got {preds.shape}, expected ({n_imgs},{n_classes})"
                )
            all_preds.append(preds)
            print(f"model {i} | loader {j} -> done; preds={preds.shape}")

logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (n_imgs, n_classes)
df_sub["labels"] = [get_labels(x, LABELS, ths) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1136577160.py in <cell line: 0>()
     11 # Here we compute per-(model,tta) predictions as (n_images, n_classes), then average.
     12 n_imgs = len(df_sub)
---> 13 n_classes = len(LABELS_)
     14 all_preds = []
     15 

NameError: name 'LABELS_' is not defined

## === cell 9
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 10
df_sub = df_sub[["image", "labels"]].copy()
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)



## === cell 11
assert os.path.isfile("submission.csv")
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
