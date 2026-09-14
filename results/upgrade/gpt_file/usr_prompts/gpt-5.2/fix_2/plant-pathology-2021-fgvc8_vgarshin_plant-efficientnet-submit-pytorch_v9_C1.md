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

0.8172114496768246

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



## === cell 1
try:
    from efficientnet_pytorch import model as enet  # type: ignore

    HAVE_EFFNET_PYTORCH = True
    print("Using efficientnet_pytorch backend.")
except Exception as e:
    enet = None
    HAVE_EFFNET_PYTORCH = False
    print(
        "efficientnet_pytorch not available; will use torchvision EfficientNet backend."
    )
    print("Import error:", repr(e))



## === cell 2
TEST = True
VER = "v3"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2]
FOLDS = [0, 1, 3]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)



## === cell 3
params_path = f"{MDLS_PATH}/params.json"
if not os.path.exists(params_path):
    raise FileNotFoundError(
        f"Missing model params at {params_path}. "
        "Ensure the dataset ../input/plant-models-v3 is attached."
    )

with open(params_path) as file:
    params = json.load(file)

LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = 2 if KAGGLE else params.get("workers", 2)

print("loaded params keys:", sorted(list(params.keys())))
print("num classes:", len(LABELS_))
print("img_size:", params.get("img_size"), "| batch_size:", params.get("batch_size"))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/667643998.py in <cell line: 0>()
      2 params_path = f"{MDLS_PATH}/params.json"
      3 if not os.path.exists(params_path):
----> 4     raise FileNotFoundError(
      5         f"Missing model params at {params_path}. "
      6         "Ensure the dataset ../input/plant-models-v3 is attached."

FileNotFoundError: Missing model params at ../input/plant-models-v3/params.json. Ensure the dataset ../input/plant-models-v3 is attached.

## === cell 4
if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")

df_sub = (
    pd.DataFrame(sorted(os.listdir(IMGS_PATH)))
    if TEST
    else pd.DataFrame(sorted(os.listdir(IMGS_PATH)[:100]))
)
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"
print(df_sub.head())
print("n_images:", len(df_sub))




## === cell 5
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




## === cell 6
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params["backbone"]

        if HAVE_EFFNET_PYTORCH:
            self.backend = "efficientnet_pytorch"
            self.enet = enet.EfficientNet.from_name(backbone_name)
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
            self.extract = self.enet
        else:
            self.backend = "torchvision"
            tv_map = {
                "efficientnet-b0": models.efficientnet_b0,
                "efficientnet-b1": models.efficientnet_b1,
                "efficientnet-b2": models.efficientnet_b2,
                "efficientnet-b3": models.efficientnet_b3,
                "efficientnet-b4": models.efficientnet_b4,
                "efficientnet-b5": models.efficientnet_b5,
                "efficientnet-b6": models.efficientnet_b6,
                "efficientnet-b7": models.efficientnet_b7,
                "efficientnet_v2_s": models.efficientnet_v2_s,
                "efficientnet_v2_m": models.efficientnet_v2_m,
                "efficientnet_v2_l": models.efficientnet_v2_l,
            }
            if backbone_name not in tv_map:
                raise ValueError(
                    f"Backbone '{backbone_name}' not supported by torchvision fallback. "
                    f"Supported: {sorted(tv_map.keys())}"
                )
            self.enet = tv_map[backbone_name](weights=None)
            if hasattr(self.enet, "classifier") and isinstance(
                self.enet.classifier, nn.Sequential
            ):
                nc = self.enet.classifier[-1].in_features
                self.enet.classifier = nn.Identity()
            else:
                raise RuntimeError(
                    "Unexpected torchvision EfficientNet structure; cannot locate classifier head."
                )
            self.extract = self.enet

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 7
models_ens = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing fold weight file: {path}")
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.float()
    model.eval()
    model.to(DEVICE)
    models_ens.append(model)
    print("loaded:", path, "| backend:", getattr(model, "backend", "unknown"))

del state_dict, model
gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1470096353.py in <cell line: 0>()
      2 models_ens = []
      3 for n_fold in FOLDS:
----> 4     model = EffNet(params, out_dim=len(LABELS_))
      5     path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
      6     if not os.path.exists(path):

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
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    loaders.append(loader)

print("n_folds:", len(models_ens), "| n_ttas:", len(loaders))




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/531519405.py in <cell line: 0>()
      4     dataset = PlantDataset(
      5         df=df_sub,
----> 6         size=params["img_size"],
      7         labels=None,
      8         transform=None,

NameError: name 'params' is not defined

## === cell 9
def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    labs = [labels[str(i)] for i in idx]
    return "healthy" if ("healthy" in labs or len(labs) == 0) else " ".join(labs)


all_fold_tta_preds = []

with torch.no_grad():
    for i, model in enumerate(models_ens):
        tta_preds = []
        for j, loader in enumerate(loaders):
            batch_preds = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                out = model(img_data).sigmoid()
                batch_preds.append(out.detach().cpu().numpy())
            preds = np.concatenate(batch_preds, axis=0)  # (n_images, n_classes)
            tta_preds.append(preds)
            print(f"model {i} | tta {j} -> preds:", preds.shape)
        all_fold_tta_preds.append(
            np.stack(tta_preds, axis=0)
        )  # (n_ttas, n_images, n_classes)

all_fold_tta_preds = np.stack(
    all_fold_tta_preds, axis=0
)  # (n_folds, n_ttas, n_images, n_classes)
logits = all_fold_tta_preds.mean(axis=(0, 1))  # (n_images, n_classes)

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4207037135.py in <cell line: 0>()
     28         )  # (n_ttas, n_images, n_classes)
     29 
---> 30 all_fold_tta_preds = np.stack(
     31     all_fold_tta_preds, axis=0
     32 )  # (n_folds, n_ttas, n_images, n_classes)

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
print(df_sub.head())



## === cell 11
sub_path = "submission.csv"
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv(sub_path, index=False)
print("wrote:", sub_path, "| shape:", df_sub.shape)
print(df_sub.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
