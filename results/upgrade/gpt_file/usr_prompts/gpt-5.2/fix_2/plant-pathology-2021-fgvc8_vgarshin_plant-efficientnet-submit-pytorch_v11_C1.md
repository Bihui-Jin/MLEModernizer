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

0.831154201292706

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I remove the failing offline `pip install` and instead use the already-installed `torchvision` EfficientNet as a drop-in backbone so the model class can be constructed and weights can be loaded. I also make the notebook robust to CPU-only environments and fix the inference aggregation bug that caused `np.vstack` to crash by correctly averaging over folds and TTAs with consistent array shapes. Finally, I ensure the test image path resolves correctly and that a valid `submission.csv` with the required `image,labels` columns is always written.'

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
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

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
pass



## === cell 2
TEST = True
VER = "v4"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2]
FOLDS = [3, 4]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

if not os.path.isdir(IMGS_PATH):
    alt1 = (
        f"../input/plant-pathology-2021-fgvc8/test_images"
        if TEST
        else f"../input/plant-pathology-2021-fgvc8/train_images"
    )
    alt2 = f"../input/test_images" if TEST else f"../input/train_images"
    for p in [alt1, alt2]:
        if os.path.isdir(p):
            IMGS_PATH = p
            break

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 3
params_path = f"{MDLS_PATH}/params.json"
if not os.path.isfile(params_path):
    raise FileNotFoundError(
        f"Missing params.json at {params_path}. Available in MDLS_PATH: {os.listdir(MDLS_PATH)[:20]}"
    )

with open(params_path) as file:
    params = json.load(file)

LABELS_ = params["labels_"]  # list-like of class names (order used during training)
LABELS = params["labels"]  # mapping index(str)->label used by get_labels below
WORKERS = 2 if KAGGLE else params.get("workers", 2)
print("loaded params:", params)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3144496130.py in <cell line: 0>()
      3 if not os.path.isfile(params_path):
      4     raise FileNotFoundError(
----> 5         f"Missing params.json at {params_path}. Available in MDLS_PATH: {os.listdir(MDLS_PATH)[:20]}"
      6     )
      7 

FileNotFoundError: [Errno 2] No such file or directory: '../input/plant-models-v4'

## === cell 4
test_files = sorted([f for f in os.listdir(IMGS_PATH) if f.lower().endswith(".jpg")])
if len(test_files) == 0:
    raise RuntimeError(f"No .jpg files found in IMGS_PATH={IMGS_PATH}")

df_sub = pd.DataFrame({"image": test_files})
df_sub["labels"] = "healthy"
print(df_sub.head())




## === cell 5
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
            raise FileNotFoundError(f"no img file read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.labels is None:
            img = flip(img, axis=self.tta)

        img = img.transpose(2, 0, 1)
        return torch.tensor(img.copy())




## === cell 6
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params.get("backbone", "efficientnet_b0")

        tv_backbones = {
            "efficientnet_b0": torchvision.models.efficientnet_b0,
            "efficientnet_b1": torchvision.models.efficientnet_b1,
            "efficientnet_b2": torchvision.models.efficientnet_b2,
            "efficientnet_b3": torchvision.models.efficientnet_b3,
            "efficientnet_b4": torchvision.models.efficientnet_b4,
            "efficientnet_b5": torchvision.models.efficientnet_b5,
            "efficientnet_b6": torchvision.models.efficientnet_b6,
            "efficientnet_b7": torchvision.models.efficientnet_b7,
        }
        if backbone_name not in tv_backbones:
            print(
                f"Warning: unknown backbone '{backbone_name}', falling back to efficientnet_b0"
            )
            backbone_name = "efficientnet_b0"

        self.enet = tv_backbones[backbone_name](weights=None)

        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

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




## === cell 7
models_list = []
missing = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if not os.path.isfile(path):
        missing.append(path)
        continue

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    print("loaded:", path)

if len(models_list) == 0:
    raise FileNotFoundError(
        f"No fold models were loaded. Missing examples: {missing[:5]}"
    )

del state_dict, model
gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1261025635.py in <cell line: 0>()
      3 missing = []
      4 for n_fold in FOLDS:
----> 5     model = EffNet(params, out_dim=len(LABELS_))
      6     path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
      7     if not os.path.isfile(path):

NameError: name 'params' is not defined

## === cell 8
loaders = []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    loaders.append(loader)

print(
    "n_models:", len(models_list), "| n_ttas:", len(loaders), "| n_test:", len(df_sub)
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4205860107.py in <cell line: 0>()
      3 for tta in TTAS:
      4     dataset = PlantDataset(
----> 5         df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
      6     )
      7     loader = torch.utils.data.DataLoader(

NameError: name 'params' is not defined

## === cell 9
def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    out = [labels[str(i)] for i in idx]
    out = "healthy" if ("healthy" in out or len(out) == 0) else " ".join(out)
    return out


all_preds = []
with torch.no_grad():
    for i, model in enumerate(models_list):
        for j, loader in enumerate(loaders):
            preds_batches = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                probs = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, C)
                preds_batches.append(probs)
            preds = np.vstack(preds_batches)  # (N, C)
            all_preds.append(preds)
            print(f"model {i} | tta {j} -> done, shape={preds.shape}")

logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)
df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3905759849.py in <cell line: 0>()
     22             print(f"model {i} | tta {j} -> done, shape={preds.shape}")
     23 
---> 24 logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)
     25 df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]
     26 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 10
print("value counts (top 20):")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 11
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(pd.read_csv(sub_path).head())
