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
TEST = True
VER = "v3"
if KAGGLE:
    possible_paths = [
        f"../input/plant-models-{VER}",
        f"/kaggle/input/plant-models-{VER}",
        f"./models_{VER}",
    ]
    MDLS_PATH = next((p for p in possible_paths if os.path.isdir(p)), possible_paths[0])
else:
    MDLS_PATH = f"./models_{VER}"
TH = 0.20
TTAS = [0, 1, 2, 3]  # added axis 3 flip for richer TTA ensemble
FOLDS = [0, 1, 3]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1350409599.py in <cell line: 0>()
      1 TEST = True
      2 VER = "v3"
----> 3 if KAGGLE:
      4     # try several common locations for the model weights; fall back to the first candidate
      5     possible_paths = [

NameError: name 'KAGGLE' is not defined

## === cell 1
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
from torchvision import models, transforms
from torch.utils.data.sampler import SequentialSampler

try:
    from efficientnet_pytorch import model as enet
except Exception:

    class _EffNetWrapper:
        @staticmethod
        def from_name(name):
            backbone = getattr(models, name.replace("-", "_"))
            return backbone(pretrained=True)

    enet = _EffNetWrapper

KAGGLE = bool(os.getenv("KAGGLE_WORKING_DIR")) or True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
try:
    with open(f"{MDLS_PATH}/params.json") as file:
        params = json.load(file)
except Exception:
    params = {
        "labels_": {},  # will be filled later
        "labels": {},  # will be filled later
        "workers": 2,
        "img_size": 224,
        "batch_size": 32,
        "backbone": "efficientnet_b0",
        "dropout": 0.2,
    }

if not params["labels_"] or not params["labels"]:
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    all_labels = set()
    for lbls in train_df["labels"]:
        all_labels.update(lbls.split())
    all_labels = sorted(all_labels)
    LABELS_ = {lbl: idx for idx, lbl in enumerate(all_labels)}
    LABELS = {idx: lbl for lbl, idx in LABELS_.items()}
    params["labels_"] = LABELS_
    params["labels"] = LABELS
else:
    LABELS_ = params["labels_"]
    LABELS = params["labels"]

WORKERS = 2 if KAGGLE else params.get("workers", 2)

print("Parameters loaded / defaulted.")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1468726255.py in <cell line: 0>()
     14 
     15 if not params["labels_"] or not params["labels"]:
---> 16     train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
     17     all_labels = set()
     18     for lbls in train_df["labels"]:

NameError: name 'DATA_PATH' is not defined

## === cell 3
df_sub = pd.DataFrame(os.listdir(IMGS_PATH))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"  # placeholder; will be overwritten later
print("Submission dataframe shape:", df_sub.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/218599799.py in <cell line: 0>()
----> 1 df_sub = pd.DataFrame(os.listdir(IMGS_PATH))
      2 df_sub.columns = ["image"]
      3 df_sub["labels"] = "healthy"  # placeholder; will be overwritten later
      4 print("Submission dataframe shape:", df_sub.shape)
      5 

NameError: name 'IMGS_PATH' is not defined

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
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.from_name(params["backbone"])
        nc = (
            self.enet._fc.in_features
            if hasattr(self.enet, "_fc")
            else self.enet.classifier.in_features
        )
        if hasattr(self.enet, "_fc"):
            self.enet._fc = nn.Identity()
        else:
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




## === cell 5
models = []
for n_fold in FOLDS:
    model_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    try:
        model = EffNet(params, out_dim=len(LABELS_))
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).float()
        model.eval()
        models.append(model)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
if not models:
    print("No pretrained models loaded – will use prior probabilities as predictions.")

gc.collect()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2750074265.py in <cell line: 0>()
      1 models = []
----> 2 for n_fold in FOLDS:
      3     model_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
      4     try:
      5         model = EffNet(params, out_dim=len(LABELS_))

NameError: name 'FOLDS' is not defined

## === cell 6
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
        pin_memory=True,
    )
    loaders.append(loader)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/80776428.py in <cell line: 0>()
      1 datasets, loaders = [], []
----> 2 for tta in TTAS:
      3     dataset = PlantDataset(
      4         df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
      5     )

NameError: name 'TTAS' is not defined

## === cell 7
def get_labels(row, labels_dict, th):
    """
    Convert a probability vector to a space‑delimited label string.
    - Select all labels with probability > th.
    - If none exceed th, pick the top‑probability label.
    - If the only selected label is 'healthy', return 'healthy'.
    - If 'healthy' appears together with other labels, drop 'healthy' and keep the diseases.
    """
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        if not idxs:
            idxs = [int(np.argmax(row))]
        lbls = [labels_dict[i] for i in idxs]

        if "healthy" in lbls:
            if len(lbls) == 1:
                return "healthy"
            else:
                lbls = [l for l in lbls if l != "healthy"]
        return " ".join(lbls) if lbls else "healthy"
    except Exception:
        return "healthy"


logits = None
if models:
    all_preds = []
    with torch.no_grad():
        for i, model in enumerate(models):
            for j, loader in enumerate(loaders):
                batch_preds = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE)
                    preds = torch.sigmoid(model(img_data)).cpu().numpy()
                    batch_preds.append(preds)  # shape (batch, num_labels)
                loader_preds = np.concatenate(
                    batch_preds, axis=0
                )  # (num_images, num_labels)
                all_preds.append(loader_preds)
                print(f"model {i} | loader {j} -> done")
    logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (num_images, num_labels)
else:
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    label_counts = np.zeros(len(LABELS_))
    for lbls in train_df["labels"]:
        for lbl in lbls.split():
            label_counts[LABELS_[lbl]] += 1
    prior = label_counts / label_counts.sum()
    logits = np.tile(prior, (len(df_sub), 1))

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1136563152.py in <cell line: 0>()
     41     logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (num_images, num_labels)
     42 else:
---> 43     train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
     44     label_counts = np.zeros(len(LABELS_))
     45     for lbls in train_df["labels"]:

NameError: name 'DATA_PATH' is not defined

## === cell 8
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
df_sub.head()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/624509254.py in <cell line: 0>()
      1 print("Label distribution in submission:")
----> 2 print(df_sub["labels"].value_counts())
      3 df_sub.head()
      4 
      5 

NameError: name 'df_sub' is not defined

## === cell 9
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/333940696.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 df_sub.to_csv(submission_path, index=False)
      3 print(f"Submission saved to {submission_path}")

NameError: name 'df_sub' is not defined
