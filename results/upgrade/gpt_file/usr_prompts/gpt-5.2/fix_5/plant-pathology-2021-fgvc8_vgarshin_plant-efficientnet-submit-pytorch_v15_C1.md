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

0.8245798707294568

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The main blocker is that this notebook expects an external pretrained-model dataset (`../input/plant-models-v6/params.json` + `model_best_*.pth`) that is not present in your environment, so `params` never loads and all downstream inference fails. I add a minimal, safe fallback path: if the pretrained bundle is missing, the code still run end-to-end by generating a valid `submission.csv` from `sample_submission.csv` (defaulting to `healthy`), matching the required row count and format. I also fix the test-image listing to use the official `sample_submission.csv` as the authoritative index to avoid row-mismatch errors (hidden test set on Kaggle). These changes are purely for correctness/execution and submission validity; they preserve the original ensemble-inference logic when the pretrained bundle is available.'
- What this solution (achieved 0.24507) has done: 'Your current score is low because the notebook is effectively submitting an “all healthy” fallback when the pretrained model bundle is missing, which produces a weak Mean F1. To move toward the target without changing the model architecture/training, the smallest legitimate improvement is to ensure the pretrained assets are found from all plausible Kaggle input locations (including the nested `/kaggle/input/.../plant-pathology-2021-fgvc8/` pattern) instead of only `../input/plant-models-v6`. I add a robust search for `params.json` and `model_best_*.pth` across common Kaggle paths and only fall back to “healthy” if nothing is found. This preserves identical inference/ensembling/thresholding logic when weights exist, and only changes path resolution to avoid accidental fallback.'
- What this solution (achieved 0.24507) has done: 'Your low score is consistent with the code falling back to an “all healthy” submission because the pretrained bundle isn’t found, so the smallest score-improving change is to actually locate and use the existing pretrained `plant-models-v6` assets if they are present anywhere under `/kaggle/input`. I add a minimal recursive search for `params.json` and `model_best_*.pth` (only if the current direct-path checks fail), and keep the exact same inference/ensembling/thresholding logic once the files are found. I also keep using `sample_submission.csv` as the authoritative test index (important for the hidden test set) and still write a valid `submission.csv` even if no weights exist. This should move the score upward toward your target without changing model architecture or evaluation semantics.'

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
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8",
]
DATA_PATH = None
for p in _CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "sample_submission.csv")):
        DATA_PATH = p
        break
if DATA_PATH is None:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"

MDLS_CANDIDATES = [
    f"../input/plant-models-{VER}",
    f"/kaggle/input/plant-models-{VER}",
    f"../input/plant-models-{VER}/plant-models-{VER}",
    f"/kaggle/input/plant-models-{VER}/plant-models-{VER}",
]

MDLS_PATH = None
for p in MDLS_CANDIDATES:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "params.json")):
        MDLS_PATH = p
        break


def _find_pretrained_bundle(ver: str):
    roots = ["/kaggle/input", "../input"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        candidate_dirs = []
        for d in os.listdir(root):
            if f"plant-models-{ver}" in d:
                candidate_dirs.append(os.path.join(root, d))
        walk_roots = candidate_dirs if candidate_dirs else [root]
        for wr in walk_roots:
            for dirpath, dirnames, filenames in os.walk(wr):
                if "params.json" in filenames:
                    params_path = os.path.join(dirpath, "params.json")
                    has_any_weight = any(
                        fn.startswith("model_best_") and fn.endswith(".pth")
                        for fn in filenames
                    )
                    if has_any_weight:
                        return dirpath
    return None


if MDLS_PATH is None:
    found = _find_pretrained_bundle(VER)
    if found is not None:
        MDLS_PATH = found
    else:
        MDLS_PATH = f"../input/plant-models-{VER}"  # keep original default for backwards compatibility

TTAS = [0, 1]
FOLDS = [0, 1, 2]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 4
params_path = f"{MDLS_PATH}/params.json"
HAS_PRETRAINED = os.path.isfile(params_path)

params = None
LABELS_ = None
LABELS = None
WORKERS = 2

if HAS_PRETRAINED:
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]  # list-like (model output order)
    LABELS = params["labels"]  # dict-like index->label strings
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
    print("loaded pretrained params keys:", sorted(list(params.keys())))
else:
    print(
        f"WARNING: Missing pretrained bundle at {MDLS_PATH} (no params.json). "
        "Will create a valid fallback submission from sample_submission.csv."
    )



## === cell 5
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")

df_sub = pd.read_csv(sample_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError(
        f"Unexpected sample_submission.csv columns: {df_sub.columns.tolist()}"
    )

df_sub["labels"] = "healthy"
display(df_sub.head())
print("n_test_images (from sample_submission):", len(df_sub))

if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Image directory not found: {IMGS_PATH}")




## === cell 6
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :]
    elif axis == 2:
        return img[:, ::-1]
    elif axis == 3:
        return img[::-1, ::-1]
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
    EfficientNet backbone -> replace classifier with Identity -> custom FC head.
    (Preserves original core logic; uses torchvision EfficientNet.)
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
if HAS_PRETRAINED:
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
else:
    print("Skipping model loading because pretrained files are not available.")



## === cell 8
datasets, loaders = [], []
if HAS_PRETRAINED:
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
else:
    print("Skipping dataloaders because pretrained files are not available.")




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


if HAS_PRETRAINED:
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



## === cell 10
print("value counts:")
print(df_sub.labels.value_counts().head(20))
display(df_sub.head())



## === cell 11
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("wrote:", sub_path, "rows:", len(df_sub))
print(df_sub.dtypes)
