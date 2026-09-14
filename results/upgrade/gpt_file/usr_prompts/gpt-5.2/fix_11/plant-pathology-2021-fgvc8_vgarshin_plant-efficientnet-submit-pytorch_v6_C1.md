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

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the EfficientNet weight-loading crash by loading the pretrained backbone weights in a way that matches your modified classifier (i.e., ignore classifier keys), while keeping the same model/head and inference logic. Then I fix the submission row-count issue by building `df_sub` directly from `sample_submission.csv` (official test set list) rather than listing the visible `test_images` folder, which is incomplete locally due to the hidden test set. Finally, I ensure `labels` formatting matches the competition requirement and always write a valid `submission.csv` with the exact expected columns.'
- What this solution (achieved 0.22786) has done: 'Your current score (0.24507) is higher than the target (0.15789), so we should *intentionally* move performance down toward the target with the smallest safe change that preserves your exact model/inference pipeline. The least invasive lever here is the decision threshold `TH`, which directly controls how many labels are predicted and thus the mean F1; increasing `TH` typically reduces positives and usually lower F1 in multilabel settings. I add a tiny automatic calibration step that chooses a slightly higher threshold (starting from your current `TH`) by applying the same post-processing rule (`get_labels`) on the already-computed test probabilities—this does not change the model, only label decoding. The code remains end-to-end, still writes a valid `submission.csv`, and keeps all paths and core logic intact.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.22786) is higher than the target (0.15789), so the goal is to move performance down toward the target with the smallest safe change that preserves your exact model and inference pipeline. The most direct, low-risk lever is the label-decoding threshold, but your current “healthy_rate”-based auto-thresholding uses only test predictions and can behave unpredictably; instead, we can choose a threshold using out-of-fold (OOF) predictions on the training set to explicitly target a mean F1 close to the destination score. This keeps the same model, same sigmoid outputs, same `get_labels` rule, and only adjusts the single scalar threshold used at submission time. We also keep a fallback to your original TH if training images/weights aren’t available, ensuring the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is above the target (0.15789), so we should intentionally reduce performance slightly with the smallest safe change that preserves your exact model/inference pipeline. The lowest-risk lever is the single scalar decoding threshold, but the current calibration uses a subset of train data in a way that can land on a threshold that’s still too “good”; instead, we select the threshold to hit a target “healthy rate” on the test predictions, which is a monotonic control knob that typically reduces mean F1 when we push it toward more “healthy-only” outputs. This keeps the model, weights, sigmoid, TTA, and label formatting unchanged—only the final label decoding threshold selection changes. We keep the existing train-based calibration as a fallback, but prefer the test-healthy-rate targeting when models are loaded (and avoid it for the random-head fallback to keep stability).'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is above the target (0.15789), so we should *intentionally* move performance downward toward the target with the smallest safe change that doesn’t alter the model or inference logic. The most direct, minimal lever is the single decoding threshold `TH`, but the current “healthy-rate targeting” can still land on a threshold that’s not “degrading” enough. I replace that with a tiny, deterministic threshold selection that uses the already-computed test probabilities to match a higher desired “healthy-only” rate (more conservative predictions), which typically reduces mean F1 in this competition. Everything else (model, weights loading, sigmoid, TTA, submission building from `sample_submission.csv`) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is above the target (0.15789), so we should intentionally reduce performance toward the target with the smallest, safest change. The least invasive control knob is still the single decoding threshold, but we push it more conservatively by increasing the target “healthy-only” rate so more samples become `"healthy"` and fewer diseases are predicted. To keep behavior stable and deterministic, we keep the same model, weights, sigmoid outputs, and TTA averaging, and only adjust the threshold-selection target from 0.92 to 0.97 (plus clamp the search lower bound at your base `TH` as before). This should move mean F1 down without changing any core modeling logic and still produce a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is above the target (0.15789), so we should intentionally reduce performance toward the target with the smallest safe change. The most direct lever that preserves your model/inference core is the final decoding threshold selection, so we make predictions more conservative by increasing the target “healthy-only” rate used in the existing binary-search threshold chooser. Concretely, we bump `target_healthy_rate` from 0.97 to 0.995 (and keep the search lower bound at your existing `TH`), which should yield more `"healthy"` outputs and typically lower mean F1. Everything else (model loading, sigmoid outputs, TTA averaging, submission built from `sample_submission.csv`, label formatting) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is above the target (0.15789), so we should intentionally reduce performance toward the target with the smallest possible change. The most direct lever that preserves your exact model/inference pipeline is the label-decoding threshold selection; making predictions more conservative (higher threshold) typically lowers mean F1 here. I only adjust the `target_healthy_rate` used by your existing binary-search threshold chooser from 0.995 to 0.9995 to force more `"healthy"` outputs, and I leave the model loading, TTA, sigmoid averaging, and submission building unchanged. This should move the score downward toward the target while still producing a valid `submission.csv`.'

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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
TEST = True
VER = "v1"
TH = 0.4
TTAS = [0]
FOLDS = [0]


def _pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_PATH = _pick_existing(
    [
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../kaggle/input/plant-pathology-2021-fgvc8",
        "./data/plant-pathology-2021-fgvc8",
        "./data",
    ]
)

if DATA_PATH is None:
    raise FileNotFoundError("Could not locate dataset root folder.")

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Images folder not found: {IMGS_PATH}")

MDLS_PATH = _pick_existing(
    [
        f"../input/plant-models-{VER}",
        f"/kaggle/input/plant-models-{VER}",
        f"../kaggle/input/plant-models-{VER}",
        f"./models_{VER}",
    ]
)

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 2
default_params = {
    "img_size": 512,
    "batch_size": 8,
    "dropout": 0.2,
    "backbone": "efficientnet-b0",
    "workers": 2,
    "labels_": [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ],
    "labels": {
        str(i): lbl
        for i, lbl in enumerate(
            [
                "complex",
                "frog_eye_leaf_spot",
                "healthy",
                "powdery_mildew",
                "rust",
                "scab",
            ]
        )
    },
}

params = default_params.copy()

if MDLS_PATH is not None:
    params_path = os.path.join(MDLS_PATH, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as file:
            loaded = json.load(file)
        params.update(loaded)

LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = 4 if KAGGLE else int(params.get("workers", 2))

print(
    "loaded/used params:",
    {k: params[k] for k in ["img_size", "batch_size", "dropout", "backbone"]},
)
print("num labels:", len(LABELS_), "WORKERS:", WORKERS)



## === cell 3
sub_path = _pick_existing(
    [
        os.path.join(DATA_PATH, "sample_submission.csv"),
        "./sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)
if sub_path is None or (not os.path.exists(sub_path)):
    raise FileNotFoundError(
        "Could not locate sample_submission.csv to build test index."
    )

df_sub = pd.read_csv(sub_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError(
        f"sample_submission.csv has unexpected columns: {df_sub.columns.tolist()}"
    )

df_sub["image"] = df_sub["image"].astype(str)
df_sub = df_sub.sort_values("image").reset_index(drop=True)
df_sub["labels"] = "healthy"

print(df_sub.head())
print("num test images (from sample_submission):", len(df_sub))




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


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, normalize=False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.normalize = normalize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None or not np.any(img):
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = flip(img, axis=self.tta)

        if self.normalize:
            img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        img = img.transpose(2, 0, 1)
        if self.labels:
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()

        backbone = str(params.get("backbone", "efficientnet-b0")).lower()
        if "b0" in backbone:
            self.enet = models.efficientnet_b0(weights=None)
        elif "b1" in backbone:
            self.enet = models.efficientnet_b1(weights=None)
        elif "b2" in backbone:
            self.enet = models.efficientnet_b2(weights=None)
        else:
            self.enet = models.efficientnet_b0(weights=None)

        if hasattr(self.enet, "classifier") and isinstance(
            self.enet.classifier, nn.Sequential
        ):
            nc = self.enet.classifier[-1].in_features
            self.enet.classifier = nn.Identity()
        else:
            nc = 1280

        self.myfc = nn.Sequential(
            nn.Dropout(float(params.get("dropout", 0.2))),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
            nn.BatchNorm1d(int(nc / 4)),
            nn.Dropout(float(params.get("dropout", 0.2))),
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
loaded_any = False

if MDLS_PATH is not None:
    for n_fold in FOLDS:
        path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        if os.path.exists(path):
            model = EffNet(params, out_dim=len(LABELS_))
            state_dict = torch.load(path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.float().eval().to(DEVICE)
            models_list.append(model)
            loaded_any = True
            print("loaded:", path)

used_pretrained_fallback = False
if not loaded_any:
    print(
        "WARNING: No model weights found in MDLS_PATH. Falling back to ImageNet-pretrained EfficientNet backbone."
    )
    backbone = str(params.get("backbone", "efficientnet-b0")).lower()
    if "b0" in backbone:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        enet = models.efficientnet_b0(weights=weights)
    elif "b1" in backbone:
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
        enet = models.efficientnet_b1(weights=weights)
    elif "b2" in backbone:
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        enet = models.efficientnet_b2(weights=weights)
    else:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        enet = models.efficientnet_b0(weights=weights)

    model = EffNet(params, out_dim=len(LABELS_))

    pretrained_sd = enet.state_dict()
    pretrained_sd = {
        k: v for k, v in pretrained_sd.items() if not k.startswith("classifier.")
    }
    missing, unexpected = model.enet.load_state_dict(pretrained_sd, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(
            f"Unexpected keys when loading pretrained backbone: {unexpected}"
        )

    model.float().eval().to(DEVICE)
    models_list.append(model)
    loaded_any = True
    used_pretrained_fallback = True
    print(
        "Loaded ImageNet-pretrained backbone as fallback; head remains randomly initialized.",
        "Missing keys (expected due to classifier removal):",
        missing,
    )

gc.collect()



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=int(params["img_size"]),
        labels=None,
        transform=None,
        tta=tta,
        normalize=bool(used_pretrained_fallback),
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    loaders.append(loader)

print("num loaders:", len(loaders))




## === cell 7
def get_labels(row, labels, th):
    idxs = [i for i, x in enumerate(row) if x > th]
    names = [labels[str(i)] for i in idxs]
    names = "healthy" if ("healthy" in names or len(names) == 0) else " ".join(names)
    return names


def _choose_th_by_target_healthy_rate(preds, labels_map, base_th, target_healthy_rate):
    lo, hi = float(base_th), 0.999
    for _ in range(18):  # enough for ~1e-5 resolution, still cheap
        mid = (lo + hi) / 2.0
        labs = [get_labels(x, labels_map, mid) for x in preds]
        healthy_rate = float(np.mean([l == "healthy" for l in labs]))
        if healthy_rate < float(target_healthy_rate):
            lo = mid
        else:
            hi = mid
    final_th = hi
    labs = [get_labels(x, labels_map, final_th) for x in preds]
    final_hr = float(np.mean([l == "healthy" for l in labs]))
    return float(final_th), float(final_hr)


chosen_th = TH

if loaded_any:
    all_model_tta_preds = []

    with torch.no_grad():
        for mi, model in enumerate(models_list):
            for tj, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True).float()
                    preds = model(img_data).sigmoid().detach().cpu().numpy()  # (B, C)
                    preds_batches.append(preds)
                preds_full = np.concatenate(preds_batches, axis=0)  # (N, C)
                all_model_tta_preds.append(preds_full)
                print(f"model {mi} | loader {tj} -> done, shape={preds_full.shape}")

    all_preds = np.mean(np.stack(all_model_tta_preds, axis=0), axis=0)  # (N, C)
    if all_preds.shape[0] != len(df_sub):
        raise RuntimeError(
            f"Prediction count mismatch: {all_preds.shape[0]} vs {len(df_sub)}"
        )

    if not used_pretrained_fallback:
        target_score = 0.1578947368421052

        target_healthy_rate = 0.9995

        chosen_th, achieved_hr = _choose_th_by_target_healthy_rate(
            all_preds, LABELS, base_th=TH, target_healthy_rate=target_healthy_rate
        )
        print(
            f"Chose TH by binary-search healthy-rate targeting: TH={chosen_th:.4f}, healthy_rate={achieved_hr:.3f} "
            f"(target healthy_rate={target_healthy_rate:.4f}) aiming to move score toward target={target_score:.5f}"
        )
    else:
        chosen_th = TH
        print(
            f"Using TH={chosen_th} (pretrained-backbone fallback active; keep stable)"
        )

    df_sub["labels"] = [get_labels(x, LABELS, chosen_th) for x in all_preds]

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
print(df_sub.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "bytes:",
    os.path.getsize("submission.csv"),
)
