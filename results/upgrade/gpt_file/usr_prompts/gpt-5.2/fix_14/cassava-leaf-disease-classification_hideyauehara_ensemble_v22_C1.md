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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8936234511937141

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the test image discovery so it doesn’t accidentally include the nested `test_images/` directory name as a “file”, which is what caused the `FileNotFoundError`. Then I ensure `df_test` is built strictly from the competition’s `sample_submission.csv` image list (correct length/order), and only use the filesystem as a fallback; this also fixes the “Invalid submission length” error. Finally, I make the image transforms compatible with Albumentations v2 (`RandomResizedCrop(height, width)`), and keep the existing ensemble/prediction logic unchanged so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the Albumentations v2 `RandomResizedCrop` initialization by using the required `size=(H,W)` argument, which removes the ValidationError and ensures `transform` is defined so later cells can run. I also add a safe fallback so `CenterCrop` doesn’t crash when the source image is smaller than 512px (by resizing up first), which prevents sporadic runtime errors on small images. Finally, I update the model checkpoint loader to use `strict=False` so common key-mismatch patterns (e.g., `model.` prefixes or classifier head names) don’t abort inference; this should materially improve score versus the current near-random predictions while keeping the same ensemble/inference core logic.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target (0.61099 → 0.8936) is large, so the most likely issue is that the inference-time model definitions don’t match how the checkpoints were trained, meaning the classifier head weights aren’t being loaded/used correctly (especially for EfficientNet). I keep your ensemble/loop logic identical, but fix the EfficientNet wrapper so it uses `model.classifier` properly and can load head weights from common checkpoint key patterns. I also make DenseNet/ResNet head-weight loading more robust by remapping typical checkpoint key names into your wrapper’s `fc.*`, without changing the architecture or transforms. These changes should materially improve accuracy while keeping the same inference-time semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.89362), so we should improve inference correctness without changing the ensemble structure, transforms list, or model wrappers. The biggest likely issue is that your checkpoints may store logits under `fc.*` for the wrapper, while the base backbone weights were saved under `model.*` (or vice-versa), so your current remapping can leave the classifier head randomly initialized for some checkpoints—this can easily cap accuracy. I make `_remap_checkpoint_keys_for_wrapper` explicitly map common EfficientNet-B7 keys (`classifier.1.*`) into your wrapper’s expected `model.classifier.1.*`, and also map common backbone prefixes into your wrapper (`convlayer.*` for ResNet, `convlayer.*`/`features.*` for DenseNet) so more pretrained weights load correctly. These are minimal, inference-only changes that preserve your architecture and prediction loop but should move the score upward toward the target.'
- What this solution (achieved 0.61099) has done: 'Your score gap to the target is large (0.61099 → 0.89362), so the smallest likely win is to ensure every checkpoint actually loads both backbone and head weights into your existing wrapper modules (right now EfficientNet/DenseNet key remaps can silently miss and leave key parts random). I keep your model wrappers, transforms, and ensemble loop exactly the same, but make the checkpoint key remapping cover the common patterns for EfficientNet-B7 (`classifier.1.*` as well as `model.classifier.1.*`) and DenseNet (`features.*` as well as already-wrapped `convlayer.*`), plus remove an over-aggressive `model.` prefixing rule that can create wrong keys. I also add a lightweight sanity print of how many parameters actually loaded (missing/unexpected) per model to catch partial loads that correlate with low leaderboard accuracy. These changes are inference-only, preserve your core logic, and are directly aimed at moving accuracy upward toward your target.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so the most likely “minimal-change” gain is fixing checkpoint loading so the classifier head weights are actually used for all backbones (especially DenseNet), because a randomly-initialized head cap accuracy around what you’re seeing. I keep your ensemble loop, transforms list, and model wrappers unchanged, but adjust the DenseNet key remapping so `features.*` keys map to the wrapper’s `convlayer.features.*` (not to a non-existent `convlayer.*` substructure). I also make the ResNet wrapper’s squeeze stable (`flatten(1)`) to avoid occasional shape issues without changing semantics, and I add a tiny “loaded weight ratio” print to verify that weights really loaded (this doesn’t affect predictions). The script still run end-to-end and write `submission.csv` in the required format.'
- What this solution (achieved 0.61099) has done: 'Your gap to the target is large (0.61099 → 0.89362), so the most likely minimal-change gain is ensuring inference uses the same input normalization/resolution the checkpoints expect and that DenseNet’s pooled features keep the batch dimension (your current `.squeeze()` can drop it for batch_size=1 and subtly break behavior). I (1) switch EfficientNet’s normalization to its torchvision-recommended mean/std while keeping the same TTA list and transforms structure, and (2) replace DenseNet’s `squeeze()` with `flatten(1)` to preserve semantics and avoid shape edge-cases. I also make `DataLoader(drop_last=False)` explicit (no behavior change expected) and keep your ensemble loop, wrappers, and checkpoint remapping logic intact so this stays an inference-only correctness improvement. The script still run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8936), so the most likely minimal-change improvement is ensuring the EfficientNet-B7 classifier head weights actually load from the checkpoint. Right now your remapping handles `head.*` and `classifier.*` but can miss the common torchvision EfficientNet key pattern `classifier.1.*` (and variants with/without `model.`), which leaves the head randomly initialized and tanks accuracy. I minimally extend `_remap_checkpoint_keys_for_wrapper` for `FinalLayerMixupModelEN` to map `classifier.1.weight/bias` (and `model.classifier.1.*`) into your wrapper’s expected `model.classifier.1.*`, without changing your model, transforms, ensemble logic, or inference loop. This should move the score upward toward your target while keeping everything else identical and still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.89362), so we should focus on an inference-correctness fix that doesn’t change your ensemble/TTA loop or model architectures. The most likely remaining issue is that some checkpoints were saved from wrappers that used different classifier attribute names (e.g., `linear.*` for EfficientNet, `fc.*` or `classifier.*` for DenseNet), so your current remapping can still miss the head weights and leave parts randomly initialized. I minimally extend `_remap_checkpoint_keys_for_wrapper` to cover these common head-key variants for EfficientNet and DenseNet while keeping `strict=False` and the rest of the pipeline identical. This should increase the fraction of correctly loaded classifier weights and move accuracy upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.89362), so the smallest likely gain is fixing inference-time checkpoint loading so the classifier head weights are actually applied (a partially/random head can cap accuracy around your current level). I keep your ensemble/TTA loop, transforms, and model wrappers unchanged, but extend `_remap_checkpoint_keys_for_wrapper` to cover common EfficientNet-B7 head key variants (`classifier.1.*`, `classifier.*`, `fc.*`, `head.*`, `linear.*`) and to properly map DenseNet backbones saved under `model.features.*` into your wrapper’s `convlayer.*`. I also add a tiny guard to unwrap checkpoints saved as `{'model': state_dict}` (common in some training scripts) so more weights load without changing semantics. These are inference-only correctness fixes and should move accuracy upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8936), so the most likely minimal-change improvement is to fix inference-time weight loading so the classifier head weights are actually used for DenseNet and EfficientNet checkpoints. I keep your ensemble/TTA loop, transforms, wrappers, and loss exactly the same, but (1) correct DenseNet’s forward to include the required ReLU after `features` (matching torchvision DenseNet semantics) and (2) extend checkpoint key remapping to cover common DenseNet head key variants (`classifier.*`) and to map EfficientNet `classifier.1.*` keys regardless of `model.` prefix. These are inference-only correctness fixes and should move accuracy upward toward your target while still producing the same valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target, so the most likely minimal-change win is correcting inference-time weight loading so your wrappers actually receive the checkpoint’s backbone + classifier weights (right now `FinalLayerMixupModel` and `FinalLayerMixupModelDenseNet` often won’t map `model.*` or common DenseNet/resnet naming patterns into `convlayer.*`, leaving large parts random). I keep your ensemble/TTA loop, transforms, model wrappers, and loss exactly the same, but make `_remap_checkpoint_keys_for_wrapper` reliably map common prefixes (`model.`, `backbone.`, `net.`) and DenseNet keys (`features.*`, `classifier.*`) into your wrapper’s expected names. I also make `predict_model` pass a dummy `labels` tensor (instead of `False`) to avoid edge-case behavior inside wrappers that were originally written for training/val signatures, without changing the test forward path. These are inference-only correctness fixes aimed at increasing accuracy toward your target while still producing the same `submission.csv` format.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(f"../input/densenet201-04-2019data/*.pth") + glob.glob(
    f"../input/eb7slseed70/efficientnet-b7sl_SEED70.orig/*.pth"
)

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
BASE_DIR_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_DIR = None
for cand in BASE_DIR_CANDIDATES:
    if os.path.exists(cand):
        BASE_DIR = cand
        break
if BASE_DIR is None:
    BASE_DIR = "../input"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
print(f"KAGGLE_KERNEL_RUN_TYPE={run_type!r}")

if os.path.isdir(os.path.join(BASE_DIR, "test_images")):
    TEST_PATH = os.path.join(BASE_DIR, "test_images")
elif os.path.isdir(
    os.path.join(BASE_DIR, "cassava-leaf-disease-classification", "test_images")
):
    TEST_PATH = os.path.join(
        BASE_DIR, "cassava-leaf-disease-classification", "test_images"
    )
elif os.path.isdir(os.path.join(BASE_DIR, "train_images")):
    TEST_PATH = os.path.join(BASE_DIR, "train_images")
else:
    TEST_PATH = BASE_DIR

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")


def list_image_files(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.isdir(folder):
        return []
    files = []
    for n in os.listdir(folder):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and n.lower().endswith(exts):
            files.append(n)
    return files


test_files = list_image_files(TEST_PATH)
print(f"Number of test images found in folder: {len(test_files)}")



## === cell 7
sample_path_candidates = [
    os.path.join(BASE_DIR, "sample_submission.csv"),
    os.path.join(
        "/kaggle/input/cassava-leaf-disease-classification", "sample_submission.csv"
    ),
    os.path.join(
        "/kaggle/data/cassava-leaf-disease-classification", "sample_submission.csv"
    ),
    os.path.join(
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "sample_submission.csv",
    ),
    os.path.join(
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "sample_submission.csv",
    ),
]
sample_csv = None
for p in sample_path_candidates:
    if os.path.exists(p):
        sample_csv = p
        break

if sample_csv is not None:
    df_test = pd.read_csv(sample_csv)
    if "image_id" not in df_test.columns:
        raise ValueError(f"sample_submission.csv missing image_id column: {sample_csv}")
    if "label" not in df_test.columns:
        df_test["label"] = 0
    print(f"Loaded sample_submission from: {sample_csv} shape={df_test.shape}")
else:
    df_test = pd.DataFrame(test_files, columns=["image_id"])
    df_test["label"] = 1
    print(
        "WARNING: sample_submission.csv not found; using filesystem-derived df_test:",
        df_test.shape,
    )



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean_resnet = [0.485, 0.456, 0.406]
std_resnet = [0.229, 0.224, 0.225]
mean_eff = [0.485, 0.456, 0.406]
std_eff = [0.229, 0.224, 0.225]
try:
    from torchvision.models import EfficientNet_B7_Weights

    mean_eff = list(EfficientNet_B7_Weights.DEFAULT.transforms().mean)
    std_eff = list(EfficientNet_B7_Weights.DEFAULT.transforms().std)
except Exception:
    pass

transform_resnetlike = {
    "test": [
        Compose(
            [
                A.LongestMaxSize(max_size=SIZE, p=1.0),
                A.PadIfNeeded(
                    min_height=SIZE,
                    min_width=SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.LongestMaxSize(max_size=SIZE, p=1.0),
                A.PadIfNeeded(
                    min_height=SIZE,
                    min_width=SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(
                    mean=mean_resnet, std=std_resnet, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}

transform_effnet = {
    "test": [
        Compose(
            [
                A.LongestMaxSize(max_size=SIZE, p=1.0),
                A.PadIfNeeded(
                    min_height=SIZE,
                    min_width=SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.LongestMaxSize(max_size=SIZE, p=1.0),
                A.PadIfNeeded(
                    min_height=SIZE,
                    min_width=SIZE,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean_eff, std=std_eff, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = torch.relu(x)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = torch.relu(x)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)

            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = torch.relu(x1)
        x2 = torch.relu(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 13
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 14
def _unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def _strip_known_prefixes(k):
    for pref in ("module.", "model.", "net.", "backbone."):
        if k.startswith(pref):
            return k[len(pref) :]
    return k


def _remap_checkpoint_keys_for_wrapper(net, ckpt):
    """
    Change rationale (score improvement): maximize the fraction of correctly-loaded
    backbone+head weights from heterogeneous checkpoint naming conventions, without
    changing any modeling/inference logic. This avoids leaving the backbone/head
    randomly initialized for some checkpoints, which can cap accuracy near ~0.6.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    out = {}
    for k, v in ckpt.items():
        base = _strip_known_prefixes(k)

        if base != k:
            base = _strip_known_prefixes(base)

        if isinstance(net, FinalLayerMixupModelEN):
            if base in (
                "head.weight",
                "classifier.weight",
                "linear.weight",
                "fc.weight",
            ):
                base = "classifier.1.weight"
            if base in ("head.bias", "classifier.bias", "linear.bias", "fc.bias"):
                base = "classifier.1.bias"

            if base.startswith(("features.", "classifier.", "stem.", "blocks.")):
                nk2 = "model." + base
            else:
                nk2 = "model." + base

        elif isinstance(net, FinalLayerMixupModel):
            if base in (
                "fc.weight",
                "classifier.weight",
                "head.weight",
                "linear.weight",
            ):
                nk2 = "fc.weight"
            elif base in ("fc.bias", "classifier.bias", "head.bias", "linear.bias"):
                nk2 = "fc.bias"
            elif base.startswith("convlayer."):
                nk2 = base
            else:
                backbone_prefixes = (
                    "conv1.",
                    "bn1.",
                    "layer1.",
                    "layer2.",
                    "layer3.",
                    "layer4.",
                    "downsample.",
                )
                if base.startswith(backbone_prefixes):
                    nk2 = "convlayer." + base
                else:
                    if base.startswith("model.") and base[len("model.") :].startswith(
                        backbone_prefixes
                    ):
                        nk2 = "convlayer." + base[len("model.") :]
                    else:
                        nk2 = base

        elif isinstance(net, FinalLayerMixupModelDenseNet):
            if base in (
                "fc.weight",
                "head.weight",
                "linear.weight",
                "classifier.weight",
            ):
                nk2 = "fc.weight"
            elif base in ("fc.bias", "head.bias", "linear.bias", "classifier.bias"):
                nk2 = "fc.bias"
            elif base.startswith("features."):
                nk2 = "convlayer." + base  # convlayer.features.*
            elif base.startswith("convlayer.features."):
                nk2 = base
            elif base.startswith("convlayer."):
                nk2 = base
            else:
                if base.startswith("model.features."):
                    nk2 = "convlayer." + base[len("model.") :]
                else:
                    nk2 = base

        else:
            nk2 = base

        out[nk2] = v
    return out


def predict_model(basename, net, dataloader):
    """
    basename: 学習済みモデル名
    net     : 学習済みモデル
    """
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            dummy_labels = torch.zeros(
                (inputs.size(0),), device=inputs.device, dtype=torch.long
            )
            outputs = net(inputs, dummy_labels, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 15
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained .pth models found; using a random-initialized fallback model to generate a valid submission."
    )
    pretrained_models = ["__FALLBACK_TORCHVISION_EFFICIENTNET_B7__"]

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    if pretrained_model == "__FALLBACK_TORCHVISION_EFFICIENTNET_B7__":
        basename = pretrained_model

    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
        active_transform = transform_resnetlike
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
        active_transform = transform_resnetlike
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
        active_transform = transform_resnetlike
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        active_transform = transform_resnetlike
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(weights=None)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        active_transform = transform_resnetlike
    elif ("efficientnet-b7" in basename) or (
        pretrained_model == "__FALLBACK_TORCHVISION_EFFICIENTNET_B7__"
    ):
        MODEL_NAME = "efficientnet-b7"
        net = efficientnet_b7(weights=None)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
        active_transform = transform_effnet
    else:
        print(f"{basename} is not supported.")
        sys.exit()

    print(f"{basename}: {MODEL_NAME}")

    if pretrained_model != "__FALLBACK_TORCHVISION_EFFICIENTNET_B7__":
        ckpt = torch.load(pretrained_model, map_location="cpu")
        ckpt = _unwrap_state_dict(ckpt)
        ckpt = _remap_checkpoint_keys_for_wrapper(net, ckpt)

        missing, unexpected = net.load_state_dict(ckpt, strict=False)

        total_params = len(list(net.state_dict().keys()))
        loaded_params = total_params - len(missing)
        print(f"Checkpoint load coverage: {loaded_params}/{total_params} keys loaded")

        if (len(unexpected) > 0) or (len(missing) > 0):
            print(
                f"Checkpoint load info: missing={len(missing)} unexpected={len(unexpected)}"
            )
            if len(missing) > 0:
                print("  missing (sample):", missing[:10])
            if len(unexpected) > 0:
                print("  unexpected (sample):", unexpected[:10])

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(active_transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
            ),
        }

        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if len(probability) == 0:
    print(
        "WARNING: No probabilities produced; using sample_submission labels as fallback."
    )
    if "label" not in df_test.columns:
        df_test["label"] = 0
else:
    probs = np.stack(probability, axis=0)  # (n_preds, n_images, n_classes)
    df_test["mean"] = probs.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 16
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"].astype(int)
else:
    df_test["label"] = df_test["label"].astype(int)



## === cell 17
df_test.head()



## === cell 18
sub = df_test[["image_id", "label"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Unique labels:", np.sort(sub["label"].unique()))
