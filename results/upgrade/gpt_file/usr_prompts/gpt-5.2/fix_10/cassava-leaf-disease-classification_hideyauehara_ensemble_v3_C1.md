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

0.8948322756119673

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the path/environment detection so the notebook reliably finds the Cassava dataset under `/kaggle/input` or `/kaggle/data`, which currently causes `FileNotFoundError` and prevents `df_test` from being created. I also make the Albumentations `RandomResizedCrop` call compatible with albumentations==2.x (your current usage is a v1-style signature that can error), and fix a shape bug from `squeeze()` that can break inference when batch size is 1. Finally, I ensure we always write a valid `submission.csv` with the required `image_id,label` columns, falling back to `sample_submission.csv` only if no models are found/usable.'
- What this solution (achieved 0.11584) has done: 'I fix the Albumentations v2 API break causing `RandomResizedCrop` to error by switching to the required `size=(H,W)` argument while keeping the same augmentation intent. I also update the deprecated torchvision model construction (`pretrained=False`) to the current `weights=None` API to avoid runtime warnings/errors under torchvision==0.21. Finally, I improve score toward the target by correctly discovering the provided pretrained model checkpoints in this environment (your current glob paths likely find 0 models, which forces a near-random submission), while keeping the ensemble/inference core logic unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low (0.11584 vs target 0.8948), which strongly suggests the submission is effectively random or mismatched rather than “just undertrained”. The smallest high-impact fix is to correctly load checkpoints even when they were saved from `DataParallel` (`module.` prefix) or wrapped under keys like `state_dict`, which would otherwise silently skip or misload weights and yield near-random predictions. I add a minimal, safe checkpoint-unwrapping + key-renaming loader while keeping the same ensemble/TTA inference logic and model definitions. I also enforce test image order to match `sample_submission.csv` so predictions align with `image_id` exactly (misalignment can also destroy accuracy while still producing a valid CSV).'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.8948), which strongly indicates a catastrophic issue like predicting on the wrong images or misaligned `image_id` order rather than “slightly suboptimal modeling.” The smallest high-impact fix is to ensure `df_test` is built from `sample_submission.csv` (the authoritative test image list/order) and that `TestDataset` loads images from the real `test_images` directory (not falling back to `train_images`). I keep your ensemble/TTA/model logic intact, but make the prediction loop iterate exactly over `sample_submission.csv` ids so `avg_proba` rows align 1:1 with submission rows. This should move accuracy sharply upward toward the target without changing architecture, loss, or inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is so far below the target (0.8948) that this is almost certainly a checkpoint/model mismatch rather than “needs tuning”. With minimal changes and without altering your model architecture/inference approach, I (1) restrict checkpoint discovery to likely cassava-related folders to avoid accidentally loading unrelated `.pth` files, and (2) make checkpoint loading robust to common nesting/prefix patterns and to “fc/classifier/_fc” name differences by remapping head keys into your wrapper’s `fc.*` when needed. This should turn your predictions from near-random into meaningful class probabilities, moving accuracy sharply upward toward the target band while keeping the ensemble+TTA logic intact. I also add a sanity check that `avg_proba` row-count matches `sample_submission` to prevent silent misalignment.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584 vs target 0.8948) is far too low to be a “tuning” issue; it almost always means you are loading no usable cassava checkpoints (so predictions are effectively random/fallback) or you are using incompatible weights due to head/key mismatches. I keep your ensemble + TTA inference exactly as-is, but make checkpoint discovery actually find real `.pth` files in this dataset environment (your current glob patterns don’t use `recursive=True` so `**` won’t work), and I make loading more robust for common wrapper prefixes (`convlayer.*` + `fc.*`) so weights that match your wrapper load instead of being skipped. Finally, I enforce that `df_test` exactly matches `sample_submission.csv` order and add a hard sanity check so we never silently write a misaligned submission.'
- What this solution (achieved 0.11584) has done: 'Your current score is so far below the target that it almost certainly comes from using incompatible checkpoints and/or accidentally falling back to `sample_submission` labels (which are not meaningful predictions), producing near-random accuracy. I make the smallest changes that (1) prefer loading only cassava-relevant checkpoints and (2) make checkpoint loading robust to common “classifier/fc/_fc” naming so the final linear layer weights actually load instead of being treated as missing/incompatible. I also stop the script from silently writing the fallback submission when models were found but produced misaligned probability shapes, by enforcing a hard shape check and only averaging probabilities with matching `(N,5)` size. These changes keep your model wrappers, TTA list, and inference loop intact while making the ensemble actually use real learned weights.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, which almost certainly means inference is effectively random; the biggest minimal fix is to stop the non-deterministic TTA that uses `RandomResizedCrop` during test-time and switch those to deterministic resized crops (same intent: multi-crop/TTA, but reproducible and not randomly changing every run). I also fix an internal inconsistency where `seed_everything` sets `cudnn.deterministic=True` but also enables `benchmark=True` (these conflict and can cause unstable outputs), making inference stable without changing model architecture or loss. Finally, I keep your checkpoint loading/ensemble logic intact but enforce deterministic behavior in the prediction loop (no grad, consistent flags), which should move accuracy up substantially toward the target if checkpoints are meaningful.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
def _glob_many(patterns, recursive=False):
    out = []
    for p in patterns:
        out += glob.glob(p, recursive=recursive)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


pretrained_models = _glob_many(
    [
        "../input/resnet50-04-2019/*.pth",
        "../input/resnet152-04-2019data/*.pth",
        "../input/eb7-00-baseline/*.pth",
        "/kaggle/input/**/cassava*/*/*.pth",
        "/kaggle/input/**/cassava*/*.pth",
        "/kaggle/input/**/cassava*/*/*.pt",
        "/kaggle/input/**/cassava*/*.pt",
        "/kaggle/input/**/cassava*/*/*.bin",
        "/kaggle/input/**/cassava*/*.bin",
        "/kaggle/working/**/*.pth",
        "/kaggle/working/**/*.pt",
        "/kaggle/data/**/cassava*/*/*.pth",
        "/kaggle/data/**/cassava*/*.pth",
    ],
    recursive=True,
)

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    to_show = np.sort(pretrained_models)[:50]
    print("\n".join(to_show))
    if len(pretrained_models) > 50:
        print(f"... (showing first 50 of {len(pretrained_models)})")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
import sys

EfficientNet = None
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet  # type: ignore
except Exception as e:
    EfficientNet = None
    print("efficientnet_pytorch not available; EfficientNet models will be skipped.")



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _first_existing(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


BASE_DIR = _first_existing(
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "data",
)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory. Checked common /kaggle/input and /kaggle/data paths."
    )

test_dir = _first_existing(
    f"{BASE_DIR}/test_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/test_images",
)
train_dir = _first_existing(
    f"{BASE_DIR}/train_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/train_images",
)

if test_dir is None:
    raise FileNotFoundError(
        f"test_images not found under BASE_DIR={BASE_DIR}. Refusing to fall back to train_images because it breaks submission alignment."
    )

TEST_PATH = test_dir
test_files = sorted(os.listdir(TEST_PATH))

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
sample_sub_path = _first_existing(
    f"{BASE_DIR}/sample_submission.csv",
    f"{BASE_DIR}/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
)

if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected dataset paths."
    )

sample_sub = pd.read_csv(sample_sub_path)

df_test = sample_sub[["image_id"]].copy()
df_test["label"] = 1  # placeholder; overwritten after inference
test_ids = df_test["image_id"].tolist()



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    test_ids = df_test["image_id"].tolist()
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=int(SIZE * 1.12), width=int(SIZE * 1.12)),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=int(SIZE * 1.12), width=int(SIZE * 1.12)),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(height=int(SIZE * 1.12), width=int(SIZE * 1.12)),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=15, p=1.0),
                A.Resize(height=int(SIZE * 1.12), width=int(SIZE * 1.12)),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
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

        index = torch.randperm(len(labels), device=labels.device)

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
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=labels.device)

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

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
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
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
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 14
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()

    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []
    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 15
def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_head_keys_for_wrapper(state_dict, model_name, net):
    if not isinstance(state_dict, dict):
        return state_dict

    remapped = dict(state_dict)

    for prefix in ["model.", "net.", "module.model.", "module.net."]:
        if any(k.startswith(prefix) for k in remapped.keys()):
            remapped = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in remapped.items()
            }

    if model_name.startswith("resnet") or model_name.startswith("resnext"):
        has_convlayer = any(k.startswith("convlayer.") for k in remapped.keys())
        has_layer = any(
            k.startswith("layer1.") or k.startswith("conv1.") for k in remapped.keys()
        )
        if (not has_convlayer) and has_layer:
            tmp = {}
            for k, v in remapped.items():
                if k.startswith("fc."):
                    tmp[k] = v
                else:
                    tmp["convlayer." + k] = v
            remapped = tmp

        if any(k.startswith("classifier.") for k in remapped.keys()) and not any(
            k.startswith("fc.") for k in remapped.keys()
        ):
            remapped = {
                (
                    ("fc." + k[len("classifier.") :])
                    if k.startswith("classifier.")
                    else k
                ): v
                for k, v in remapped.items()
            }

    if model_name.startswith("densenet"):
        has_convlayer = any(k.startswith("convlayer.") for k in remapped.keys())
        has_features = any(k.startswith("features.") for k in remapped.keys())
        if (not has_convlayer) and has_features:
            tmp = {}
            for k, v in remapped.items():
                if k.startswith("classifier."):
                    tmp["fc." + k[len("classifier.") :]] = v
                else:
                    tmp["convlayer." + k] = v
            remapped = tmp
        else:
            if any(k.startswith("classifier.") for k in remapped.keys()) and not any(
                k.startswith("fc.") for k in remapped.keys()
            ):
                remapped = {
                    (
                        ("fc." + k[len("classifier.") :])
                        if k.startswith("classifier.")
                        else k
                    ): v
                    for k, v in remapped.items()
                }

    if model_name.startswith("efficientnet"):
        if any(k.startswith("_fc.") for k in remapped.keys()) and not any(
            k.startswith("model._fc.") for k in remapped.keys()
        ):
            remapped = {
                (("model." + k) if k.startswith("_fc.") else k): v
                for k, v in remapped.items()
            }
        if any(k.startswith("fc.") for k in remapped.keys()) and not any(
            k.startswith("model._fc.") for k in remapped.keys()
        ):
            remapped = {
                (("model._fc." + k[len("fc.") :]) if k.startswith("fc.") else k): v
                for k, v in remapped.items()
            }
        if any(k.startswith("classifier.") for k in remapped.keys()) and not any(
            k.startswith("model._fc.") for k in remapped.keys()
        ):
            remapped = {
                (
                    ("model._fc." + k[len("classifier.") :])
                    if k.startswith("classifier.")
                    else k
                ): v
                for k, v in remapped.items()
            }

    return remapped


def _try_load_weights(net, MODEL_NAME, pretrained_model_path):
    state = torch.load(pretrained_model_path, map_location="cpu")
    state = _unwrap_state_dict(state)
    state = _strip_module_prefix(state)
    state = _remap_head_keys_for_wrapper(state, MODEL_NAME, net)

    try:
        if MODEL_NAME == "efficientnet-b7":
            missing, unexpected = net.model.load_state_dict(state, strict=False)
        else:
            missing, unexpected = net.load_state_dict(state, strict=False)
    except Exception as e:
        return False, f"Exception during load_state_dict: {e}"

    if isinstance(missing, list):
        if len(missing) > 200:
            return (
                False,
                f"Too many missing keys ({len(missing)}); likely incompatible checkpoint.",
            )
        head_missing = [
            k for k in missing if k.startswith("fc.") or k.startswith("model._fc.")
        ]
        if len(head_missing) >= 2:
            return (
                False,
                f"Head appears missing ({len(head_missing)} keys); likely head name mismatch or incompatible checkpoint.",
            )

    return (
        True,
        f"Loaded with missing={len(missing) if isinstance(missing, list) else 'NA'}, unexpected={len(unexpected) if isinstance(unexpected, list) else 'NA'}",
    )




## === cell 16
def _is_likely_cassava_ckpt(path):
    p = path.lower()
    keywords = ["cassava", "leaf", "disease", "clf", "classification"]
    return any(k in p for k in keywords)


filtered_pretrained_models = [
    p for p in pretrained_models if _is_likely_cassava_ckpt(p)
]
if len(filtered_pretrained_models) > 0:
    print(
        f"Using filtered checkpoints: {len(filtered_pretrained_models)}/{len(pretrained_models)}"
    )
    pretrained_models = filtered_pretrained_models
else:
    print(
        "No cassava-keyword checkpoints found; falling back to all discovered checkpoints."
    )

probability = []
start_time = time.time()

supported_models = 0

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(weights=None)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(weights=None)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        MODEL_NAME = "efficientnet-b7"
        if EfficientNet is None:
            print(f"Skipping {basename} because EfficientNet is unavailable.")
            continue
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        continue

    print(f"{basename}: {MODEL_NAME} ({pretrained_model})")

    ok, msg = _try_load_weights(net, MODEL_NAME, pretrained_model)
    if not ok:
        print(f"Failed to load usable weights for {pretrained_model}: {msg}. Skipping.")
        del net
        torch.cuda.empty_cache()
        continue

    print(f"Checkpoint load: {msg}")
    supported_models += 1

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )
        }
        proba = predict_model(basename, net, dataloader)

        if proba.shape != (len(df_test), num_classes):
            print(
                f"Skipping proba due to shape mismatch: got {proba.shape}, expected {(len(df_test), num_classes)}"
            )
        else:
            probability.append(proba)

    del net
    torch.cuda.empty_cache()

if len(probability) > 0:
    avg_proba = np.mean(np.stack(probability, axis=0), axis=0)  # (N,5)
    df_test["mean"] = avg_proba.argmax(axis=1)
else:
    df_test["mean"] = (
        1  # placeholder; will be replaced by sample_submission labels fallback
    )

print(f"supported_models_used: {supported_models}")
print(f"total usable TTA/model probability arrays: {len(probability)}")
print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
if (len(pretrained_models) == 0) or (supported_models == 0) or (len(probability) == 0):
    df_test["label"] = sample_sub["label"].astype(int)
else:
    if "avg_proba" in globals() and len(avg_proba) == len(df_test) == len(sample_sub):
        df_test["label"] = avg_proba.argmax(axis=1).astype(int)
    else:
        df_test = sample_sub[["image_id"]].copy()
        df_test["label"] = sample_sub["label"].astype(int)

assert list(df_test["image_id"].values) == list(
    sample_sub["image_id"].values
), "image_id order mismatch vs sample_submission.csv"



## === cell 18
df_test.head()



## === cell 19
sub = df_test[["image_id", "label"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
