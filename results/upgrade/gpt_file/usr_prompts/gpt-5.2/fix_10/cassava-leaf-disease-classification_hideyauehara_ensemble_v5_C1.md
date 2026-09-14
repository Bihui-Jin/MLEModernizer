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

0.8943789664551224

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I (1) remove the failing external EfficientNet import and instead rely only on torchvision models to keep the notebook runnable in this environment. I (2) fix the environment/path detection so it consistently finds the Kaggle dataset under `/kaggle/input/cassava-leaf-disease-classification` (and falls back to existing local paths if needed), which unblocks `test_files` creation. I (3) update Albumentations `RandomResizedCrop`/`Rotate` calls to the v2 API so transforms construct correctly, and (4) make the ensembling aggregation robust so `df_test['label']` is always produced even when `pretrained_models` is empty (fallback to sample_submission). These changes are execution-focused and should yield a valid `submission.csv` without changing the intended prediction semantics when model weights are present.'
- What this solution (achieved 0.11584) has done: 'Your current low score is consistent with the code often falling back to `sample_submission.csv` (constant labels) because it doesn’t find/load any pretrained `.pth` files in this environment, and it also uses an incorrect `net(inputs, False, "test")` call that can crash or silently mis-handle model forward signatures. I make minimal changes to (1) discover pretrained weights from the actual mounted dataset paths you have (`/kaggle/input/...` and `/kaggle/data/...`) while keeping the same ensemble logic, and (2) fix `predict_model()` to call the model’s `forward()` correctly in `"test"` mode without passing a fake labels argument. These two fixes should move accuracy substantially upward toward your target without changing architecture, loss, or the ensembling semantics. I also keep the existing safe fallback to `sample_submission.csv` if no weights are available, so it always produces a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89438), and the biggest likely cause is that the code is loading mismatched state_dict keys: the saved checkpoints often include the original classifier head (e.g., `fc.weight`) while your wrapper renames it to `fc.*`, so the head stays randomly initialized and predictions collapse. I keep the same architecture/wrappers/TTAs and only change weight-loading to (1) strip common prefixes (`module.`), (2) remap classifier keys (`fc.`/`classifier.`) onto your wrapper’s `fc.*`, and (3) validate that the loaded head has the right shape; if not, we skip that model rather than averaging garbage. This is a minimal change that keeps your ensemble logic intact but should move accuracy substantially toward the target by actually using the trained classifier weights. I also make the seed settings consistent (deterministic=True implies benchmark=False) to avoid unstable scores across runs without changing evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the most likely issue is that inference is still effectively running with an untrained (random) classifier head for many checkpoints due to a key-mapping bug in `_coerce_state_dict_for_wrapper` (it remaps `fc.*` to itself and doesn’t handle common `convlayer.*`/`features.*` wrapper prefixes). I make a minimal, inference-only change to robustly remap checkpoint keys into your existing wrapper modules (`convlayer.*`/`features.*` + `fc.*`) and still skip incompatible checkpoints rather than averaging garbage. I also ensure we don’t accidentally run stochastic test-time augmentation (`RandomResizedCrop`/`Rotate`) by switching those test transforms to deterministic equivalents while keeping the same TTA structure (multiple views) so accuracy moves up toward your target without changing the model or training logic. The output remains a valid `submission.csv` with `image_id,label` aligned to the test set.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should focus on getting the ensemble to actually use the pretrained checkpoints rather than effectively running with a random/unloaded classifier head. The most likely remaining blocker is the key-remapping logic for checkpoints: for ResNet/ResNeXt you currently force almost everything under `convlayer.*`, which prevents `fc.*` from loading when a checkpoint was saved from a plain torchvision model (`fc.weight`, `fc.bias`) or from your wrapper with different prefixes. I make a minimal inference-only fix to `_coerce_state_dict_for_wrapper` so it correctly maps (a) plain torchvision `fc.*` into wrapper `fc.*`, and (b) backbone weights into `convlayer.*`, without touching training/loss/architecture. I also make the test transforms fully deterministic (replace random crop with center crop) to avoid unnecessary stochasticity during test-time augmentation, which should improve accuracy toward your target without changing the overall TTA/ensemble structure.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest meaningful improvement is to ensure the submission is based on real model predictions rather than a fallback/invalid inference path. I keep your model wrappers, transforms, and ensembling intact, but (1) fix the test-time forward call to match each wrapper’s `forward(inputs, labels, phase)` signature (using a dummy labels tensor so the `phase="test"` branch is reliably taken), and (2) stop skipping EfficientNet-B7 checkpoints by switching to the native `torchvision.models.efficientnet_b7` and wrapping it with your existing `FinalLayerMixupModelEN`. These changes are inference-only, preserve evaluation semantics, and should move accuracy substantially toward your target if those `.pth` files are valid.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to stop “silently” producing near-random predictions caused by input preprocessing mismatches and shape bugs at inference. I keep your ensemble + TTA loop and the same model wrappers, but (1) replace the invalid `CenterCrop(512,512)` (many cassava images are 400x400) with a deterministic `Resize -> CenterCrop` pipeline that matches typical training/inference expectations and avoids runtime/cv2 edge behavior, and (2) fix the `squeeze()` usage in wrappers to only remove the spatial dims (so batch size 1 doesn’t break logits shape). I also add a minimal guard to ensure the submission rows are aligned to `sample_submission.csv` order (by `image_id`), preventing accidental misalignment that can crater accuracy even when predictions are good. These changes are inference-only, preserve the overall logic/semantics, and are the most likely to move accuracy substantially upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest change likely to move accuracy upward is to fix a preprocessing mismatch: your inference normalization uses ImageNet mean/std, but many cassava checkpoints were trained with “0.5/0.5” normalization (common in these public weight packs), which can crater performance even when weights load correctly. I keep your ensemble, wrappers, TTA structure, and checkpoint loading as-is, and only (1) switch test-time normalization to 0.5/0.5 (and keep everything deterministic), and (2) ensure test files are ordered exactly like `sample_submission.csv` to avoid any accidental ordering/alignment issues. These are inference-only changes that preserve core logic and should substantially increase accuracy toward your target.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest change likely to move accuracy up is to fix a key mismatch that still prevents EfficientNet checkpoints from loading their classifier head correctly (your wrapper rewrites `classifier[1]` but the checkpoints commonly save `classifier.1.*`, which you currently don’t map onto `model.classifier.1.*`). I make a minimal, inference-only update to `_coerce_state_dict_for_wrapper` for `wrapper_kind=="efficientnet"` to correctly remap `classifier.1.*` and `classifier.*` keys (and strip common prefixes) so the head is actually loaded. I also add a tiny guard to skip checkpoints that don’t match `num_classes` as you already do, keeping the ensemble behavior unchanged. This should materially increase score toward your target without changing architecture, transforms, or the inference loop.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob(f"../input/densenet201-04-2019data/*.pth")
    + glob.glob(f"../input/resnet152-04-2019data/*.pth")
    + glob.glob(f"../input/eb7-00-baseline/*.pth")
    + glob.glob(f"/kaggle/input/densenet201-04-2019data/*.pth")
    + glob.glob(f"/kaggle/input/resnet152-04-2019data/*.pth")
    + glob.glob(f"/kaggle/input/eb7-00-baseline/*.pth")
    + glob.glob(f"/kaggle/data/densenet201-04-2019data/*.pth")
    + glob.glob(f"/kaggle/data/resnet152-04-2019data/*.pth")
    + glob.glob(f"/kaggle/data/eb7-00-baseline/*.pth")
)

pretrained_models = list(dict.fromkeys(sorted(pretrained_models)))

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(pretrained_models))



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
EfficientNet = None



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
cand_base_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "data",
]

BASE_DIR = None
for p in cand_base_dirs:
    if os.path.exists(p):
        if p == "data" and os.path.exists(
            os.path.join(p, "cassava-leaf-disease-classification")
        ):
            p = os.path.join(p, "cassava-leaf-disease-classification")
        BASE_DIR = p
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not find cassava dataset directory in known locations."
    )

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")

if run_type == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = os.listdir(TEST_PATH)[:320]
else:
    print("In Kaggle/local environment (Batch/default).")
    TEST_PATH = f"{BASE_DIR}/test_images"
    test_files = os.listdir(TEST_PATH)

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
sample_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
test_set = set(test_files)
ordered_test_files = [img for img in sample_sub["image_id"].tolist() if img in test_set]
if len(ordered_test_files) != len(test_files):
    ordered_test_files = test_files

df_test = pd.DataFrame(ordered_test_files, columns=["image_id"])
df_test["label"] = 1



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = {
    "test": [
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(
                    height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR, p=1.0
                ),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Rotate(limit=30, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 10
pass




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze(-1).squeeze(-1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze(-1).squeeze(-1)
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
        mixed_x = mixed_x.squeeze(-1).squeeze(-1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
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
            x = x.squeeze(-1).squeeze(-1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze(-1).squeeze(-1)
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

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze(-1).squeeze(-1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 13
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

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
        import sys as _sys

        _sys.exit()




## === cell 14
pass




## === cell 15
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
            raise FileNotFoundError(
                f"Image not found or unreadable: {TEST_PATH}/{image_id}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
def _coerce_state_dict_for_wrapper(state_dict, wrapper_kind):
    """
    Change (score): fix key mapping so checkpoints saved from *plain torchvision* models load correctly.
    Critical fix here: EfficientNet checkpoints often store head as 'classifier.1.weight'/'classifier.1.bias'
    (or with 'module.'/'model.' prefixes). Our wrapper expects 'model.classifier.1.*', so without this remap
    the head stays random and accuracy collapses. We remap those keys while keeping the rest of the logic intact.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    remapped = {}

    if wrapper_kind in ("resnet", "resnext"):
        for k, v in cleaned.items():
            nk = k
            if nk.startswith("fc."):
                remapped[nk] = v
                continue
            if nk.startswith(("convlayer.", "fc.")):
                remapped[nk] = v
                continue
            remapped["convlayer." + nk] = v

    elif wrapper_kind == "densenet":
        for k, v in cleaned.items():
            nk = k
            if nk.startswith("classifier."):
                nk = "fc." + nk[len("classifier.") :]
                remapped[nk] = v
                continue
            if nk.startswith("fc."):
                remapped[nk] = v
                continue
            if nk.startswith(("convlayer.", "fc.")):
                remapped[nk] = v
                continue
            if nk.startswith("features."):
                nk = "convlayer." + nk[len("features.") :]
            else:
                nk = "convlayer." + nk
            remapped[nk] = v

    elif wrapper_kind == "efficientnet":
        for k, v in cleaned.items():
            nk = k

            if nk.startswith("classifier.1."):
                nk = "model." + nk  # -> model.classifier.1.*
                remapped[nk] = v
                continue

            if nk.startswith("model."):
                remapped[nk] = v
                continue

            remapped["model." + nk] = v

    else:
        remapped = cleaned

    return remapped


def _load_checkpoint_into_net(net, ckpt_path, wrapper_kind, num_classes):
    """
    Keep the same behavior: allow non-strict load but require that the wrapper fc head loads with correct shape.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _coerce_state_dict_for_wrapper(raw, wrapper_kind)

    if wrapper_kind == "efficientnet":
        fc_w = sd.get("model.classifier.1.weight", None)
        fc_b = sd.get("model.classifier.1.bias", None)
        head_missing_names = ("model.classifier.1.weight", "model.classifier.1.bias")
    else:
        fc_w = sd.get("fc.weight", None)
        fc_b = sd.get("fc.bias", None)
        head_missing_names = ("fc.weight", "fc.bias")

    if fc_w is None or fc_b is None:
        return False, f"checkpoint missing head params {head_missing_names} after remap"
    if tuple(fc_w.shape)[0] != num_classes or tuple(fc_b.shape)[0] != num_classes:
        return (
            False,
            f"head shape mismatch: w{tuple(fc_w.shape)} b{tuple(fc_b.shape)}",
        )

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if any(n in missing for n in head_missing_names):
        return (
            False,
            f"head params still missing after load_state_dict (missing={missing})",
        )
    return True, f"loaded (missing={len(missing)} unexpected={len(unexpected)})"


def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")

        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            dummy_labels = torch.zeros(
                (inputs.size(0),), dtype=torch.long, device=inputs.device
            )

            outputs = net(inputs, dummy_labels, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained models found; falling back to sample_submission.csv labels as a valid submission."
    )
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        wrapper_kind = None
        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
            wrapper_kind = "resnet"
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
            wrapper_kind = "resnet"
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
            wrapper_kind = "resnet"
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
            wrapper_kind = "resnext"
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
            wrapper_kind = "densenet"
        elif "efficientnet-b7" in basename or "eb7" in basename:
            MODEL_NAME = "efficientnet_b7"
            net = models.efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 8
            wrapper_kind = "efficientnet"
        else:
            print(f"{basename} is not supported; skipping.")
            continue

        print(f"{basename}: {MODEL_NAME}")

        ok, msg = _load_checkpoint_into_net(
            net, pretrained_model, wrapper_kind, num_classes
        )
        print(f"{basename} load: {msg}")
        if not ok:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

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
                    pin_memory=True,
                )
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "All models were skipped or failed to load; falling back to sample_submission.csv."
        )
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    else:
        proba_mean = np.mean(np.stack(probability, axis=0), axis=0)  # (N, 5)
        df_test["label"] = proba_mean.argmax(axis=1).astype(int)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")



## === cell 19
sample_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
df_out = sample_sub[["image_id"]].merge(
    df_test[["image_id", "label"]], on="image_id", how="left"
)
if df_out["label"].isna().any():
    df_out["label"] = sample_sub["label"].values
df_out["label"] = df_out["label"].astype(int)

df_out[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out[["image_id", "label"]].shape)
