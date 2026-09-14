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

0.8961922030825022

# 6. Current score

0.18685

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17825) has done: 'I fix the test image path issue caused by nested `test_images/test_images` folders by auto-detecting the actual directory that contains `.jpg` files and filtering out subdirectory names from `os.listdir`. This remove the `FileNotFoundError` in the DataLoader and ensure predictions run to completion so the `mean` column is created. Then I enforce that the submission uses the exact `image_id` ordering/length from `sample_submission.csv` (and only fills `label`), which fixes the “same length as the answers” invalid submission error. These changes are score-neutral (they mainly fix I/O and alignment) and keep the model/inference logic intact.'
- What this solution (achieved 0.18535) has done: 'Your current score (0.17825) is far below the target (0.89619), so we should increase accuracy; the biggest minimal-risk issue is that your inference wrapper (`FinalLayerMixupModelEN`) replaces the EfficientNet classifier but never loads weights for that new layer, so any `.pth` weights for EfficientNet won’t match and you end up with a mostly-random head (and/or silently skipped keys with `strict=False`). I minimally change the EfficientNet wrapper to keep the original model architecture intact and instead *only* adapt incoming checkpoints by mapping common key names (`fc.*` -> `classifier.1.*`) and removing `module.` prefixes before loading, so real trained weights land in the correct place. I also disable augmentation randomness during test-time by making the second TTA transform deterministic (keep the same two-pass structure but replace random crop/flip with resize/center-crop), which stabilizes and typically improves accuracy without changing your ensemble logic. Everything else (model choices, averaging, argmax, submission alignment) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is far below the target, so we should increase accuracy with minimal risk by fixing the most likely remaining cause of near-random predictions: some checkpoints may still not be loading into your wrapper modules due to mismatched key prefixes (e.g., weights saved for the *base* model without the `model.` prefix used inside your wrapper, or saved for the wrapper `fc` head rather than `model.classifier.1`). I extend the state-dict remapping to (1) automatically add the wrapper prefix when needed, and (2) handle EfficientNet head keys in both directions (`fc.*`, `classifier.1.*`, and `model.classifier.1.*`), while keeping the exact same model/inference flow. I also enforce `CenterCrop` safety by adding a preceding `Resize` in the first TTA transform (otherwise images smaller than 512 can be cropped incorrectly), which is a small, semantics-preserving preprocessing fix that typically boosts accuracy. Everything else (ensembling, TTA count=2, argmax, submission alignment/format) stays unchanged and still writes `submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is far below the target (0.18685 vs 0.89619), so we should increase accuracy with the smallest changes that address likely “random-head / wrong-weights-loaded” failure modes while preserving your ensemble/TTA flow. I (1) make the EfficientNet wrapper compatible with checkpoints that were trained with an external `fc` head by adding an `fc` layer and using it only when the checkpoint provides those weights (otherwise keep using `model.classifier[1]`), and (2) broaden the state-dict key remapping for EfficientNet to correctly route `fc.*`, `classifier.*`, and `model.*` keys regardless of how the checkpoint was saved. This keeps the same model family, forward semantics, averaging, and argmax submission logic, but should load the intended trained classifier weights instead of leaving a random head. The script still runs end-to-end and writes a valid `submission.csv` with the correct ordering from `sample_submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is far below the target (0.18685 vs 0.89619), so we should increase accuracy with the smallest changes that address likely “wrong head used / wrong weights landed” issues at inference. I (1) fix the EfficientNet wrapper so it can correctly consume checkpoints saved either as the wrapper (`model.*`) or as a plain torchvision EfficientNet (`features.*`, `classifier.*`) by automatically routing those keys into `net.model.*`, and (2) ensure `fc.*` weights (if present) are also accepted whether saved as `fc.*` or `model.fc.*` so the intended trained head is actually used. These are minimal, inference-only compatibility changes (no training changes) and keep the ensemble/TTA/argmax logic identical while making weight loading effective. The script still run end-to-end and write a valid `submission.csv` in the sample submission order.'
- What this solution (achieved 0.18685) has done: 'Your current score (0.18685) is far below the target (0.89619), so we should increase accuracy by fixing the most likely remaining “weights not actually loaded” failure mode without changing your ensemble/TTA flow. The minimal, high-impact fix is to make state-dict loading robust for **all** model wrappers: for ResNet/DenseNet your wrapper parameters are named `fc.*` but many checkpoints save `model.fc.*` or the base model’s `fc.*`, so strict loading silently misses the head and yields near-random outputs. I add small, model-specific key remapping for ResNet/DenseNet (and keep your existing EfficientNet remapping), then enforce `net.eval()` + disable train-time randomness consistently (no architecture/training changes). The script still runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is extremely low vs the target, which strongly suggests most checkpoints are still not actually being used (state-dict keys don’t match your wrapper modules, so the trained heads/backbones don’t load and predictions become near-random). I make one minimal, inference-only fix: extend the state-dict remapping so it correctly routes **base-model keys** (e.g., `features.*`, `classifier.*`, `fc.*`) into your wrapper namespaces for **all three wrappers** (EfficientNet/ResNet/DenseNet), and then enforce `strict=True` after remap (only falling back to `strict=False` if still necessary). This preserves your exact model architectures, TTA(2) loop, averaging, and argmax submission logic; it only increases the likelihood that the intended trained weights are actually loaded. The script still run end-to-end and write `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is far below the target, so we should increase accuracy by ensuring your ensemble checkpoints actually load into the wrapper models (otherwise you effectively predict with random heads and get ~0.2 accuracy). I make one minimal, inference-only change: after remapping keys, load with `strict=True` and explicitly validate that a large fraction of keys matched; if not, try a small set of additional, safe prefix-remap candidates (e.g., shifting between `model.` / no-prefix / wrapper prefixes like `convlayer.` and `features.`) and pick the one with the best match rate. This preserves your exact model architectures, forward logic, TTA(2), softmax averaging, and argmax submission, but greatly increases the chance that the intended trained weights are actually used. The script still run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.18685) has done: 'Your score is far below the target (0.18685 vs 0.89619), which is consistent with checkpoints still not loading into the wrapper models correctly (so you’re effectively using random weights). I make a minimal inference-only change: strengthen state-dict remapping and selection by trying a few additional safe key translations (especially for the common “trained as plain torchvision model” vs “trained as wrapper with convlayer/features prefixes” cases) and then choosing the variant with the best match ratio before loading. I also make the loader print the match ratio for transparency and only fall back to non-strict loading when strict loading cannot work. This keeps your architecture, TTA=2 averaging, and argmax submission semantics identical, but increases the chance that the intended trained weights are actually used.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob(f"../input/densenet201-04-2019data/*.pth")
    + glob.glob(f"../input/eb7m-seed7x/efficientnet-b7m_SEED70/*.pth")
    + glob.glob(f"../input/eb7m-seed7x/efficientnet-b7m_SEED72/*.pth")
    + glob.glob(f"../input/eb7m-seed7x/efficientnet-b7m_SEED73/*.pth")
    + glob.glob(f"../input/eb7m-seed7x/efficientnet-b7sl_SEED71.pretrained/*.pth")
)

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)))



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
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _find_image_dir(base_dir: str, split: str) -> str:
    """
    Return a directory path that contains jpg images for the given split.
    Tries common nesting patterns seen in Kaggle inputs.
    """
    candidates = [
        os.path.join(base_dir, f"{split}_images"),
        os.path.join(base_dir, f"{split}_images", f"{split}_images"),
        os.path.join(base_dir, split),
        os.path.join(base_dir, split, f"{split}_images"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            try:
                files = os.listdir(p)
            except Exception:
                continue
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return p
    if os.path.isdir(base_dir):
        try:
            files = os.listdir(base_dir)
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return base_dir
        except Exception:
            pass
    return candidates[0]


KAGGLE_BASE = "/kaggle/input/cassava-leaf-disease-classification"
if os.path.isdir(KAGGLE_BASE):
    BASE_DIR = KAGGLE_BASE
    run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    if run_type == "Interactive":
        print("Test run in Kaggle environment (Interactive).")
        TEST_PATH = _find_image_dir(BASE_DIR, "train")
        all_files = os.listdir(TEST_PATH)
        test_files = sorted([f for f in all_files if f.lower().endswith(".jpg")])[:32]
    else:
        print("In Kaggle environment (Batch/default).")
        TEST_PATH = _find_image_dir(BASE_DIR, "test")
        all_files = os.listdir(TEST_PATH)
        test_files = sorted([f for f in all_files if f.lower().endswith(".jpg")])
else:
    print("In the local environment.")
    BASE_DIR = "data"
    if os.path.isdir(f"{BASE_DIR}/test_images") or os.path.isdir(f"{BASE_DIR}/test"):
        TEST_PATH = _find_image_dir(BASE_DIR, "test")
        all_files = os.listdir(TEST_PATH)
        test_files = sorted([f for f in all_files if f.lower().endswith(".jpg")])
    else:
        TEST_PATH = _find_image_dir(BASE_DIR, "train")
        all_files = os.listdir(TEST_PATH)
        test_files = sorted([f for f in all_files if f.lower().endswith(".jpg")])[:32]

print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
df_test = pd.read_csv(sample_sub_path)
df_test["label"] = 1  # dummy init; overwritten later
print("Loaded sample_submission:", df_test.shape)



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE, p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE, p=1.0),
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
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
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
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
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
        mixed_x = mixed_x.squeeze()
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
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.fc = nn.Linear(num_ftrs, num_classes)  # optional alt head
        self.criterion = criterion
        self._use_fc_head = False  # set True if checkpoint provides fc.* weights

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self._forward_logits(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self._forward_logits(inputs)
            return outputs

        print("ここにきてはいけない")
        raise SystemExit(1)

    def _forward_logits(self, inputs):
        if not self._use_fc_head:
            return self.model(inputs)
        x = self.model.features(inputs)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier[0](x)  # dropout
        return self.fc(x)




## === cell 13
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
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
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)

            if device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = net(inputs, None, "test")
            else:
                outputs = net(inputs, None, "test")

            probability.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 15
def _cleanup_and_remap_state_dict_for_model(
    sd: dict, model_name: str, net: nn.Module = None
) -> dict:
    """
    Change rationale (score-improvement, minimal semantic change):
    - Very low accuracy strongly suggests that checkpoint keys don't match wrapper keys,
      so most weights are skipped. We keep your wrappers/forward semantics identical and
      only normalize/remap key names to improve load success.
    """
    if not isinstance(sd, dict):
        return sd

    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    remap = {}

    if "efficientnet" in model_name:
        for k, v in sd.items():
            nk = k

            if not nk.startswith("model."):
                if nk.startswith(("features.", "classifier.", "avgpool.")):
                    nk = "model." + nk

            if nk.startswith("model.fc."):
                nk = nk.replace("model.fc.", "fc.", 1)

            if nk in ("classifier.1.weight", "classifier.1.bias"):
                nk = "model." + nk

            remap[nk] = v
        return remap

    if "resnet" in model_name or "resnext" in model_name:
        for k, v in sd.items():
            nk = k

            if nk.startswith("model."):
                nk = nk.replace("model.", "", 1)

            if nk.startswith(
                ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
            ):
                nk = "convlayer." + nk
            elif nk.startswith("fc."):
                pass
            elif nk.startswith("convlayer.") or nk.startswith("fc."):
                pass
            else:
                nk = "convlayer." + nk

            if nk.startswith("convlayer.fc."):
                nk = nk.replace("convlayer.fc.", "fc.", 1)

            remap[nk] = v
        return remap

    if "densenet" in model_name:
        for k, v in sd.items():
            nk = k

            if nk.startswith("model."):
                nk = nk.replace("model.", "", 1)

            if nk.startswith("classifier."):
                nk = nk.replace("classifier.", "fc.", 1)

            if nk.startswith("convlayer."):
                nk = nk.replace("convlayer.", "features.", 1)

            remap[nk] = v
        return remap

    return sd


def _state_dict_match_report(net: nn.Module, sd: dict):
    model_keys = set(net.state_dict().keys())
    sd_keys = set(sd.keys())
    matched = model_keys & sd_keys
    missing = model_keys - sd_keys
    unexpected = sd_keys - model_keys
    match_ratio = (len(matched) / max(1, len(model_keys))) * 100.0
    return match_ratio, len(matched), len(missing), len(unexpected)


def _try_load_with_prefix_variants(net: nn.Module, sd_in: dict, model_name: str):
    """
    Change rationale (score-improvement, minimal semantic change):
    - Try a small set of deterministic key-translation variants that commonly occur when
      saving either the base torchvision model or the wrapper model.
    - Choose variant with the highest key match ratio, then load strict if possible
      (fall back to non-strict only if strict fails).
    """
    if not isinstance(sd_in, dict):
        return False, "non-dict", 0.0

    candidates = []

    def add_candidate(name, sd):
        if isinstance(sd, dict) and len(sd) > 0:
            candidates.append((name, sd))

    add_candidate("as_is", sd_in)

    if any(k.startswith("model.") for k in sd_in.keys()):
        add_candidate(
            "strip_model_prefix",
            {k.replace("model.", "", 1): v for k, v in sd_in.items()},
        )

    add_candidate(
        "add_model_prefix",
        {
            ("model." + k) if not k.startswith("model.") else k: v
            for k, v in sd_in.items()
        },
    )

    if ("resnet" in model_name) or ("resnext" in model_name):
        sd_to_wrapper = {}
        sd_strip_wrapper = {}
        for k, v in sd_in.items():
            nk = k.replace("model.", "", 1) if k.startswith("model.") else k
            if nk.startswith(
                ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
            ):
                sd_to_wrapper["convlayer." + nk] = v
            elif nk.startswith("fc."):
                sd_to_wrapper["fc." + nk.split("fc.", 1)[1]] = v
            else:
                sd_to_wrapper[nk] = v
            if nk.startswith("convlayer."):
                sd_strip_wrapper[nk.replace("convlayer.", "", 1)] = v
            else:
                sd_strip_wrapper[nk] = v
        add_candidate("resnet_to_wrapper", sd_to_wrapper)
        add_candidate("resnet_strip_convlayer", sd_strip_wrapper)

    if "densenet" in model_name:
        sd_dn_to_wrapper = {}
        sd_dn_strip_features = {}
        for k, v in sd_in.items():
            nk = k.replace("model.", "", 1) if k.startswith("model.") else k
            if nk.startswith("classifier."):
                sd_dn_to_wrapper[nk.replace("classifier.", "fc.", 1)] = v
            else:
                sd_dn_to_wrapper[nk] = v
            if nk.startswith("features."):
                sd_dn_strip_features[nk.replace("features.", "", 1)] = v
            else:
                sd_dn_strip_features[nk] = v
        add_candidate("densenet_to_wrapper", sd_dn_to_wrapper)
        add_candidate("densenet_strip_features_prefix", sd_dn_strip_features)

    if "efficientnet" in model_name and any(
        k.startswith("model.model.") for k in sd_in.keys()
    ):
        add_candidate(
            "strip_model_model_prefix",
            {k.replace("model.model.", "model.", 1): v for k, v in sd_in.items()},
        )

    best = None
    best_rep = None
    for name, sd in candidates:
        ratio, m, miss, unexp = _state_dict_match_report(net, sd)
        if (best is None) or (ratio > best_rep[0]):
            best = (name, sd)
            best_rep = (ratio, m, miss, unexp)

    best_name, best_sd = best
    best_ratio, m, miss, unexp = best_rep
    print(
        f"[load-check] best_variant={best_name} match={best_ratio:.1f}% matched={m} missing={miss} unexpected={unexp}"
    )

    try:
        net.load_state_dict(best_sd, strict=True)
        return True, f"strict:{best_name}", best_ratio
    except Exception:
        net.load_state_dict(best_sd, strict=False)
        return True, f"nonstrict:{best_name}", best_ratio


probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    pretrained_models = ["__fallback_efficientnet_b7_imagenet__"]

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    if basename == "":
        basename = str(pretrained_model)

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
    elif ("efficientnet-b7" in basename) or (
        pretrained_model == "__fallback_efficientnet_b7_imagenet__"
    ):
        MODEL_NAME = "efficientnet-b7"
        if pretrained_model == "__fallback_efficientnet_b7_imagenet__":
            from torchvision.models import EfficientNet_B7_Weights

            base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        else:
            base = efficientnet_b7(weights=None)
        net = FinalLayerMixupModelEN(base, criterion, num_classes, False)
        BATCH_SIZE = 10
    else:
        print(f"{basename} is not supported.")
        raise SystemExit(1)

    print(f"{basename}: {MODEL_NAME}")

    if pretrained_model != "__fallback_efficientnet_b7_imagenet__":
        sd_raw = torch.load(pretrained_model, map_location="cpu")
        sd = _cleanup_and_remap_state_dict_for_model(sd_raw, MODEL_NAME, net=net)

        if isinstance(net, FinalLayerMixupModelEN):
            has_fc = any(k in sd for k in ("fc.weight", "fc.bias"))
            net._use_fc_head = bool(has_fc)

        ok, how, ratio = _try_load_with_prefix_variants(net, sd, MODEL_NAME)
        if not ok or ratio < 50.0:
            print(
                f"[warning] Low state-dict match ratio ({ratio:.1f}%) for {basename}. "
                f"This likely harms accuracy; check checkpoint/model mismatch. Load mode: {how}"
            )

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
                pin_memory=(device == "cuda"),
            )
        }

        proba = predict_model(basename, net, dataloader)  # (N, 5)
        probability.append(proba.astype(np.float32))

    del net
    if device == "cuda":
        torch.cuda.empty_cache()

prob_arr = np.stack(probability, axis=0)
mean_proba = prob_arr.mean(axis=0)
df_test["mean"] = mean_proba.argmax(axis=1).astype(int)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 16
df_test["label"] = df_test["mean"].astype(int)
df_test = df_test[["image_id", "label"]]
df_test.head()



## === cell 17
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
