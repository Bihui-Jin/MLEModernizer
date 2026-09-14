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

0.8946811725596857

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing EfficientNet dependency by switching the EfficientNet branch to use torchvision’s built-in `efficientnet_b7` (same family, no external package), and load weights robustly with `map_location` and optional key-prefix cleanup. I also make the path selection always point to the real Kaggle dataset location (`/kaggle/input/...`) so `test_files` is defined and images are found. Albumentations v2 changed `RandomResizedCrop`’s API, so I update those calls to the new signature to remove the validation error. Finally, I make the ensemble aggregation stable even if no models are found (fallback to sample_submission) and fix the shape handling that caused the `AxisError`, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'Your low score is consistent with the ensemble not actually using the intended pretrained checkpoints (so predictions are effectively near-random), mostly because the model glob paths point to datasets that likely don’t exist in your environment. I make the checkpoint discovery robust by also searching the actual cassava dataset directory (and common Kaggle input roots) for `.pth` files, while keeping the exact same inference/ensemble logic. I also fix a subtle shape bug in the ResNet/DenseNet heads (`squeeze()` can drop the batch dimension when batch_size=1), which can silently break logits/probabilities and hurt accuracy. These are minimal changes that keep the same models, transforms, and averaging semantics, but ensure the code loads real weights and produces well-formed probabilities.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89468), and the most likely cause is that the inference head weights are not being restored correctly: the wrapper classes create a new `fc`/`classifier` layer but the checkpoint keys typically belong to the original backbone’s head (e.g., `fc.*`, `classifier.*`), so the loaded weights don’t land in the active head. I keep the same architectures/wrappers/inference logic, but minimally extend the checkpoint loader to remap common head keys (e.g., `fc.*` -> `fc.*` in wrapper, or `fc.*` -> `model.classifier.1.*` for torchvision EfficientNet-B7) so the intended trained head weights actually get used. I also make the basename matching a bit more robust (case-insensitive, accept `efficientnet_b7` / `eb7`) to avoid silently exiting or skipping correct checkpoints. These changes should significantly raise accuracy toward your target without changing the model family, transforms, ensemble averaging, or loss.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, and the most likely reason is that you’re loading checkpoints into wrapper models whose head parameter names don’t match the checkpoint head keys, so the trained classifier head isn’t actually restored. I make a minimal, targeted fix: in each wrapper, alias the newly-created head layer to the original backbone’s expected attribute name (`model.fc` / `model.classifier`) so `load_state_dict` lands weights into the active layer without changing architecture or inference semantics. I also make `load_checkpoint_flexible` explicitly remap common wrapper head keys (`fc.*`) to EfficientNet’s `model.classifier.1.*` when needed, again without changing the model family or ensembling logic. These changes should move accuracy sharply upward toward your target while preserving your core pipeline.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low for this competition, which is most consistent with the classifier heads not matching what the checkpoints expect (so the trained head weights are silently not applied). I make a minimal change to the wrapper models so their head attribute names match the original torchvision backbones (`model.fc` / `model.classifier`), letting your existing `load_checkpoint_flexible(..., strict=False)` correctly load the trained head without changing architecture or inference semantics. I also make `predict_model` call EfficientNet correctly (it currently passes extra args to a module whose forward only accepts `inputs`), which can otherwise break or degrade predictions. These are small, targeted fixes intended to move accuracy sharply upward toward your target while keeping the rest of your ensembling and TTA unchanged.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, which is most consistent with inference not actually using the trained heads from the checkpoints (so predictions are effectively random). I make a minimal fix to the wrapper models so their classifier head attribute names match what torchvision backbones (and thus most checkpoints) expect, allowing `load_state_dict(..., strict=False)` to correctly restore the trained head weights. I also make `predict_model` call the model in a way that works for both wrapper types without changing ensembling/TTA semantics. These changes keep the same architectures, transforms, averaging, and submission format, but should move accuracy sharply upward toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os


def find_pretrained_models():
    patterns = []
    patterns += ["../input/resnet152-04-2019data/*.pth", "../input/eb7m-seed70/*.pth"]

    patterns += [
        "/kaggle/input/*/*.pth",
        "/kaggle/input/*/*/*.pth",
        "/kaggle/input/*/*/*/*.pth",
    ]

    patterns += [
        "/kaggle/input/cassava-leaf-disease-classification/**/*.pth",
        "/kaggle/input/cassava-leaf-disease-classification/**/**/*.pth",
    ]

    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))

    found = sorted(list(dict.fromkeys(found)))

    filtered = []
    for p in found:
        try:
            if os.path.getsize(p) >= 50_000:  # 50KB
                filtered.append(p)
        except OSError:
            pass

    return filtered


pretrained_models = find_pretrained_models()

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)[:50]))
    if len(pretrained_models) > 50:
        print(f"... ({len(pretrained_models)-50} more)")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms
from torchvision.models import efficientnet_b7 as tv_efficientnet_b7
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
EfficientNet = None  # kept only to preserve references; not used.



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
KAGGLE_BASE = "/kaggle/input/cassava-leaf-disease-classification"
LEGACY_BASE = "../input/cassava-leaf-disease-classification"

if os.path.isdir(KAGGLE_BASE):
    BASE_DIR = KAGGLE_BASE
elif os.path.isdir(LEGACY_BASE):
    BASE_DIR = LEGACY_BASE
else:
    BASE_DIR = "data"

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
if run_type == "Interactive":
    print("Test run in Kaggle environment (Interactive).")
    TEST_PATH = f"{BASE_DIR}/train_images"
    test_files = os.listdir(TEST_PATH)[:320]
elif run_type == "Batch":
    print("In Kaggle environment (Batch).")
    TEST_PATH = f"{BASE_DIR}/test_images"
    test_files = os.listdir(TEST_PATH)
else:
    if os.path.isdir(f"{BASE_DIR}/test_images"):
        print("KAGGLE_KERNEL_RUN_TYPE unset; using test_images.")
        TEST_PATH = f"{BASE_DIR}/test_images"
        test_files = os.listdir(TEST_PATH)
    else:
        print("In the local environment fallback.")
        TEST_PATH = f"{BASE_DIR}/train_images"
        test_files = os.listdir(TEST_PATH)[:320]

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1



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
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
def load_checkpoint_flexible(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state)} at {ckpt_path}")

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        cleaned[nk] = v

    model_keys = set(model.state_dict().keys())

    remapped = dict(cleaned)
    if any(k.startswith("model.classifier.") for k in model_keys):
        for k, v in cleaned.items():
            if k in ("fc.weight", "fc.bias"):
                dst = (
                    "model.classifier.1.weight"
                    if k.endswith("weight")
                    else "model.classifier.1.bias"
                )
                if dst in model_keys:
                    remapped[dst] = v
            if k.startswith("classifier.") and ("model." + k) in model_keys:
                remapped["model." + k] = v

    for k, v in cleaned.items():
        if k.startswith("classifier.") and (
            "fc.weight" in model_keys or "fc.bias" in model_keys
        ):
            if k == "classifier.weight":
                remapped["fc.weight"] = v
            elif k == "classifier.bias":
                remapped["fc.bias"] = v

    if any(k.startswith("fc.") for k in model_keys):
        for k, v in cleaned.items():
            if k in ("classifier.weight", "classifier.bias"):
                dst = "fc.weight" if k.endswith("weight") else "fc.bias"
                if dst in model_keys:
                    remapped[dst] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if len(missing) > 0:
        print(
            f"[WARN] Missing keys when loading {os.path.basename(ckpt_path)}: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    if len(unexpected) > 0:
        print(
            f"[WARN] Unexpected keys when loading {os.path.basename(ckpt_path)}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )
    return model




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features

        self.fc = nn.Linear(num_ftrs, num_classes)

        self.model = model
        self.model.fc = self.fc

        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)
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
        mixed_x = torch.flatten(mixed_x, 1)
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

        self.model = model
        self.model.classifier = self.fc

        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.model.classifier(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)
            outputs = self.model.classifier(x)
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
        mixed_x = torch.flatten(mixed_x, 1)
        outputs = self.model.classifier(mixed_x)

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

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported EfficientNet model head structure.")

        self.model = model
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        import sys

        sys.exit()




## === cell 14
assert os.path.isdir(TEST_PATH), f"TEST_PATH does not exist: {TEST_PATH}"




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

            if isinstance(
                net,
                (
                    FinalLayerMixupModel,
                    FinalLayerMixupModelDenseNet,
                    FinalLayerMixupModelEN,
                ),
            ):
                dummy_labels = torch.zeros(
                    inputs.size(0), dtype=torch.long, device=inputs.device
                )
                outputs = net(inputs, dummy_labels, "test")
            else:
                outputs = net(inputs)

            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
probability = []

start_time = time.time()

if len(pretrained_models) == 0:
    print("[WARN] No pretrained models found. Writing sample_submission as fallback.")
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        basename_l = basename.lower()

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename_l:
            MODEL_NAME = "resnet18"
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename_l:
            MODEL_NAME = "resnet50"
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename_l:
            MODEL_NAME = "resnet152"
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename_l:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename_l:
            MODEL_NAME = "densenet201"
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif (
            ("efficientnet-b7" in basename_l)
            or ("efficientnet_b7" in basename_l)
            or ("eb7" in basename_l)
        ):
            MODEL_NAME = "efficientnet-b7"
            net = tv_efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            import sys

            sys.exit()

        print(f"{basename}: {MODEL_NAME}")
        print(f"Loading checkpoint: {pretrained_model}")

        net = load_checkpoint_flexible(net, pretrained_model, device)

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

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prob_arr = np.stack(probability, axis=0)
    df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    if "mean" in df_test.columns:
        df_test["label"] = df_test["mean"]



## === cell 19
df_test



## === cell 20
sub = df_test[["image_id", "label"]].copy()
sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
