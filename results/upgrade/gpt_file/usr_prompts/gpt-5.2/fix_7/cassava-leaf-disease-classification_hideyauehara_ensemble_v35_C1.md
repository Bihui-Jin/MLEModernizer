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

- What this solution (achieved 0.11584) has done: 'I make the notebook run end-to-end by (1) removing the unavailable `efficientnet_pytorch` dependency and safely skipping EfficientNet-B7 weights if present, (2) fixing path detection so it always finds the Kaggle `/kaggle/input/cassava-leaf-disease-classification` images, and (3) updating Albumentations `RandomResizedCrop` to the v2 API to stop validation errors. I also make the ensemble aggregation robust (so `probability` always has the expected shape) and ensure a valid `submission.csv` is always written using `sample_submission.csv` for ordering. These are execution/format fixes and should improve score vs “no submission”, without changing the core ensemble/TTA inference logic beyond necessary compatibility.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low because the notebook is not actually finding/using the intended pretrained weights in your current environment (so predictions degrade to an almost-constant fallback), and because the current “test set” discovery can silently point at the wrong folder when paths differ. I make minimal, execution-safe changes to (1) locate pretrained `.pth` files robustly under the available `/kaggle/input` and `/kaggle/data` trees, (2) always use `sample_submission.csv` to define the exact test ordering while still reading images from the real `test_images/` directory, and (3) load checkpoints more safely by handling common key-prefix patterns (`module.`, `model.`, etc.) without changing the model architecture or inference logic. These changes should move accuracy upward toward your target by ensuring the ensemble is actually applied to the right test set in the right order, without altering the core model/ensemble/TTA semantics.'
- What this solution (achieved 0.11584) has done: 'Your score is low mainly because the notebook is ensembling *every* `.pth` it can find (including unrelated/incorrect checkpoints) and loading them with `strict=False`, which often yields near-random predictions that get averaged together. I make a minimal, score-directed change to only use checkpoints that match this solution’s supported architectures by filename, and I add a lightweight compatibility filter that skips checkpoints whose tensors don’t match the current model’s parameter shapes (instead of partially loading them). This preserves your core inference/ensemble/TTA logic, but prevents “bad models” from poisoning the average, which should move accuracy upward toward your target. I also keep the sample-submission ordering exactly as you already do and still always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score suggests the ensemble is effectively not using the intended trained weights (or is skipping most of them), so predictions collapse toward a near-constant class. To move accuracy upward toward the target with minimal logic change, I (1) load checkpoints more robustly by handling nested keys and common prefix patterns, and (2) avoid partially-loaded models by requiring a high overlap of matching tensor shapes before using a checkpoint (so “bad” checkpoints don’t poison the mean). I also enforce a deterministic, correct test ordering by always building predictions in the exact `sample_submission.csv` order and add a hard check that every test image exists (fail fast instead of silently degrading). The model architectures, TTA set, and averaging logic stay the same; this only fixes weight usage/alignment issues that are currently causing the low score.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.89468), so we should improve accuracy substantially while keeping your ensemble/TTA inference logic intact. The biggest issue is that your checkpoint compatibility check is overly strict: it requires ~80% of *all* model parameters to match, which almost always fails because your wrapper replaces the final classifier (`fc`/`classifier`) so those keys/shapes won’t match the checkpoint, causing most weights to be skipped and triggering the near-constant fallback. I relax compatibility to ignore classifier-head keys during overlap computation (still enforcing exact shape matches for the backbone) so real checkpoints load and contribute, and I add a small safeguard for `x.squeeze()` potentially dropping the batch dimension when batch_size==1 (a rare but harmful edge case). These are minimal changes that preserve your architecture, transforms, and averaging semantics, but should move the score sharply upward toward the target by actually using the intended pretrained weights.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should improve accuracy while keeping your ensemble/TTA inference logic intact. The main likely cause is that almost all checkpoints are being skipped because `_is_compatible_state_dict` compares overlap against the *entire* backbone key count and also fails when checkpoint keys are prefixed (e.g., `convlayer.*`, `fc.*`) relative to your wrapper, so you end up with a near-fallback prediction. I minimally (1) map common checkpoint keyspaces into your wrapper’s keyspace (e.g., `features.* -> convlayer.*`, `fc.* -> fc.*`, `classifier.* -> fc.*` as appropriate), and (2) fix the compatibility test to measure overlap against the keys actually present in the checkpoint (excluding head keys), so valid weights load instead of being skipped. This keeps architecture, TTA set, averaging, and argmax semantics unchanged, but ensures real trained weights are used, which should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os

CANDIDATE_WEIGHT_GLOBS = [
    "../input/densenet201-04-2019data/*.pth",
    "../input/eb7m-seed70/*.pth",
    "/kaggle/input/**/**/*.pth",
    "/kaggle/input/**/*.pth",
    "/kaggle/data/**/**/*.pth",
    "/kaggle/data/**/*.pth",
]

SUPPORTED_NAME_PATTERNS = [
    "resnet18",
    "resnet50",
    "resnet152",
    "resnext101",
    "densenet201",
    "efficientnet-b7",  # still skipped later if efficientnet_pytorch is unavailable
]

pretrained_models_all = []
for g in CANDIDATE_WEIGHT_GLOBS:
    pretrained_models_all.extend(glob.glob(g, recursive=True))
pretrained_models_all = sorted(list(dict.fromkeys(pretrained_models_all)))


def _is_supported_weight_path(p: str) -> bool:
    bn = os.path.basename(p).lower()
    return any(pat in bn for pat in SUPPORTED_NAME_PATTERNS)


pretrained_models = [p for p in pretrained_models_all if _is_supported_weight_path(p)]

print(
    f"{len(pretrained_models)} supported-model candidates found (from {len(pretrained_models_all)} .pth files)."
)
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)[:50]))
    if len(pretrained_models) > 50:
        print("... (truncated)")
else:
    print(
        "No supported pretrained models found in the specified folders. "
        "The notebook will still produce a valid submission (fallback)."
    )



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
class _EfficientNetStub:
    @staticmethod
    def from_name(name: str):
        raise ModuleNotFoundError(
            "efficientnet_pytorch is not installed in this environment; "
            "EfficientNet weights will be skipped."
        )


EfficientNet = _EfficientNetStub



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
CANDIDATE_BASE_DIRS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
BASE_DIR = None
for p in CANDIDATE_BASE_DIRS:
    if os.path.exists(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    BASE_DIR = "/kaggle/data/cassava-leaf-disease-classification"

TEST_PATH = None
for cand in [
    f"{BASE_DIR}/test_images",
    f"{BASE_DIR}/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]:
    if os.path.isdir(cand):
        TEST_PATH = cand
        break

if TEST_PATH is None:
    TEST_PATH = f"{BASE_DIR}/train_images"

sample_sub_path = None
for cand in [
    f"{BASE_DIR}/sample_submission.csv",
    f"{BASE_DIR}/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
]:
    if os.path.isfile(cand):
        sample_sub_path = cand
        break
if sample_sub_path is None:
    raise FileNotFoundError(
        "sample_submission.csv was not found in candidate locations."
    )

sample_sub = pd.read_csv(sample_sub_path)
test_files = sample_sub["image_id"].tolist()

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"sample_sub_path: {sample_sub_path}")
print(f"Number of test images from sample_submission: {len(test_files)}")

_missing = [
    img for img in test_files[:50] if not os.path.isfile(os.path.join(TEST_PATH, img))
]
if len(_missing) > 0:
    raise FileNotFoundError(
        f"TEST_PATH does not contain expected test images (e.g., { _missing[0] }). TEST_PATH={TEST_PATH}"
    )



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
                A.HorizontalFlip(p=1.0),
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
    ]
}



## === cell 10
pass




## === cell 11
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
            x = torch.flatten(x, 1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
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

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

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




## === cell 13
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




## === cell 16
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

            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 17
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix_if_present(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    keys = list(state_dict.keys())
    for pref in prefixes:
        if all(k.startswith(pref) for k in keys):
            return {k[len(pref) :]: v for k, v in state_dict.items()}
    return state_dict


def _cleanup_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        cleaned[nk] = v
    return cleaned


def _remap_state_dict_to_wrapper_space(state: dict, model_name: str) -> dict:
    """
    Minimal, score-directed fix: many checkpoints are saved from the *base torchvision model*
    (keys like 'features.*' or 'fc.*'), while our inference uses wrapper modules
    (keys like 'convlayer.*' and 'fc.*'). We remap common patterns so that valid weights load
    instead of being skipped/partially loaded, improving accuracy without changing architecture.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    remapped = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("convlayer.") or nk.startswith("fc."):
            remapped[nk] = v
            continue

        if model_name.startswith("densenet"):
            if nk.startswith("features."):
                nk = "convlayer." + nk[len("features.") :]
            elif nk.startswith("classifier."):
                nk = "fc." + nk[len("classifier.") :]
        else:
            if nk.startswith("fc."):
                nk = "fc." + nk[len("fc.") :]
            elif nk.startswith(
                ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
            ):
                nk = "convlayer.0." + nk

        remapped[nk] = v

    return remapped


def _is_compatible_state_dict(net: nn.Module, state: dict) -> bool:
    """
    Minimal, score-directed fix: previous compatibility required overlap against the full model
    backbone keycount, which wrongly rejects valid checkpoints (especially when only backbone
    keys exist in the checkpoint). We instead:
      - ignore head keys
      - require a high fraction of *checkpoint backbone keys* to match by name+shape.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return False

    model_sd = net.state_dict()

    def _is_head_key(k: str) -> bool:
        k = str(k)
        return (
            k.startswith("fc.")
            or k.startswith("classifier.")
            or k.startswith("model._fc.")
            or k.endswith(".fc.weight")
            or k.endswith(".fc.bias")
            or k.endswith(".classifier.weight")
            or k.endswith(".classifier.bias")
        )

    ckpt_backbone_keys = [
        k
        for k, v in state.items()
        if (k in model_sd)
        and (not _is_head_key(k))
        and torch.is_tensor(v)
        and torch.is_tensor(model_sd[k])
    ]
    if len(ckpt_backbone_keys) == 0:
        return False

    matched = 0
    for k in ckpt_backbone_keys:
        if tuple(state[k].shape) == tuple(model_sd[k].shape):
            matched += 1
        else:
            return False  # backbone shape mismatch is a hard fail

    return matched >= int(0.90 * len(ckpt_backbone_keys))


probability = []

start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "No pretrained models available; using sample_submission labels as a safe fallback."
    )
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0].lower()

        criterion = nn.CrossEntropyLoss()

        net = None
        BATCH_SIZE = 16
        MODEL_NAME = None

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
            print(
                f"{basename}: efficientnet-b7 skipped (efficientnet_pytorch not installed)."
            )
            continue
        else:
            continue

        print(f"{basename}: {MODEL_NAME}  (weights: {pretrained_model})")

        ckpt = torch.load(pretrained_model, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_prefix_if_present(state)
        state = _cleanup_state_dict_keys(state)
        state = _remap_state_dict_to_wrapper_space(state, MODEL_NAME)

        if not _is_compatible_state_dict(net, state):
            print(
                f"{basename}: skipped (checkpoint incompatible with {MODEL_NAME} after key remap / backbone shape check)"
            )
            del net
            continue

        net.load_state_dict(state, strict=True)

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            dataset = {
                "test": TestDataset(df_test, transform=transform_),
            }
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=(device == "cuda"),
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(probability) > 0:
        mean_proba = np.stack(probability, axis=0).mean(axis=0)
        df_test["mean"] = mean_proba.argmax(axis=1)
    else:
        print(
            "All discovered models were skipped/unsupported; will fall back to sample_submission labels."
        )

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 18
if "mean" in df_test.columns:
    pred_map = dict(zip(df_test["image_id"].values, df_test["mean"].astype(int).values))
    sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(1).astype(int)
    df_test = sample_sub
else:
    df_test = sample_sub



## === cell 19
df_test.head()



## === cell 20
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
