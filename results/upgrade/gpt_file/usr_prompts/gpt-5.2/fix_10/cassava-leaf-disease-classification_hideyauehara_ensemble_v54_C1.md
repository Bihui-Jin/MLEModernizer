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

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the dependency on the missing `efficientnet_pytorch` package by switching EfficientNet loading to torchvision’s built-in EfficientNet-B7 while keeping the same “efficientnet-b7” branch logic. I also fix dataset path detection so it always points to the real Kaggle input folder (instead of falling back to non-existent local `data/train_images`). Finally, I update the Albumentations `RandomResizedCrop` call to the v2 API so transforms build correctly, and I harden the ensemble aggregation so it can’t crash with an axis error and always writes a valid `submission.csv` matching `sample_submission.csv` order.'
- What this solution (achieved 0.05531) has done: 'Your very low score is most consistent with a silent weight-mismatch bug: the wrapper `FinalLayerMixupModelEN` currently replaces the EfficientNet classifier with a new random `Linear`, so when you load the `.pth` the head often doesn’t match (or you end up not using the loaded head), producing near-random predictions. I make the EfficientNet wrapper preserve and use the real torchvision EfficientNet-B7 classifier (`model.classifier`) exactly, so loaded checkpoints are applied to the correct parameters and inference uses the intended head. I also fix the dataset path selection so it always predicts on `test_images` (not `train_images`) regardless of run type, because submitting predictions for the wrong set can crater accuracy. These changes keep the same overall logic (same ensemble loop, same transforms, same aggregation) and should move the score substantially toward your 0.8936 target.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is so far below the target (0.8936) that it strongly indicates the pretrained checkpoints are not being applied to the model you actually run at inference. I make a minimal, architecture-preserving fix to the EfficientNet wrapper so it does **not** replace the classifier layer (which breaks checkpoint compatibility) and so the forward path matches torchvision EfficientNet’s native forward. I also simplify/robustify checkpoint loading for EfficientNet by trying a small set of common key-mapping fallbacks (e.g., `model.` prefix, `_fc` vs `classifier.1`) without changing any modeling logic. These changes should move predictions from near-random to “as-trained”, which is the smallest legitimate step likely to bring accuracy much closer to your target.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random, so the smallest realistic way to move toward the 0.8936 target is to ensure the *exact same classifier head shape* as used during training is created before loading checkpoints. Right now EfficientNet-B7 is instantiated with the default 1000-class classifier, so strict loading usually fails (or you end up effectively running with an untrained head), crushing accuracy. I make a minimal, architecture-preserving fix: replace EfficientNet’s `classifier[1]` with a `Linear(..., 5)` **before** loading weights, and extend the checkpoint key-mapping to support both `classifier.weight/bias` and `classifier.1.*` conventions. This keeps your ensemble, transforms, inference loop, and aggregation intact, but makes the loaded weights actually apply to the model you run.'
- What this solution (achieved 0.05531) has done: 'Your score is near-random, so the smallest realistic move toward the 0.8936 target is to make sure the EfficientNet-B7 checkpoints load into the *exact same parameter names and head shape* that were used when training. I (1) set EfficientNet’s classifier to 5 classes before loading weights, and (2) expand the checkpoint key-mapping to also handle the common `classifier.0.*` (single Linear) vs `classifier.1.*` (Dropout+Linear) mismatch, which otherwise leaves the head randomly initialized. I also switch test file discovery to always follow `sample_submission.csv` ordering (instead of `os.listdir`), which prevents any accidental mismatch/duplication and stabilizes inference without changing the modeling logic. Everything else (ensemble loop, transforms/TTAs, aggregation, submission writing) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your score is near-random, so the smallest likely fix toward the 0.8936 target is to ensure the EfficientNet-B7 head shape and parameter names match what the checkpoints expect before loading them. I (1) set EfficientNet’s classifier to 5 classes in a way that supports both torchvision (`classifier.1`) and efficientnet_pytorch-style (`_fc`) checkpoints, and (2) extend the key-remapping so `_fc.*` can load into either `classifier.1.*` or `classifier.*` depending on the instantiated head. I also make the test image list strictly follow `sample_submission.csv` (and filter to existing files) to prevent any accidental image/order mismatch that can crater accuracy. Core inference/ensemble logic, transforms, and aggregation remain unchanged.'
- What this solution (achieved 0.05531) has done: 'Your score is near-random relative to the target, so the smallest plausible fix is to ensure the EfficientNet inference path matches the checkpoints’ original architecture: right now your EfficientNet wrapper only replaces the classifier but does not expose `_fc`, which many Cassava EfficientNet-B7 checkpoints use, so strict loading can silently miss the head and keep random weights. I minimally augment `FinalLayerMixupModelEN` to provide a compatible `_fc` alias to the final Linear layer (without changing the forward computation), and update the key-remapping so `_fc.*` can be loaded directly and reliably. I also make the load routine validate that the classifier weights actually changed from initialization (sanity check) and fall back to the most compatible mapping if not, which is directly aimed at moving accuracy toward your target. Everything else (ensemble loop, TTAs, transforms, aggregation, submission writing/pathing) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random, so the smallest legitimate move toward the 0.8936 target is to ensure the non-EfficientNet wrappers (ResNet/DenseNet) also create a 5-class head *before* loading checkpoints; right now they always create a fresh random `Linear`, so strict loading often fails or leaves the head random, crushing accuracy. I minimally modify `FinalLayerMixupModel` and `FinalLayerMixupModelDenseNet` to *reuse and reshape the original classifier layer* (`model.fc` / `model.classifier`) instead of defining a separate `self.fc`, keeping the same forward/test logic. I also make checkpoint loading for non-EfficientNet robust to common key prefixes (`model.`, `module.`) similar to your EfficientNet loader, without changing ensemble/TTAs/transforms. This should move inference from random outputs to the actual trained heads and bring accuracy much closer to your target while preserving core logic.'
- What this solution (achieved 0.05531) has done: 'Your score is near-random, so the smallest likely way to move toward the 0.8936 target is to ensure *all* model wrappers expose the exact same classifier parameter names/shapes that the checkpoints were trained with, so loading doesn’t silently miss the head. I minimally adjust the ResNet and DenseNet wrappers to keep the original `model.fc` / `model.classifier` modules (rather than creating a separate `self.fc` that changes key names), and I route forward through the underlying `model` so checkpoint keys match naturally. I also make non-EfficientNet checkpoint loading robust to common head-key variants (`fc.*` vs `model.fc.*`, etc.) while keeping strict loading when possible. These changes preserve the same model families, transforms, TTA loop, and aggregation, but should turn predictions from random to “as-trained”, materially increasing accuracy toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = (
    glob.glob(f"../input/efntb7/efficientnet-b7m_SEED70/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7m_SEED72/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7m_SEED73/*.pth")
    + glob.glob(f"../input/efntb7/efficientnet-b7sl_SEED71.pretrained/*.pth")
)

print(f"{len(pretrained_models)} models found.")
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7


class EfficientNet:
    @staticmethod
    def from_name(name: str):
        if name != "efficientnet-b7":
            raise ValueError(
                f"Only efficientnet-b7 supported in this environment, got: {name}"
            )
        m = efficientnet_b7(weights=None)
        return m




## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
KAGGLE_BASE = Path("/kaggle/input/cassava-leaf-disease-classification")
ALT_KAGGLE_BASE = Path("../input/cassava-leaf-disease-classification")

if KAGGLE_BASE.exists():
    BASE_DIR = str(KAGGLE_BASE)
elif ALT_KAGGLE_BASE.exists():
    BASE_DIR = str(ALT_KAGGLE_BASE)
else:
    BASE_DIR = "data"

test_dir = Path(f"{BASE_DIR}/test_images")
train_dir = Path(f"{BASE_DIR}/train_images")
if test_dir.exists():
    TEST_PATH = str(test_dir)
    print("Using test_images for submission inference.")
else:
    TEST_PATH = str(train_dir)
    print(
        "WARNING: test_images not found; falling back to train_images (this will not score well on Kaggle)."
    )

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")



## === cell 7
sample_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sub0 = pd.read_csv(sample_path)[["image_id"]].copy()
    exists_mask = sub0["image_id"].map(lambda x: (Path(TEST_PATH) / x).exists())
    if not bool(exists_mask.all()):
        missing = int((~exists_mask).sum())
        print(
            f"WARNING: {missing} images from sample_submission not found under TEST_PATH; filtering them out."
        )
        sub0 = sub0.loc[exists_mask].reset_index(drop=True)
    df_test = sub0
else:
    test_files = sorted(os.listdir(TEST_PATH))
    df_test = pd.DataFrame(test_files, columns=["image_id"])

df_test["label"] = 1
print("df_test shape:", df_test.shape)



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
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    p=1.0,
                ),
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

        CHANGE (score fix): Keep the classifier as model.fc (do not create a separate self.fc module)
        so checkpoint keys like 'fc.weight'/'fc.bias' match and load cleanly; forward uses model(x).
        """
        super(FinalLayerMixupModel, self).__init__()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

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

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=inputs.device)

        x1 = inputs
        x2 = inputs[index]

        mixed_x = lam * x1 + (1 - lam) * x2
        outputs = self.model(mixed_x)

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

        CHANGE (score fix): Keep classifier as model.classifier (no separate self.fc),
        so checkpoint keys 'classifier.*' match; forward uses model(x).
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

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

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels), device=inputs.device)

        x1 = inputs
        x2 = inputs[index]
        mixed_x = lam * x1 + (1 - lam) * x2

        outputs = self.model(mixed_x)

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
        if not hasattr(model, "classifier"):
            raise ValueError("Expected torchvision EfficientNet with .classifier.")

        if isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) >= 2 and isinstance(
                model.classifier[-1], nn.Linear
            ):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                self._fc = model.classifier[-1]  # alias
            elif len(model.classifier) == 1 and isinstance(
                model.classifier[0], nn.Linear
            ):
                in_features = model.classifier[0].in_features
                model.classifier[0] = nn.Linear(in_features, num_classes)
                self._fc = model.classifier[0]  # alias
            else:
                last_linear = None
                for m in reversed(model.classifier):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break
                if last_linear is None:
                    raise ValueError(
                        "Could not infer EfficientNet classifier in_features."
                    )
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(last_linear.in_features, num_classes),
                )
                self._fc = model.classifier[-1]  # alias
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            self._fc = model.classifier  # alias
        else:
            raise ValueError(
                f"Unsupported EfficientNet classifier type: {type(model.classifier)}"
            )

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
            raise FileNotFoundError(f"Could not read image: {TEST_PATH}/{image_id}")
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

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = net(inputs, False, "test")
                probability.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
def _extract_state_dict(state):
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def _strip_known_prefixes(sd: dict):
    if not isinstance(sd, dict):
        return sd
    prefixes = ["model.", "module.", "net.", "backbone."]
    out_candidates = [sd]
    for p in prefixes:
        if any(k.startswith(p) for k in sd.keys()):
            out_candidates.append(
                {k[len(p) :]: v for k, v in sd.items() if k.startswith(p)}
            )
    out_candidates = sorted(out_candidates, key=lambda d: len(d), reverse=True)
    return out_candidates[0]


def _head_weight_signature(net_en: FinalLayerMixupModelEN):
    with torch.no_grad():
        sd = net_en.model.state_dict()
        for k in ["classifier.1.weight", "classifier.0.weight", "classifier.weight"]:
            if k in sd:
                w = sd[k].detach().float().cpu()
                return float(w.mean().item()), float(w.std().item())
    return None


def _try_load_efficientnet_state_dict(net_en: FinalLayerMixupModelEN, state):
    """
    CHANGE (score fix): Robustly load EfficientNet checkpoints with differing key styles.
    Now also supports direct _fc alias loading by keeping _fc.* keys (mapped via alias module),
    and validates that classifier weights actually changed vs initialization.
    """
    sd = _extract_state_dict(state)
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint does not contain a valid state dict.")

    init_sig = _head_weight_signature(net_en)

    candidates = [sd]

    for prefix in ["model.", "module.", "net.", "backbone."]:
        if any(k.startswith(prefix) for k in sd.keys()):
            candidates.append(
                {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
            )

    def remap_head_keys(d):
        out = dict(d)

        model_sd_keys = set(net_en.model.state_dict().keys())
        expects_classifier_1 = (
            "classifier.1.weight" in model_sd_keys
            and "classifier.1.bias" in model_sd_keys
        )
        expects_classifier_0 = (
            "classifier.0.weight" in model_sd_keys
            and "classifier.0.bias" in model_sd_keys
        )
        expects_classifier_plain = (
            "classifier.weight" in model_sd_keys and "classifier.bias" in model_sd_keys
        )

        def map_to_expected(w_key, b_key):
            if expects_classifier_1:
                out["classifier.1.weight"] = out.pop(w_key)
                out["classifier.1.bias"] = out.pop(b_key)
            elif expects_classifier_0:
                out["classifier.0.weight"] = out.pop(w_key)
                out["classifier.0.bias"] = out.pop(b_key)
            elif expects_classifier_plain:
                out["classifier.weight"] = out.pop(w_key)
                out["classifier.bias"] = out.pop(b_key)
            else:
                out["classifier.1.weight"] = out.pop(w_key)
                out["classifier.1.bias"] = out.pop(b_key)

        if "_fc.weight" in out and "_fc.bias" in out:
            map_to_expected("_fc.weight", "_fc.bias")

        if (
            "classifier.weight" in out
            and "classifier.bias" in out
            and not expects_classifier_plain
        ):
            w = out.pop("classifier.weight")
            b = out.pop("classifier.bias")
            out["__tmp_w__"] = w
            out["__tmp_b__"] = b
            map_to_expected("__tmp_w__", "__tmp_b__")
            out.pop("__tmp_w__", None)
            out.pop("__tmp_b__", None)

        if (
            "classifier.0.weight" in out
            and "classifier.0.bias" in out
            and not expects_classifier_0
        ):
            w = out.pop("classifier.0.weight")
            b = out.pop("classifier.0.bias")
            out["__tmp_w__"] = w
            out["__tmp_b__"] = b
            map_to_expected("__tmp_w__", "__tmp_b__")
            out.pop("__tmp_w__", None)
            out.pop("__tmp_b__", None)

        if (
            "classifier.1.weight" in out
            and "classifier.1.bias" in out
            and not expects_classifier_1
        ):
            w = out.pop("classifier.1.weight")
            b = out.pop("classifier.1.bias")
            out["__tmp_w__"] = w
            out["__tmp_b__"] = b
            map_to_expected("__tmp_w__", "__tmp_b__")
            out.pop("__tmp_w__", None)
            out.pop("__tmp_b__", None)

        return out

    candidates = candidates + [remap_head_keys(c) for c in candidates]

    last_err = None
    for i, cand in enumerate(candidates):
        try:
            net_en.model.load_state_dict(cand, strict=True)
            loaded_sig = _head_weight_signature(net_en)
            if (
                init_sig is not None
                and loaded_sig is not None
                and np.allclose(init_sig, loaded_sig, rtol=0, atol=1e-7)
            ):
                continue
            return True, f"loaded_strict_candidate_{i}"
        except Exception as e:
            last_err = e

    net_en.model.load_state_dict(candidates[0], strict=False)
    return False, f"loaded_nonstrict_due_to: {type(last_err).__name__}: {last_err}"


def _try_load_generic_state_dict(net: nn.Module, state):
    """
    CHANGE (score fix): For non-EfficientNet models, try a few minimal key remaps so the
    5-class head loads even if the wrapper/model nesting changed key names.
    Keeps strict=True when a compatible candidate exists.
    """
    sd0 = _strip_known_prefixes(_extract_state_dict(state))
    if not isinstance(sd0, dict):
        raise ValueError("Checkpoint does not contain a valid state dict.")

    candidates = [sd0]

    if any(k.startswith("model.") for k in sd0.keys()):
        candidates.append({k[len("model.") :]: v for k, v in sd0.items()})

    if hasattr(net, "model") and isinstance(getattr(net, "model"), nn.Module):
        candidates.append({f"model.{k}": v for k, v in sd0.items()})

    if hasattr(net, "model") and any(k.startswith("fc.") for k in sd0.keys()):
        candidates.append(
            {("model." + k if k.startswith("fc.") else k): v for k, v in sd0.items()}
        )

    last_err = None
    for i, cand in enumerate(candidates):
        try:
            net.load_state_dict(cand, strict=True)
            return True, f"loaded_strict_candidate_{i}"
        except Exception as e:
            last_err = e

    net.load_state_dict(candidates[0], strict=False)
    return False, f"loaded_nonstrict_due_to: {type(last_err).__name__}: {last_err}"




## === cell 18
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained .pth models found in ../input/efntb7/. Falling back to constant predictions."
    )
else:
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
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            raise SystemExit(1)

        print(f"{basename}: {MODEL_NAME}")

        state = torch.load(pretrained_model, map_location="cpu")

        if MODEL_NAME == "efficientnet-b7":
            ok, msg = _try_load_efficientnet_state_dict(net, state)
            print(f"EfficientNet checkpoint load: {msg}")
        else:
            ok, msg = _try_load_generic_state_dict(net, state)
            print(f"Generic checkpoint load: {msg}")

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
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
                    persistent_workers=False,
                ),
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        if device == "cuda":
            torch.cuda.empty_cache()

if len(probability) == 0:
    df_test["mean"] = 0
else:
    probs = np.stack(probability, axis=0)  # (n_preds, n_samples, n_classes)
    df_test["mean"] = probs.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
sample_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sub = pd.read_csv(sample_path)
    pred_map = dict(zip(df_test["image_id"].tolist(), df_test["mean"].tolist()))
    sub["label"] = sub["image_id"].map(pred_map).fillna(0).astype(int)
    df_test = sub
else:
    if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
        df_test = pd.DataFrame(
            {
                "image_id": df_test["image_id"].iloc[:1],
                "label": [int(df_test["mean"].iloc[0])],
            }
        )
    else:
        df_test["label"] = df_test["mean"].astype(int)



## === cell 20
df_test.head()



## === cell 21
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(pd.read_csv("submission.csv").head())
